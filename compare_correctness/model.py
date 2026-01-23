# model.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import time
import pickle

try:
    from flash_attn_interface import flash_attn_varlen_func
except:
    from flash_attn import flash_attn_varlen_func

@dataclass
class ModelArgs:
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: Optional[int] = 4
    head_dim: int = 128
    vocab_size: int = 50257
    expert_dim: int = 8192
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0

SAVED_ACT_GRADS = {}

def save_grad(name):
    def hook(grad):
        # clone() and detach() are crucial to prevent memory leaks
        # and to ensure you have the data even if the graph is freed.
        SAVED_ACT_GRADS[name] = grad.detach().clone() 
    return hook


class SeqlensInfo:
    def __init__(self, seqlens_q_np, device, seqlens_k_np=None):

        self.cu_seqlens_q = np.zeros(len(seqlens_q_np) + 1, dtype=np.int32)
        self.cu_seqlens_q[1:] = np.cumsum(seqlens_q_np)
        self.cu_seqlens_q = torch.from_numpy(self.cu_seqlens_q).to(device)
        if seqlens_k_np is not None:
            self.cu_seqlens_k = np.zeros(len(seqlens_k_np) + 1, dtype=np.int32)
            self.cu_seqlens_k[1:] = np.cumsum(seqlens_k_np)
            self.cu_seqlens_k = torch.from_numpy(self.cu_seqlens_k).to(device)
        else:
            self.cu_seqlens_k = self.cu_seqlens_q.clone()

        self.max_seqlen_q = np.max(seqlens_q_np)

        if seqlens_k_np is not None:
            self.max_seqlen_k = np.max(seqlens_k_np)
        else:
            self.max_seqlen_k = self.max_seqlen_q

        self.total_tokens = np.sum(seqlens_q_np)

        seq_positions = []

        for i in range(len(seqlens_q_np)):
            seq_positions += list(range(0, seqlens_q_np[i]))

        self.seq_positions = torch.tensor(seq_positions, device="cpu").long()

        

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    seq_positions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies RoPE to flattened/packed token sequences.

    Args:
        xq: (total_tokens, n_heads, head_dim)
        xk: (total_tokens, n_kv_heads, head_dim)
        freqs_cis: (max_seq_len, head_dim // 2) - Precomputed cache
        seq_positions: (total_tokens,) - 1D tensor of position IDs
    """
    # 1. View as complex numbers
    # Input: (total_tokens, n_heads, head_dim)
    # Reshape: (total_tokens, n_heads, head_dim // 2, 2)
    # Complex View: (total_tokens, n_heads, head_dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 2. Select frequencies for specific token positions
    # freqs_cis: (max_len, head_dim // 2)
    # seq_positions: (total_tokens,)
    # Resulting freqs: (total_tokens, head_dim // 2)
    freqs = freqs_cis[seq_positions]

    # 3. Reshape for broadcasting across heads
    # We need to match xq_ shape: (total_tokens, n_heads, head_dim // 2)
    # Current freqs: (total_tokens, head_dim // 2)
    # Unsqueeze dim 1 to add the "heads" dimension: (total_tokens, 1, head_dim // 2)
    freqs = freqs.unsqueeze(1)
    
    # 4. Apply rotation (complex multiplication) & Flatten back to real
    # Result is (total_tokens, n_heads, head_dim // 2, 2) -> flatten last two dims
    xq_out = torch.view_as_real(xq_ * freqs).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, layer_id, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        seqlens_info: SeqlensInfo,
        step_num: int,
    ):
        
        total_tokens, model_dim = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis, seqlens_info.seq_positions)
            
        # # These hooks are registered on the tensors for THIS forward pass only.
        # # They disappear when the graph is freed, so they won't accumulate.
        # if self.layer_id == 15:
        #     xq.register_hook(save_grad('dq'))
        #     xk.register_hook(save_grad('dk'))
        #     xv.register_hook(save_grad('dv'))
        #     torch.save(xq, "fwd_xq.pt")
        #     torch.save(xk, "fwd_xk.pt")
        #     torch.save(xv, "fwd_xv.pt")
            

        output = flash_attn_varlen_func(
            xq, xk, xv,
            seqlens_info.cu_seqlens_q, seqlens_info.cu_seqlens_k,
            seqlens_info.max_seqlen_q, seqlens_info.max_seqlen_k,
            deterministic=True,
            causal=True
        )

        # if self.layer_id == 15:
        #     output.register_hook(save_grad('d_attn_out'))


        output = output.view(-1, self.n_local_heads * self.head_dim)

        final_out = self.wo(output)

        return final_out


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        expert_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, expert_dim, bias=False)
        self.w2 = nn.Linear(expert_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, expert_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(layer_id, args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            expert_dim=args.expert_dim,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        seqlens_info: SeqlensInfo,
        step_num: int,
    ):
        
        norm = self.attention_norm(x)

        h = x + self.attention(norm, freqs_cis, seqlens_info, step_num)

        ffn_norm = self.ffn_norm(h)

        out = h + self.feed_forward(ffn_norm)

        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        params.max_seq_len = 2 ** 20

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )


    def forward(self, tokens: torch.Tensor, seqlens_info: SeqlensInfo, step_num: int):
        
        inp_device = tokens.device

        total_tokens = tokens.shape[0]

        h = self.tok_embeddings(tokens)

        self.freqs_cis = self.freqs_cis.to(h.device)
        
        for layer in self.layers:
            h = layer(h, self.freqs_cis, seqlens_info, step_num)
        
        h = self.norm(h)
        output = self.output(h)

        return output



    def load_model_weights(self, model_path: dict):


        self.tok_embeddings.weight.data = torch.load(model_path + "/embed/w_tok_embeddings.pt").to(self.tok_embeddings.weight.device)
        for i in range(self.n_layers):
            self.layers[i].attention_norm.weight.data = torch.load(model_path + f"/layers/{i}/w_attn_norm.pt").to(self.layers[i].attention_norm.weight.device)
            self.layers[i].attention.wq.weight.data = torch.load(model_path + f"/layers/{i}/w_q.pt").T.to(self.layers[i].attention.wq.weight.device)
            self.layers[i].attention.wk.weight.data = torch.load(model_path + f"/layers/{i}/w_k.pt").T.to(self.layers[i].attention.wk.weight.device)
            self.layers[i].attention.wv.weight.data = torch.load(model_path + f"/layers/{i}/w_v.pt").T.to(self.layers[i].attention.wv.weight.device)
            self.layers[i].attention.wo.weight.data = torch.load(model_path + f"/layers/{i}/w_o.pt").T.to(self.layers[i].attention.wo.weight.device)
            self.layers[i].ffn_norm.weight.data = torch.load(model_path + f"/layers/{i}/w_ffn_norm.pt").to(self.layers[i].ffn_norm.weight.device)
            self.layers[i].feed_forward.w1.weight.data = torch.load(model_path + f"/layers/{i}/w_1.pt").T.to(self.layers[i].feed_forward.w1.weight.device)
            self.layers[i].feed_forward.w2.weight.data = torch.load(model_path + f"/layers/{i}/w_2.pt").T.to(self.layers[i].feed_forward.w2.weight.device)
            self.layers[i].feed_forward.w3.weight.data = torch.load(model_path + f"/layers/{i}/w_3.pt").T.to(self.layers[i].feed_forward.w3.weight.device)
        
        self.norm.weight.data = torch.load(model_path + "/head/w_final_norm.pt").to(self.norm.weight.device)
        self.output.weight.data = torch.load(model_path + "/head/w_head_proj.pt").T.to(self.output.weight.device)
            
            
