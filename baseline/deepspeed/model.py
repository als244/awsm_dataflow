# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch.cuda.nvtx as nvtx # Import NVTX

from torch.utils.checkpoint import checkpoint
import deepspeed

from liger_kernel.transformers.rms_norm import LigerRMSNorm

from liger_kernel.transformers.llama4_rope import liger_llama4_text_rotary_pos_emb as LigerRope

from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from chunked_ce import chunked_linear_cross_entropy

from scattermoe.mlp import MLP, GLUMLP

from attention import do_attention

from select_bins import select_bins

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

@dataclass
class ModelArgs:
    """Configuration for the model."""
    # Architecture (matches config keys directly)
    vocab_size: int = 128256
    n_layers: int = 32
    d_model: int = 4096
    head_dim: int = 128
    n_heads: int = 32
    n_kv_heads: int = 8
    expert_dim: int = 14336
    num_shared_experts: int = 1
    num_routed_experts: int = 0
    top_k: int = 0
    is_causal: bool = True

    # Datatypes (populated from config "datatypes" dict)
    embed_dtype: torch.dtype = torch.bfloat16
    head_proj_dtype: torch.dtype = torch.bfloat16
    attn_proj_dtype: torch.dtype = torch.bfloat16
    expert_proj_dtype: torch.dtype = torch.bfloat16
    router_dtype: torch.dtype = torch.bfloat16
    norm_dtype: torch.dtype = torch.bfloat16
    residual_dtype: torch.dtype = torch.bfloat16

    # Defaults not in config
    expert_mlp_type: str = "swiglu"
    rope_theta: float = 500000
    rms_norm_epsilon: float = 1e-5
    rand_seed: int = 42

    @classmethod
    def from_config(cls, cfg: dict) -> "ModelArgs":
        """Create ModelArgs from a config dict (e.g. one entry from configs.json)."""
        datatypes = cfg.get("datatypes", {})
        return cls(
            vocab_size=cfg.get("vocab_size", cls.vocab_size),
            n_layers=cfg.get("n_layers", cls.n_layers),
            d_model=cfg.get("d_model", cls.d_model),
            head_dim=cfg.get("head_dim", cls.head_dim),
            n_heads=cfg.get("n_heads", cls.n_heads),
            n_kv_heads=cfg.get("n_kv_heads", cls.n_kv_heads),
            expert_dim=cfg.get("expert_dim", cls.expert_dim),
            num_shared_experts=cfg.get("num_shared_experts", cls.num_shared_experts),
            num_routed_experts=cfg.get("num_routed_experts", cls.num_routed_experts),
            top_k=cfg.get("top_k", cls.top_k),
            is_causal=cfg.get("is_causal", cls.is_causal),
            embed_dtype=_DTYPE_MAP.get(datatypes.get("embed", "bfloat16"), torch.bfloat16),
            head_proj_dtype=_DTYPE_MAP.get(datatypes.get("head_proj", "bfloat16"), torch.bfloat16),
            attn_proj_dtype=_DTYPE_MAP.get(datatypes.get("attn_proj", "bfloat16"), torch.bfloat16),
            expert_proj_dtype=_DTYPE_MAP.get(datatypes.get("expert_proj", "bfloat16"), torch.bfloat16),
            router_dtype=_DTYPE_MAP.get(datatypes.get("router", "bfloat16"), torch.bfloat16),
            norm_dtype=_DTYPE_MAP.get(datatypes.get("norm", "bfloat16"), torch.bfloat16),
            residual_dtype=_DTYPE_MAP.get(datatypes.get("residual", "bfloat16"), torch.bfloat16),
        )

@dataclass
class SwiGLUConfig:
    """Configuration for the LigerSwigluMLP"""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    hidden_act: str = "silu"


def precompute_theta_pos_frequencies(head_dim: int, max_seq_len: int, theta: float, device = "cpu"):
    """Precomputes the rotary frequencies for RoPE."""
    theta_base = torch.tensor(theta, device=device)
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex6

class Attention(nn.Module):
    """Multi-Head Attention updated to use FlashAttention."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.n_kv_heads = args.n_kv_heads
        self.d_model = args.d_model
        self.dtype = args.attn_proj_dtype
        self.wq = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False, dtype=self.dtype)
        self.wk = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False, dtype=self.dtype)
        self.wv = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False, dtype=self.dtype)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False, dtype=self.dtype)

    def forward(self, x: torch.Tensor, freqs):
        nvtx.range_push("Attention") # NVTX Start
        
        batch_size, seq_len, _ = x.shape
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
       
        xq, xk = LigerRope(xq, xk, freqs)

        attn_result = do_attention(xq, xk, xv, causal=True, deterministic=True)

        attn_result = attn_result.view(batch_size, seq_len, -1)
        
        attn_out_proj = self.wo(attn_result)
        
        nvtx.range_pop() # NVTX End
        return attn_out_proj

class DenseFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        swiglu_config = SwiGLUConfig()
        swiglu_config.hidden_size = args.d_model
        swiglu_config.intermediate_size = args.expert_dim
        
        self.swiglu = LigerSwiGLUMLP(swiglu_config)
        

    def forward(self, x: torch.Tensor):
        nvtx.range_push("FeedForward") # NVTX Start
        
        result = self.swiglu(x)
        
        nvtx.range_pop() # NVTX End
        return result


class SparseFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.top_k = args.top_k
        self.router_dtype = args.router_dtype
        self.d_model = args.d_model
        self.expert_dim = args.expert_dim
        self.num_routed_experts = args.num_routed_experts
        self.router = nn.Linear(self.d_model, self.num_routed_experts, bias=False, dtype=self.router_dtype)
        self.moe_layer = GLUMLP(input_size=self.d_model, hidden_size=self.expert_dim, activation=nn.SiLU(), num_experts=self.num_routed_experts, top_k=self.top_k)


    def forward(self, x: torch.Tensor):
        nvtx.range_push("MoE") # NVTX Start

        batch_size, sequence_length, hidden_dim = x.shape

        x = x.view(-1, hidden_dim)
        routed_x = self.router(x)

        routing_weights = F.softmax(routed_x, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(x.dtype)

        final_hidden_states = self.moe_layer(x, routing_weights, selected_experts)

        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)

        nvtx.range_pop() # NVTX End
        return final_hidden_states

class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id):
        super().__init__()
        self.d_model = args.d_model
        self.layer_id = layer_id
        self.attention = Attention(args)

        if args.num_routed_experts == 0:
            self.feed_forward = DenseFeedForward(args)
        else:
            self.feed_forward = SparseFeedForward(args)
        
        self.attention_norm = LigerRMSNorm(self.d_model, eps=args.rms_norm_epsilon)
        self.ffn_norm = LigerRMSNorm(self.d_model, eps=args.rms_norm_epsilon)

    def forward(self, x: torch.Tensor, freqs):
        nvtx.range_push("Attention Sub-Block")
        h = x + self.attention(x, freqs)
        nvtx.range_pop()
        
        nvtx.range_push("FeedForward Sub-Block")
        out = h + self.feed_forward(self.ffn_norm(h))
        nvtx.range_pop()
        
        return out

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        
        self.embed_dtype = args.embed_dtype
        self.head_proj_dtype = args.head_proj_dtype

        self.d_model = args.d_model
        self.head_dim = args.head_dim
        self.rope_theta = args.rope_theta

        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(self.vocab_size, self.d_model, dtype=self.embed_dtype)
        self.layers = nn.ModuleList([DecoderBlock(args, i) for i in range(self.n_layers)])
        self.norm = LigerRMSNorm(self.d_model, eps=args.rms_norm_epsilon)
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False, dtype=self.head_proj_dtype)

        self.max_seq_len = 2 ** 20
        self.freqs_complex = precompute_theta_pos_frequencies(self.head_dim, self.max_seq_len, self.rope_theta)
        
        self.loss_fn = LigerFusedLinearCrossEntropyLoss()

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor, save_act_layer_frac = 0.0):
        batch_size, seq_len = tokens.shape
        
        act_layers_saved = select_bins(self.n_layers, save_act_layer_frac)

        nvtx.range_push("Token Embeddings")
        h = self.tok_embeddings(tokens)
        nvtx.range_pop()

        freqs = self.freqs_complex[:seq_len].to(h.device)

        nvtx.range_push("Decoder Layers")
        for i, layer in enumerate(self.layers):
            nvtx.range_push(f"Layer {i}")
            if i in act_layers_saved:
                h = layer(h, freqs)
            else:
                #h = checkpoint(layer, h, freqs, use_reentrant=False) # Call the layer directly
                h = deepspeed.checkpointing.checkpoint(layer, h, freqs)
            nvtx.range_pop()
        nvtx.range_pop()
            
        nvtx.range_push("Final Norm")
        h = self.norm(h)
        nvtx.range_pop()


        nvtx.range_push("Output Projection and Cross Entropy Loss")
        #loss = self.loss_fn(self.output.weight, h.view(-1, self.d_model), labels.view(-1))
        loss = chunked_linear_cross_entropy(h.view(-1, self.d_model), self.output.weight, labels.view(-1)) 
        nvtx.range_pop()
        
        return loss