# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch.cuda.nvtx as nvtx # Import NVTX

from torch.utils.checkpoint import checkpoint
import deepspeed

from quack.rmsnorm import QuackRMSNorm

from liger_kernel.transformers.llama4_rope import liger_llama4_text_rotary_pos_emb as LigerRope

from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from quack.linear_cross_entropy import chunked_linear_cross_entropy

from sonicmoe import MoE, KernelBackendMoE
from sonicmoe.enums import ActivationType

from attention import do_attention

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
    moe_kernel_backend: str = "sonicmoe"
    moe_weight_init_std: float = 0.02
    moe_add_bias: bool = False

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


# Map string activation types to SonicMoE ActivationType enum
_ACTIVATION_TYPE_MAP = {
    "swiglu": ActivationType.SWIGLU,
    "geglu": ActivationType.GEGLU,
    "reglu": ActivationType.REGLU,
    "gelu": ActivationType.GELU,
    "relu": ActivationType.RELU,
    "silu": ActivationType.SILU,
}

# Map string kernel backend to SonicMoE KernelBackendMoE enum
_KERNEL_BACKEND_MAP = {
    "sonicmoe": KernelBackendMoE.sonicmoe,
    "scattermoe": KernelBackendMoE.scattermoe,
    "torch": KernelBackendMoE.torch,
}


class SparseFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.kernel_backend = _KERNEL_BACKEND_MAP.get(
            args.moe_kernel_backend, KernelBackendMoE.sonicmoe
        )

        activation_type = _ACTIVATION_TYPE_MAP.get(
            args.expert_mlp_type, ActivationType.SWIGLU
        )

        self.moe_layer = MoE(
            num_experts=args.num_routed_experts,
            num_experts_per_tok=args.top_k,
            hidden_size=args.d_model,
            intermediate_size=args.expert_dim,
            activation_function=activation_type,
            add_bias=args.moe_add_bias,
            std=args.moe_weight_init_std,
        )

    def forward(self, x: torch.Tensor):
        nvtx.range_push("MoE") # NVTX Start

        output, _aux_loss = self.moe_layer(
            x, kernel_backend_moe=self.kernel_backend
        )

        nvtx.range_pop() # NVTX End
        return output


class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id):
        super().__init__()
        self.d_model = args.d_model
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.is_sparse = args.num_routed_experts > 0

        if not self.is_sparse:
            self.feed_forward = DenseFeedForward(args)
        else:
            self.feed_forward = SparseFeedForward(args)
        
        self.attention_norm = QuackRMSNorm(self.d_model, eps=args.rms_norm_epsilon)
        self.ffn_norm = QuackRMSNorm(self.d_model, eps=args.rms_norm_epsilon)

    def forward(self, x: torch.Tensor, freqs):
        nvtx.range_push("Attention Sub-Block")
        h = x + self.attention(self.attention_norm(x), freqs)
        nvtx.range_pop()
        
        nvtx.range_push("FeedForward Sub-Block")
        ffn_input = self.ffn_norm(h)
        ffn_output = self.feed_forward(ffn_input)
        out = h + ffn_output
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
        self.norm = QuackRMSNorm(self.d_model, eps=args.rms_norm_epsilon)
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False, dtype=self.head_proj_dtype)

        self.max_seq_len = 2 ** 20
        self.freqs_complex = precompute_theta_pos_frequencies(self.head_dim, self.max_seq_len, self.rope_theta)

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor, checkpoint_layer_freq: int = 0):
        batch_size, seq_len = tokens.shape

        nvtx.range_push("Token Embeddings")
        h = self.tok_embeddings(tokens)
        nvtx.range_pop()

        freqs = self.freqs_complex[:seq_len].to(h.device)

        nvtx.range_push("Decoder Layers")
        for i, layer in enumerate(self.layers):
            nvtx.range_push(f"Layer {i}")
            # checkpoint_layer_freq == 0 => no checkpointing.
            # checkpoint_layer_freq == 1 => checkpoint every layer.
            # Otherwise => checkpoint iff i % checkpoint_layer_freq == 0.
            if checkpoint_layer_freq > 0 and (i % checkpoint_layer_freq) == 0:
                h = deepspeed.checkpointing.checkpoint(layer, h, freqs)
            else:
                h = layer(h, freqs)
            nvtx.range_pop()
        nvtx.range_pop()
            
        nvtx.range_push("Final Norm")
        h = self.norm(h)
        nvtx.range_pop()

        nvtx.range_push("Output Projection and Cross Entropy Loss")
        loss = chunked_linear_cross_entropy(h, self.output.weight, labels.view(-1))
        nvtx.range_pop()
        
        return loss