#!/usr/bin/env python3
"""
Megatron Core: Llama3-8B training with aggressive CPU offloading.

Offloading enabled:
  - Activations → CPU (via Transformer Engine cpu_offload context)
  - Weights → CPU (via Transformer Engine cpu_offload context)
  - Optimizer states → CPU (via Megatron Core HybridDeviceOptimizer)

Architecture: Llama3-8B dimensions
  - 32 layers, hidden_dim=4096, ffn_hidden=14336 (SwiGLU)
  - GQA: 32 query heads, 8 KV heads, head_dim=128
  - RMSNorm, RoPE, no bias

Usage:
  torchrun --nproc_per_node=1 --nnodes=1 train_llama3_8b_offload.py

Requirements:
  - megatron-core (pip install megatron-core)
  - transformer-engine >= 1.10.0 (for cpu offload context)
  - torch >= 2.1
  - apex (optional, for fused optimizers)
"""

import os
import sys
import torch
import torch.distributed as dist

# Enable TE v1 CPU offload code path
#os.environ["NVTE_CPU_OFFLOAD_V1"] = "1"

# ---------------------------------------------------------------------------
# Hyperparameters / constants
# ---------------------------------------------------------------------------
NUM_LAYERS = 12
HIDDEN_SIZE = 4096
FFN_HIDDEN_SIZE = 14336
NUM_ATTENTION_HEADS = 32
NUM_QUERY_GROUPS = 8              # KV heads for GQA
SEQ_LENGTH = 65536
VOCAB_SIZE = 128256               # Llama3 tokenizer
ROTARY_BASE = 500000              # Llama3 RoPE base

CPU_OFFLOAD_NUM_LAYERS = NUM_LAYERS - 1       # number of layers to offload activations

MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_ITERS = 100
LR = 3e-4
MIN_LR = 3e-5

# ---------------------------------------------------------------------------
# 1. Bootstrap torch.distributed – required even for single-GPU Megatron Core
# ---------------------------------------------------------------------------

def init_distributed():
    """Initialize torch.distributed and set device."""
    if not dist.is_initialized():
        # torchrun sets these; fall back to single-GPU defaults
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
    return dist.get_rank(), dist.get_world_size()


rank, world_size = init_distributed()

# ---------------------------------------------------------------------------
# 2. Megatron Core imports (after dist init)
# ---------------------------------------------------------------------------

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.distributed import DistributedDataParallel as MCoreDDP
from megatron.core.distributed import DistributedDataParallelConfig

# ---------------------------------------------------------------------------
# 3. Initialize Megatron parallel state
# ---------------------------------------------------------------------------

parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    context_parallel_size=1,
)

# ---------------------------------------------------------------------------
# 4. Model config – Llama3-8B dimensions with CPU offloading
# ---------------------------------------------------------------------------

# Architecture defined by constants at top of file

transformer_config = TransformerConfig(
    # --- Architecture ---
    num_layers=NUM_LAYERS,
    hidden_size=HIDDEN_SIZE,
    ffn_hidden_size=FFN_HIDDEN_SIZE,     # SwiGLU intermediate size
    num_attention_heads=NUM_ATTENTION_HEADS,
    num_query_groups=NUM_QUERY_GROUPS,   # KV heads (GQA)
    # head_dim is inferred: hidden_size / num_attention_heads = 128

    # --- Normalization ---
    normalization="RMSNorm",
    layernorm_epsilon=1e-5,

    # --- Activation ---
    activation_func=torch.nn.functional.silu,
    gated_linear_unit=True,          # SwiGLU
    bias_activation_fusion=False,    # no bias in Llama

    # --- Positional encoding ---
    # NOTE: position_embedding_type and rotary_base are set on GPTModel, not TransformerConfig

    # --- Precision ---
    bf16=True,
    fp16=False,
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,

    # --- No bias (Llama-style) ---
    add_bias_linear=False,
    add_qkv_bias=False,

    # --- Sequence length ---
    # NOTE: seq_length is set on GPTModel as max_sequence_length, not on TransformerConfig

    # ===================================================================
    # CPU OFFLOADING – activations and weights
    # These are consumed by TransformerBlock and passed to
    # transformer_engine.pytorch.cpu_offload.get_cpu_offload_context()
    # ===================================================================
    cpu_offloading=True,
    cpu_offloading_num_layers=CPU_OFFLOAD_NUM_LAYERS,
    cpu_offloading_activations=True,      # offload saved activations
    cpu_offloading_weights=True,          # offload layer weights when not in use
    cpu_offloading_double_buffering=True, # pre-fetch next layer while current runs

    # ===================================================================
    # ACTIVATION CHECKPOINTING (recomputation)
    #
    # Controls which activations are discarded during forward and
    # recomputed during backward to save GPU memory.
    #
    # ---- recompute_granularity: (str or None) ----
    #   None        → no checkpointing; all activations stored (fastest, most memory)
    #   "selective" → recompute specific submodules (controlled by recompute_modules).
    #                 Default recomputes only core_attn. Best memory/speed tradeoff.
    #   "full"      → recompute entire transformer layer forward during backward.
    #                 Maximum memory savings but requires recompute_method and
    #                 recompute_num_layers to be set.
    #
    # ---- recompute_modules: (list[str], used when granularity="selective") ----
    #   Specifies WHICH submodules to recompute. Default: ["core_attn"]
    #   You can combine multiple modules in the list.
    #
    #   Standard checkpointing (saves input, recomputes full submodule forward):
    #     "core_attn"  → recompute core attention (softmax, dropout, QK^T·V).
    #                    Highest memory savings per FLOP cost — attention
    #                    activations scale with seq_len² but compute is cheap.
    #     "mlp"        → recompute the dense MLP submodule. Useful for hybrid
    #                    models or when MLP intermediates dominate memory.
    #     "moe"        → recompute MoE layer (token dispatch + expert compute).
    #
    #   Output-discarding checkpointing (discards output, recomputes on backward):
    #     "layernorm"  → recompute input_layernorm and pre_mlp_layernorm.
    #                    Small compute cost, modest memory saving.
    #     "moe_act"    → recompute GroupedMLP activation function.
    #     "mla_up_proj"→ recompute Multi-Latent Attention up projection + RoPE.
    #
    #   Examples:
    #     recompute_modules=["core_attn"]                    # default, just attention
    #     recompute_modules=["core_attn", "mlp"]             # attention + MLP
    #     recompute_modules=["core_attn", "layernorm"]       # attention + norms
    #     recompute_modules=["core_attn", "moe", "moe_act"] # for MoE models
    #
    # ---- recompute_method: (str, only used when granularity="full") ----
    #   "uniform"   → divide layers into recompute_num_layers uniform chunks;
    #                  each chunk recomputes its layers during backward.
    #   "block"     → recompute the first recompute_num_layers layers only;
    #                  remaining layers store activations normally.
    #
    # ---- recompute_num_layers: (int, only used when granularity="full") ----
    #   Number of layers to recompute (meaning depends on recompute_method):
    #     - "uniform": number of uniform chunks
    #     - "block":   first N layers are recomputed, rest are stored
    #   Set to num_layers (32) to recompute everything.
    #
    # ---- Interaction with cpu_offloading (layer-level, TE-based) ----
    #   "selective" + offloading → recomputes chosen modules, offloads the
    #                              rest to CPU. Reduces both GPU memory and
    #                              PCIe offload volume. Recommended combo.
    #   "full" + offloading     → almost nothing to offload since activations
    #                              are recomputed. Offloading mostly handles
    #                              non-recomputed residuals/norms.
    #   None + offloading       → all activations offloaded to CPU. Maximum
    #                              PCIe traffic but no recompute overhead.
    #
    # ===================================================================
    # FINE-GRAINED ACTIVATION OFFLOADING (alternative to cpu_offloading)
    #
    # !! MUTUALLY EXCLUSIVE with cpu_offloading=True !!
    # Cannot be used simultaneously — see the assertion we discussed earlier.
    #
    # fine_grained_activation_offloading: (bool, default False)
    #   Module-level offloading instead of layer-level. Offloads the INPUT
    #   of specified submodules to CPU, giving precise control over which
    #   tensors move to CPU vs stay on GPU.
    #
    # offload_modules: (list[str], used when fine_grained_activation_offloading=True)
    #   Specifies which submodule inputs to offload to CPU:
    #     "attn_norm"   → input to attention layernorm
    #     "qkv_linear"  → input to QKV projection
    #     "core_attn"   → input to core attention
    #     "attn_proj"   → input to attention output projection
    #     "mlp_norm"    → input to MLP layernorm
    #     "expert_fc1"  → input to MoE expert first linear
    #     "moe_act"     → input to MoE activation function
    #
    #   Example (if you were NOT using cpu_offloading):
    #     fine_grained_activation_offloading=True,
    #     offload_modules=["qkv_linear", "core_attn", "attn_proj"],
    #
    #   Key difference from cpu_offloading:
    #     cpu_offloading (TE layer-level) → wraps entire layer in offload
    #       context, bulk-moves ALL activations/weights for that layer.
    #     fine_grained (module-level)     → surgically offloads specific
    #       tensor inputs, letting you keep hot tensors on GPU. More
    #       control, less PCIe traffic, but no weight offloading.
    #
    #   NOTE: With TE >= 2.10.0, set env var NVTE_CPU_OFFLOAD_V1=1
    #         to prevent fine-grained offloading from also moving weights.
    #
    # ===================================================================
    # COMBINING RECOMPUTE + OFFLOADING (strategy guide)
    #
    # You can combine recompute_modules with EITHER cpu_offloading OR
    # fine_grained_activation_offloading (but not both offloading types):
    #
    #   Strategy 1: Recompute attention + TE layer offloading (this script)
    #     recompute_granularity="selective"
    #     recompute_modules=["core_attn"]
    #     cpu_offloading=True
    #     → Attention recomputed (no activations stored or offloaded)
    #     → MLP/norm activations offloaded to CPU by TE context
    #     → Weights offloaded to CPU when layer is inactive
    #
    #   Strategy 2: Recompute attention + fine-grained MLP offloading
    #     recompute_granularity="selective"
    #     recompute_modules=["core_attn"]
    #     fine_grained_activation_offloading=True
    #     offload_modules=["mlp_norm", "expert_fc1"]
    #     → Attention recomputed, MLP inputs selectively offloaded
    #     → No weight offloading (fine-grained doesn't support it)
    #
    #   Strategy 3: Recompute everything, no offloading
    #     recompute_granularity="full"
    #     recompute_method="uniform"
    #     recompute_num_layers=32
    #     → Minimum memory, maximum recompute cost (~30% more FLOPs)
    # ===================================================================
    # NOTE: This version of megatron-core does NOT allow combining
    # cpu_offloading with activation recomputation. You must pick one:
    #   - cpu_offloading=True  + recompute_granularity=None  (this script)
    #   - cpu_offloading=False + recompute_granularity="selective" or "full"
    #
    # Newer versions (dev branch) may allow the combination.
    # With offloading enabled, activations are moved to CPU instead of
    # being recomputed, so you still get the memory savings.
    # ===================================================================
    recompute_granularity="full",             # disabled — incompatible with cpu_offloading in this version
    # recompute_modules=["core_attn"],     # would use with recompute_granularity="selective"
    recompute_method="block",          # would use with recompute_granularity="full"
    recompute_num_layers=32,             # would use with recompute_granularity="full"

    # --- Fine-grained offloading (DISABLED — mutually exclusive with cpu_offloading) ---
    # fine_grained_activation_offloading=True,
    # offload_modules=["qkv_linear", "core_attn", "attn_proj", "mlp_norm"],

    # --- Misc ---
    init_method_std=0.02,
    use_cpu_initialization=True,          # init on CPU, then move to GPU
    perform_initialization=True,
    fp32_residual_connection=False,
    apply_query_key_layer_scaling=False,
)

# ---------------------------------------------------------------------------
# 5. Build model
# ---------------------------------------------------------------------------

# Use Transformer Engine layer spec for TE-accelerated kernels + cpu offload
layer_spec = get_gpt_layer_with_transformer_engine_spec()

model = GPTModel(
    config=transformer_config,
    transformer_layer_spec=layer_spec,
    vocab_size=VOCAB_SIZE,
    max_sequence_length=SEQ_LENGTH,
    parallel_output=True,
    position_embedding_type="rope",
    rotary_base=ROTARY_BASE,
)

# Move to GPU
model.cuda(torch.cuda.current_device())

# ---------------------------------------------------------------------------
# 6. Wrap in DDP (needed even for single GPU if using distributed optimizer)
# ---------------------------------------------------------------------------

ddp_config = DistributedDataParallelConfig(
    grad_reduce_in_fp32=False,            # accumulate/reduce grads in bf16 (not fp32)
    overlap_grad_reduce=False,            # single GPU, no benefit
    overlap_param_gather=False,
    use_distributed_optimizer=True,       # required for optimizer CPU offload
    check_for_nan_in_grad=True,
)

model = MCoreDDP(
    config=transformer_config,
    ddp_config=ddp_config,
    module=model,
    disable_bucketing=False,
)

# ---------------------------------------------------------------------------
# 7. Optimizer config with CPU offloading
# ---------------------------------------------------------------------------

optimizer_config = OptimizerConfig(
    # --- Optimizer type ---
    optimizer="adam",
    lr=LR,
    min_lr=MIN_LR,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_eps=1e-8,

    # --- Gradient clipping ---
    clip_grad=1.0,

    # ===================================================================
    # OPTIMIZER CPU OFFLOADING
    # Moves master params, Adam m & v states, and the optimizer
    # step itself to CPU. Gradients are D2H transferred after
    # reduce-scatter, optimizer runs on CPU, updated params H2D back.
    # ===================================================================
    use_distributed_optimizer=True,
    optimizer_cpu_offload=True,
    optimizer_offload_fraction=1.0,       # offload 100% of optimizer states

    # --- Precision-aware optimizer (required with CPU offload) ---
    # With bf16 optimizer states, memory per param drops from 12 bytes
    # (fp32 param + fp32 m + fp32 v) to 6 bytes (bf16 param + bf16 m + bf16 v).
    # Tradeoff: reduced numerical precision in optimizer state tracking.
    # Works well in practice for most LLM training.
    use_precision_aware_optimizer=True,
    main_params_dtype=torch.bfloat16,     # master params in bf16 (default: fp32)
    exp_avg_dtype=torch.bfloat16,         # Adam first moment (m) in bf16
    exp_avg_sq_dtype=torch.bfloat16,      # Adam second moment (v) in bf16

    # --- Overlap D2H/H2D with CPU optimizer step ---
    overlap_cpu_optimizer_d2h_h2d=True,

    # --- Learning rate schedule ---
    # NOTE: lr_decay_style, lr_warmup_iters, lr_decay_iters are NOT in OptimizerConfig.
    # They are handled by the training loop / LR scheduler separately.
    # For this script we use a constant LR (no scheduler).
)

# Build the distributed optimizer with CPU offload
optimizer = get_megatron_optimizer(
    config=optimizer_config,
    model_chunks=[model],
)

# ---------------------------------------------------------------------------
# 8. Training loop
# ---------------------------------------------------------------------------

# Dummy data generator (replace with your real dataloader)
def get_dummy_batch(micro_batch_size=MICRO_BATCH_SIZE, seq_len=SEQ_LENGTH, vocab_size=VOCAB_SIZE):
    """Generate random input tokens and labels."""
    tokens = torch.randint(0, vocab_size, (micro_batch_size, seq_len),
                           device=torch.cuda.current_device())
    labels = torch.randint(0, vocab_size, (micro_batch_size, seq_len),
                           device=torch.cuda.current_device())
    # Megatron GPTModel expects:
    #   input_ids: (batch, seq)
    #   position_ids: (batch, seq)
    #   attention_mask: None (causal by default) or custom
    #   labels: (batch, seq) for loss computation
    position_ids = torch.arange(seq_len, device=torch.cuda.current_device())
    position_ids = position_ids.unsqueeze(0).expand(micro_batch_size, -1)
    return tokens, position_ids, labels


def forward_step(model, tokens, position_ids, labels):
    """Run forward pass and compute loss."""
    output = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=None,       # causal masking handled internally
        labels=labels,
    )
    # GPTModel returns per-token loss when labels are provided.
    # Reduce to scalar for backward().
    loss = output.mean()
    return loss


def train(
    model,
    optimizer,
    num_iters=100,
    micro_batch_size=1,
    gradient_accumulation_steps=4,
    seq_len=4096,
    log_interval=1,
):
    """Main training loop with gradient accumulation."""
    model.train()
    import time

    for iteration in range(1, num_iters + 1):
        iter_start = time.perf_counter()
        optimizer.zero_grad()
        accumulated_loss = 0.0

        for micro_step in range(gradient_accumulation_steps):
            tokens, position_ids, labels = get_dummy_batch(
                micro_batch_size=micro_batch_size,
                seq_len=seq_len,
            )

            # Forward
            loss = forward_step(model, tokens, position_ids, labels)

            # Scale loss for gradient accumulation
            scaled_loss = loss / gradient_accumulation_steps

            # Backward
            scaled_loss.backward()
            accumulated_loss += loss.detach().item()

        # Finish gradient sync (DDP reduce)
        model.finish_grad_sync()

        # Optimizer step (runs on CPU when cpu_offload is enabled)
        # This handles: grad D2H → CPU Adam step → param H2D → all-gather
        step_result = optimizer.step()
        # Different megatron-core versions return 2 or 3+ values
        if isinstance(step_result, tuple):
            update_successful = step_result[0]
            grad_norm = step_result[1]
        else:
            update_successful = step_result
            grad_norm = 0.0

        torch.cuda.synchronize()
        iter_end = time.perf_counter()
        step_time_ms = (iter_end - iter_start) * 1000
        tokens_per_step = micro_batch_size * gradient_accumulation_steps * seq_len
        throughput_tok_sec = tokens_per_step / (step_time_ms / 1000)

        if iteration % log_interval == 0 and rank == 0:
            avg_loss = accumulated_loss / gradient_accumulation_steps
            gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            gpu_mem_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(
                f"Iter {iteration:>5d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Grad norm: {grad_norm} | "
                f"Step: {step_time_ms:.0f} ms | "
                f"Throughput: {throughput_tok_sec:.0f} tok/s | "
                f"GPU mem: {gpu_mem_gb:.2f}/{gpu_mem_reserved_gb:.2f} GB | "
                f"Update: {'yes' if update_successful else 'skipped (nan/inf)'}"
            )

    if rank == 0:
        print("\nTraining complete.")


# ---------------------------------------------------------------------------
# 9. Print config summary and run
# ---------------------------------------------------------------------------

if rank == 0:
    print("=" * 70)
    print("Llama3-8B Training with Aggressive CPU Offloading")
    print("=" * 70)
    print(f"  Layers:              {transformer_config.num_layers}")
    print(f"  Hidden size:         {transformer_config.hidden_size}")
    print(f"  FFN hidden size:     {transformer_config.ffn_hidden_size}")
    print(f"  Query heads:         {transformer_config.num_attention_heads}")
    print(f"  KV heads:            {transformer_config.num_query_groups}")
    print(f"  Head dim:            {transformer_config.hidden_size // transformer_config.num_attention_heads}")
    print(f"  Vocab size:          {VOCAB_SIZE}")
    print(f"  Sequence length:     {SEQ_LENGTH}")
    print(f"  Precision:           bf16")
    print("-" * 70)
    print(f"  CPU offload acts:    {transformer_config.cpu_offloading_activations}")
    print(f"  CPU offload weights: {transformer_config.cpu_offloading_weights}")
    print(f"  Double buffering:    {transformer_config.cpu_offloading_double_buffering}")
    print(f"  CPU offload optim:   {optimizer_config.optimizer_cpu_offload}")
    print(f"  Offload fraction:    {optimizer_config.optimizer_offload_fraction}")
    print(f"  Overlap D2H/H2D:     {optimizer_config.overlap_cpu_optimizer_d2h_h2d}")
    print(f"  Recompute:           {transformer_config.recompute_granularity}")
    print("-" * 70)
    print(f"  DDP grad dtype:      bf16")
    print(f"  Master params dtype: bf16")
    print(f"  Adam exp_avg dtype:  bf16")
    print(f"  Adam exp_avg_sq:     bf16")
    print(f"  Distributed optim:   {ddp_config.use_distributed_optimizer}")
    print("=" * 70)
    print()

    # Estimate memory from actual model parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_params_b = num_params / 1e9
    print(f"Actual parameter count: {num_params:,} ({num_params_b:.2f}B)")
    print()
    print("Estimated GPU memory budget (all bf16):")
    print(f"  Model params (bf16, on GPU):       ~{num_params * 2 / 1e9:.1f} GB")
    print(f"  Optimizer states (bf16, offloaded): ~{num_params * 6 / 1e9:.1f} GB on CPU")
    print(f"    (vs ~{num_params * 12 / 1e9:.1f} GB if fp32 — 50% reduction)")
    print(f"  DDP grad buffer (bf16, ON GPU):     ~{num_params * 2 / 1e9:.1f} GB (cannot offload)")
    print(f"    (vs ~{num_params * 4 / 1e9:.1f} GB if fp32 — 50% reduction)")
    print(f"  Activations (offloaded per layer):  variable (depends on batch/seq)")
    print(f"  Weights (offloaded when inactive):  ~{num_params * 2 / 1e9:.1f} GB reduced peak")
    print()

train(
    model=model,
    optimizer=optimizer,
    num_iters=NUM_ITERS,
    micro_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    seq_len=SEQ_LENGTH,
    log_interval=1,
)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
parallel_state.destroy_model_parallel()
dist.destroy_process_group()
