#!/usr/bin/env python3
"""
Megatron Core: Llama3-8B training with configurable memory optimization.

All offloading and recomputation options are exposed as command-line arguments.
Three orthogonal memory systems can be tuned independently:

  1. Activation memory  → recompute and/or offload (see --recompute-*, --cpu-offloading, --fine-grained-*)
  2. Weight memory       → TE layer-level offloading (--cpu-offloading-weights)
  3. Optimizer memory    → CPU offload via distributed optimizer (--optimizer-cpu-offload)

Examples:

  # Strategy A — TE bulk layer offloading (activations + weights to CPU)
  torchrun --nproc_per_node=1 train_llama3_8b_offload.py \\
      --cpu-offloading --cpu-offloading-activations --cpu-offloading-weights

  # Strategy B — Fine-grained offload + selective recompute (recommended)
  torchrun --nproc_per_node=1 train_llama3_8b_offload.py \\
      --fine-grained-activation-offloading \\
      --offload-modules core_attn attn_proj mlp_norm \\
      --recompute-granularity selective --recompute-modules core_attn

  # Strategy C — Full recompute, no offloading
  torchrun --nproc_per_node=1 train_llama3_8b_offload.py \\
      --recompute-granularity full --recompute-method uniform --recompute-num-layers 1

  # Strategy D — Selective recompute only
  torchrun --nproc_per_node=1 train_llama3_8b_offload.py \\
      --recompute-granularity selective --recompute-modules core_attn

  # No activation optimization (baseline, store everything)
  torchrun --nproc_per_node=1 train_llama3_8b_offload.py

  # Disable optimizer CPU offload too (full GPU baseline)
  torchrun --nproc_per_node=1 train_llama3_8b_offload.py \\
      --no-optimizer-cpu-offload

Requirements:
  - megatron-core >= 0.12.0
  - transformer-engine >= 1.10.0 (for cpu offload context)
  - torch >= 2.1
  - For --fine-grained-activation-offloading with TE >= 2.10.0:
      set env NVTE_CPU_OFFLOAD_V1=1 (the script does this automatically)
"""

import argparse
import os
import sys
import time
import torch
import torch.distributed as dist
import ctypes

_cudart = ctypes.CDLL('libcudart.so')

try:
    _nvtxlib = ctypes.CDLL('libnvToolsExt.so')
except Exception as e:
    print(f"Error nvtx lib: {e}")
    _nvtxlib = None


def start_profile(self):
    return _cudart.cudaProfilerStart()
    
def stop_profile(self):
    return _cudart.cudaProfilerStop()


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Llama3-8B Megatron Core training with configurable memory optimization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Constraint summary (enforced by TransformerConfig.__post_init__):
  • --cpu-offloading and --fine-grained-activation-offloading are MUTUALLY EXCLUSIVE.
  • --fine-grained-activation-offloading requires --offload-modules (≥1 module).
  • --fine-grained-activation-offloading IS compatible with --recompute-granularity selective.
  • --recompute-granularity full requires --recompute-method and --recompute-num-layers.
  • --optimizer-cpu-offload is independent of all activation/weight strategies.
""",
    )

    # ----- Model architecture -----
    arch = p.add_argument_group("Model architecture")
    arch.add_argument("--num-layers", type=int, default=12,
                      help="Number of transformer layers (default: 12)")
    arch.add_argument("--hidden-size", type=int, default=4096,
                      help="Hidden dimension (default: 4096)")
    arch.add_argument("--ffn-hidden-size", type=int, default=14336,
                      help="FFN intermediate size for SwiGLU (default: 14336)")
    arch.add_argument("--num-attention-heads", type=int, default=32,
                      help="Number of query attention heads (default: 32)")
    arch.add_argument("--num-query-groups", type=int, default=8,
                      help="Number of KV heads for GQA (default: 8)")
    arch.add_argument("--seq-length", type=int, default=4096,
                      help="Sequence length (default: 4096)")
    arch.add_argument("--vocab-size", type=int, default=128256,
                      help="Vocabulary size (default: 128256, Llama3 tokenizer)")
    arch.add_argument("--rotary-base", type=float, default=500000,
                      help="RoPE base frequency (default: 500000)")

    # ----- TE layer-level CPU offloading (bulk, per-layer) -----
    #
    # Uses transformer_engine.pytorch.cpu_offload.get_cpu_offload_context()
    # to wrap each transformer layer. During forward, activations saved for
    # backward are bulk-moved to CPU via async D2H. Weights of inactive
    # layers can also be offloaded. Double-buffering pre-fetches the next
    # layer's weights while the current layer runs.
    #
    # This is a "coarse-grained" approach — the entire layer's state is
    # offloaded, giving maximum memory savings at the cost of high PCIe
    # bandwidth usage.
    #
    # MUTUALLY EXCLUSIVE with --fine-grained-activation-offloading.
    # Compatibility with --recompute-granularity depends on mcore version.
    #
    # Source: megatron/core/transformer/transformer_block.py
    te_offload = p.add_argument_group(
        "TE layer-level CPU offloading",
        description=(
            "Bulk offloading via Transformer Engine's cpu_offload context. "
            "Wraps each layer and moves activations/weights to CPU. "
            "MUTUALLY EXCLUSIVE with --fine-grained-activation-offloading."
        ),
    )
    te_offload.add_argument("--cpu-offloading", action="store_true", default=False,
                            help="Enable TE layer-level CPU offloading (activations + weights)")
    te_offload.add_argument("--cpu-offloading-num-layers", type=int, default=None,
                            help="Number of layers to offload (default: num_layers - 1)")
    te_offload.add_argument("--cpu-offloading-activations", action="store_true", default=False,
                            help="Offload saved activations to CPU (requires --cpu-offloading)")
    te_offload.add_argument("--cpu-offloading-weights", action="store_true", default=False,
                            help="Offload inactive layer weights to CPU (requires --cpu-offloading)")
    te_offload.add_argument("--no-cpu-offloading-double-buffering", action="store_true", default=False,
                            help="Disable double-buffering (pre-fetch next layer while current runs)")

    # ----- Fine-grained activation offloading (module-level) -----
    #
    # Instead of offloading an entire layer's state, this targets the INPUT
    # tensor of specific submodules. During forward, the chosen inputs are
    # asynchronously moved to CPU; during backward, they're retrieved.
    #
    # Key advantage: you can combine offloading (for heavy tensors where
    # D2H/H2D overlaps with compute) with selective recompute (for cheap
    # modules like layernorm). This minimizes both PCIe traffic and recompute.
    #
    # DOES NOT support weight offloading — only activation tensors.
    # MUTUALLY EXCLUSIVE with --cpu-offloading.
    # IS COMPATIBLE with --recompute-granularity selective.
    #
    # Requires:
    #   - TE layer spec (Transformer Engine implementation)
    #
    # Source: megatron/core/pipeline_parallel/fine_grained_activation_offload.py
    #         megatron/core/transformer/attention.py (per-module flags)
    #         megatron/core/transformer/transformer_layer.py (attn_norm, mlp_norm)
    fg_offload = p.add_argument_group(
        "Fine-grained activation offloading",
        description=(
            "Module-level offloading: offload specific submodule inputs to CPU. "
            "MUTUALLY EXCLUSIVE with --cpu-offloading. "
            "COMPATIBLE with --recompute-granularity selective."
        ),
    )
    fg_offload.add_argument("--fine-grained-activation-offloading", action="store_true", default=True,
                            help="Enable fine-grained (module-level) activation offloading")
    fg_offload.add_argument(
        "--offload-modules", nargs="+", default=["qkv_linear", "core_attn", "attn_proj"],
        choices=["attn_norm", "qkv_linear", "core_attn", "attn_proj",
                 "mlp_norm", "expert_fc1", "moe_act"],
        help=(
            "Which submodule inputs to offload to CPU. Required when "
            "--fine-grained-activation-offloading is set. "
            "Choices: attn_norm (attention layernorm input), "
            "qkv_linear (QKV projection input), "
            "core_attn (core attention input — Q,K,V tensors), "
            "attn_proj (attention output projection input), "
            "mlp_norm (MLP layernorm input), "
            "expert_fc1 (MoE expert first linear — MoE only), "
            "moe_act (MoE activation — MoE only), "
            "Default: [qkv_linear, core_attn, attn_proj]"
        ),
    )

    # ----- Activation recomputation (checkpointing) -----
    #
    # Instead of storing activations in memory, discard them during forward
    # and recompute them during backward. Trades FLOPs for memory.
    #
    # "selective" recomputes only specific submodules (best tradeoff).
    # "full" recomputes entire layers (maximum memory savings, ~30% more FLOPs).
    #
    # Compatible with --fine-grained-activation-offloading (selective only).
    # Compatibility with --cpu-offloading depends on mcore version.
    #
    # Source: megatron/core/transformer/transformer_block.py
    recomp = p.add_argument_group(
        "Activation recomputation (checkpointing)",
        description=(
            "Discard activations during forward, recompute during backward. "
            "Use 'selective' to recompute specific modules, 'full' for all."
        ),
    )
    recomp.add_argument(
        "--recompute-granularity", type=str, default="selective",
        choices=["selective", "full"],
        help=(
            "Recompute level. 'selective': recompute only modules listed in "
            "--recompute-modules (default: core_attn). 'full': recompute "
            "entire transformer layers (requires --recompute-method and "
            "--recompute-num-layers). None: no recomputation."
        ),
    )
    recomp.add_argument(
        "--recompute-modules", nargs="+", default=["core_attn", "mlp", "layernorm"],
        choices=["core_attn", "mlp", "moe", "layernorm", "moe_act", "mla_up_proj"],
        help=(
            "Submodules to recompute (used with --recompute-granularity selective). "
            "Standard (save input, recompute forward): "
            "core_attn (attention softmax/QKV — best bang for buck, O(seq²) memory), "
            "mlp (dense MLP), moe (MoE layer). "
            "Output-discarding (discard output, recompute on backward): "
            "layernorm (input + pre_mlp layernorms — cheap), "
            "moe_act (MoE only: GroupedMLP activation), "
            "mla_up_proj (MLA only: Multi-Latent Attention up projection + RoPE). "
            "Default: [core_attn, mlp, layernorm]"
        ),
    )
    recomp.add_argument(
        "--recompute-method", type=str, default=None,
        choices=["uniform", "block"],
        help=(
            "Method for full recompute. 'uniform': divide layers into chunks "
            "of --recompute-num-layers size. 'block': recompute the first "
            "--recompute-num-layers layers, store the rest."
        ),
    )
    recomp.add_argument("--recompute-num-layers", type=int, default=None,
                        help="Number of layers/chunks for full recompute")

    # ----- Optimizer -----
    optim = p.add_argument_group("Optimizer")
    optim.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    optim.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate (default: 3e-5)")
    optim.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay (default: 0.1)")
    optim.add_argument("--adam-beta1", type=float, default=0.9, help="Adam beta1 (default: 0.9)")
    optim.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 (default: 0.95)")
    optim.add_argument("--adam-eps", type=float, default=1e-8, help="Adam epsilon (default: 1e-8)")
    optim.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping norm (default: 1.0)")

    # ----- Optimizer CPU offloading -----
    #
    # Moves master params, Adam m & v states, and the optimizer step to CPU.
    # After reduce-scatter, gradients are D2H transferred, optimizer runs on
    # CPU, then updated params are H2D transferred back.
    #
    # INDEPENDENT of activation offloading strategy — can be combined with
    # any of the activation strategies above.
    #
    # Requires --use-distributed-optimizer (enforced by mcore).
    # With --use-precision-aware-optimizer, optimizer states can be stored
    # in bf16 instead of fp32, cutting CPU memory by ~50%.
    #
    # Source: megatron/core/optimizer/cpu_offloading.py (HybridDeviceOptimizer)
    #         megatron/core/optimizer/distrib_optimizer.py
    optim_offload = p.add_argument_group(
        "Optimizer CPU offloading",
        description=(
            "Offload optimizer states to CPU. Independent of activation strategy."
        ),
    )
    optim_offload.add_argument("--no-optimizer-cpu-offload", action="store_true", default=False,
                               help="Disable optimizer CPU offloading (default: enabled)")
    optim_offload.add_argument("--optimizer-offload-fraction", type=float, default=1.0,
                               help="Fraction of optimizer states to offload (default: 1.0)")
    optim_offload.add_argument("--no-overlap-cpu-optimizer", action="store_true", default=False,
                               help="Disable overlapping D2H/H2D with CPU optimizer step")
    optim_offload.add_argument("--no-precision-aware-optimizer", action="store_true", default=False,
                               help="Disable bf16 optimizer states (use fp32 instead)")

    # ----- Training -----
    train_g = p.add_argument_group("Training")
    train_g.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size (default: 1)")
    train_g.add_argument("--gradient-accumulation-steps", type=int, default=16,
                         help="Gradient accumulation steps (default: 16)")
    train_g.add_argument("--num-iters", type=int, default=5, help="Number of training iterations (default: 5)")
    train_g.add_argument("--log-interval", type=int, default=1, help="Log every N iterations (default: 1)")

    args = p.parse_args()

    # ---- Derived defaults ----
    if args.cpu_offloading_num_layers is None:
        args.cpu_offloading_num_layers = args.num_layers - 1

    # ---- Validation ----
    if args.cpu_offloading and args.fine_grained_activation_offloading:
        p.error(
            "--cpu-offloading and --fine-grained-activation-offloading are "
            "mutually exclusive. TE layer-level offloading wraps entire layers; "
            "fine-grained offloading targets specific submodule inputs. "
            "Choose one."
        )

    if args.fine_grained_activation_offloading and not args.offload_modules:
        p.error(
            "--fine-grained-activation-offloading requires --offload-modules "
            "with at least one module. Choices: attn_norm, qkv_linear, "
            "core_attn, attn_proj, mlp_norm, expert_fc1, moe_act"
        )

    if args.recompute_granularity == "full":
        if args.recompute_method is None:
            p.error("--recompute-granularity full requires --recompute-method (uniform or block)")
        if args.recompute_num_layers is None:
            p.error("--recompute-granularity full requires --recompute-num-layers")

    if args.cpu_offloading and not (args.cpu_offloading_activations or args.cpu_offloading_weights):
        # If user passed --cpu-offloading but neither sub-flag, enable both
        # (matching the common "offload everything" intent)
        args.cpu_offloading_activations = True
        args.cpu_offloading_weights = True

    return args


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    # ----- 1. Bootstrap torch.distributed -----
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    rank = dist.get_rank()

    # ----- 2. Megatron Core imports (after dist init) -----
    from megatron.core import parallel_state
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.optimizer.optimizer_config import OptimizerConfig
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.core.distributed import DistributedDataParallel as MCoreDDP
    from megatron.core.distributed import DistributedDataParallelConfig

    # ----- 3. Initialize parallel state -----
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
    )

    # ----- 4. Build TransformerConfig -----
    #
    # Activation memory kwargs are built from CLI flags. The three systems
    # (TE layer offload, fine-grained offload, recompute) are assembled
    # independently and passed together — TransformerConfig.__post_init__
    # enforces mutual exclusivity constraints.

    # TE layer-level offloading kwargs
    te_kwargs = dict(
        cpu_offloading=args.cpu_offloading,
        cpu_offloading_num_layers=args.cpu_offloading_num_layers,
        cpu_offloading_activations=args.cpu_offloading_activations,
        cpu_offloading_weights=args.cpu_offloading_weights,
        cpu_offloading_double_buffering=not args.no_cpu_offloading_double_buffering,
    )

    # Fine-grained offloading kwargs
    fg_kwargs = dict(
        fine_grained_activation_offloading=args.fine_grained_activation_offloading,
        offload_modules=args.offload_modules if args.fine_grained_activation_offloading else [],
    )

    # Recompute kwargs
    recomp_kwargs = dict(
        recompute_granularity=args.recompute_granularity,
    )
    if args.recompute_granularity == "selective":
        recomp_kwargs["recompute_modules"] = args.recompute_modules
    elif args.recompute_granularity == "full":
        recomp_kwargs["recompute_method"] = args.recompute_method
        recomp_kwargs["recompute_num_layers"] = args.recompute_num_layers

    transformer_config = TransformerConfig(
        # --- Architecture ---
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_query_groups,

        # --- Normalization ---
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,

        # --- Activation ---
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        bias_activation_fusion=False,

        # --- Precision ---
        bf16=True,
        fp16=False,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,

        # --- No bias (Llama-style) ---
        add_bias_linear=False,
        add_qkv_bias=False,

        # --- Misc ---
        init_method_std=0.02,
        use_cpu_initialization=True,
        perform_initialization=True,
        fp32_residual_connection=False,
        apply_query_key_layer_scaling=False,

        # --- Memory optimization (from CLI) ---
        **te_kwargs,
        **fg_kwargs,
        **recomp_kwargs,
    )

    # ----- 5. Build model -----
    # TE layer spec is REQUIRED for both cpu_offloading and
    # fine_grained_activation_offloading.
    layer_spec = get_gpt_layer_with_transformer_engine_spec()

    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=layer_spec,
        vocab_size=args.vocab_size,
        max_sequence_length=args.seq_length,
        parallel_output=True,
        position_embedding_type="rope",
        rotary_base=args.rotary_base,
    )
    model.cuda(torch.cuda.current_device())

    # ----- 6. Wrap in DDP -----
    use_dist_optim = not args.no_optimizer_cpu_offload  # dist optim required for CPU offload
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        use_distributed_optimizer=use_dist_optim,
        check_for_nan_in_grad=True,
    )

    model = MCoreDDP(
        config=transformer_config,
        ddp_config=ddp_config,
        module=model,
        disable_bucketing=False,
    )

    # ----- 7. Optimizer -----
    optimizer_cpu_offload = not args.no_optimizer_cpu_offload
    use_precision_aware = not args.no_precision_aware_optimizer

    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        clip_grad=args.clip_grad,

        use_distributed_optimizer=use_dist_optim,
        optimizer_cpu_offload=optimizer_cpu_offload,
        optimizer_offload_fraction=args.optimizer_offload_fraction if optimizer_cpu_offload else 0.0,

        use_precision_aware_optimizer=use_precision_aware,
        main_params_dtype=torch.bfloat16 if use_precision_aware else torch.float32,
        exp_avg_dtype=torch.bfloat16 if use_precision_aware else torch.float32,
        exp_avg_sq_dtype=torch.bfloat16 if use_precision_aware else torch.float32,

        overlap_cpu_optimizer_d2h_h2d=(optimizer_cpu_offload and not args.no_overlap_cpu_optimizer),
    )

    optimizer = get_megatron_optimizer(
        config=optimizer_config,
        model_chunks=[model],
    )

    # ----- 8. Print config summary -----
    if rank == 0:
        print("=" * 70)
        print("Llama3-8B Training — Configurable Memory Optimization")
        print("=" * 70)

        print(f"  Layers:              {args.num_layers}")
        print(f"  Hidden size:         {args.hidden_size}")
        print(f"  FFN hidden size:     {args.ffn_hidden_size}")
        print(f"  Query heads:         {args.num_attention_heads}")
        print(f"  KV heads:            {args.num_query_groups}")
        print(f"  Head dim:            {args.hidden_size // args.num_attention_heads}")
        print(f"  Vocab size:          {args.vocab_size}")
        print(f"  Sequence length:     {args.seq_length}")
        print(f"  Micro batch size:    {args.micro_batch_size}")
        print(f"  Grad accum steps:    {args.gradient_accumulation_steps}")
        print(f"  Precision:           bf16")

        print("-" * 70)
        print("  ACTIVATION MEMORY STRATEGY:")
        if args.cpu_offloading:
            print(f"    TE layer offload:  ON")
            print(f"      offload acts:    {args.cpu_offloading_activations}")
            print(f"      offload weights: {args.cpu_offloading_weights}")
            print(f"      double buffer:   {not args.no_cpu_offloading_double_buffering}")
            print(f"      num layers:      {args.cpu_offloading_num_layers}")
        elif args.fine_grained_activation_offloading:
            print(f"    Fine-grained:      ON")
            print(f"      offload modules: {args.offload_modules}")
        else:
            print(f"    Offloading:        OFF")

        if args.recompute_granularity:
            print(f"    Recompute:         {args.recompute_granularity}")
            if args.recompute_granularity == "selective":
                print(f"      modules:         {args.recompute_modules}")
            else:
                print(f"      method:          {args.recompute_method}")
                print(f"      num layers:      {args.recompute_num_layers}")
        else:
            print(f"    Recompute:         OFF")

        print("-" * 70)
        print("  OPTIMIZER:")
        print(f"    CPU offload:       {optimizer_cpu_offload}")
        if optimizer_cpu_offload:
            print(f"      fraction:        {args.optimizer_offload_fraction}")
            print(f"      overlap D2H/H2D: {not args.no_overlap_cpu_optimizer}")
        print(f"    Precision-aware:   {use_precision_aware}")
        if use_precision_aware:
            print(f"      master params:   bf16")
            print(f"      exp_avg:         bf16")
            print(f"      exp_avg_sq:      bf16")
        print(f"    Distributed optim: {use_dist_optim}")

        print("=" * 70)

        num_params = sum(p.numel() for p in model.parameters())
        num_params_b = num_params / 1e9
        optim_bytes = 6 if use_precision_aware else 12
        print(f"\n  Parameters: {num_params:,} ({num_params_b:.2f}B)")
        print(f"  Model params (bf16, GPU):         ~{num_params * 2 / 1e9:.1f} GB")
        print(f"  Optimizer states ({'bf16' if use_precision_aware else 'fp32'}, "
              f"{'CPU' if optimizer_cpu_offload else 'GPU'}):  "
              f"~{num_params * optim_bytes / 1e9:.1f} GB")
        print(f"  DDP grad buffer (bf16, GPU):      ~{num_params * 2 / 1e9:.1f} GB")
        print()

    # ----- 9. Training loop -----
    def get_dummy_batch():
        tokens = torch.randint(0, args.vocab_size,
                               (args.micro_batch_size, args.seq_length),
                               device=torch.cuda.current_device())
        labels = torch.randint(0, args.vocab_size,
                               (args.micro_batch_size, args.seq_length),
                               device=torch.cuda.current_device())
        position_ids = torch.arange(args.seq_length,
                                    device=torch.cuda.current_device())
        position_ids = position_ids.unsqueeze(0).expand(args.micro_batch_size, -1)
        return tokens, position_ids, labels


    start_profile()

    model.train()

    for iteration in range(1, args.num_iters + 1):
        iter_start = time.perf_counter()
        optimizer.zero_grad()
        accumulated_loss = 0.0

        for _ in range(args.gradient_accumulation_steps):
            tokens, position_ids, labels = get_dummy_batch()

            output = model(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=None,
                labels=labels,
            )
            loss = output.mean()
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            accumulated_loss += loss.detach().item()

        model.finish_grad_sync()

        step_result = optimizer.step()
        if isinstance(step_result, tuple):
            update_successful, grad_norm = step_result[0], step_result[1]
        else:
            update_successful, grad_norm = step_result, 0.0

        torch.cuda.synchronize()
        iter_end = time.perf_counter()
        step_time_ms = (iter_end - iter_start) * 1000
        tokens_per_step = (args.micro_batch_size * args.gradient_accumulation_steps
                           * args.seq_length)
        throughput = tokens_per_step / (step_time_ms / 1000)

        if iteration % args.log_interval == 0 and rank == 0:
            avg_loss = accumulated_loss / args.gradient_accumulation_steps
            gpu_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            gpu_res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(
                f"Iter {iteration:>5d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Grad norm: {grad_norm} | "
                f"Step: {step_time_ms:.0f} ms | "
                f"Throughput: {throughput:.0f} tok/s | "
                f"GPU mem: {gpu_gb:.2f}/{gpu_res_gb:.2f} GiB | "
                f"Update: {'yes' if update_successful else 'skipped (nan/inf)'}"
            )

    if rank == 0:
        print("\nTraining complete.")
    
    stop_profile()

    # ----- Cleanup -----
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()