#!/usr/bin/env python3
"""
Megatron Core training with configurable memory optimization.

Supports both dense and MoE architectures loaded from a model_dims.json file.
All offloading and recomputation options are exposed as command-line arguments.

Examples:

  # Dense model from config
  torchrun --nproc_per_node=1 train.py --model llama3_8B --model-dims model_dims.json

  # MoE model from config
  torchrun --nproc_per_node=1 train.py --model olmoe_7Bx1B --model-dims model_dims.json

  # Override seq length or other arch params
  torchrun --nproc_per_node=1 train.py --model llama3_8B --seq-length 2048

  # Fine-grained offload + selective recompute
  torchrun --nproc_per_node=1 train.py --model llama3_8B \\
      --fine-grained-activation-offloading \\
      --offload-modules core_attn attn_proj \\
      --recompute-granularity selective --recompute-modules core_attn layernorm

Requirements:
  - megatron-core >= 0.12.0
  - transformer-engine >= 1.10.0
  - torch >= 2.1
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.distributed as dist


# ===========================================================================
# Model dims loading
# ===========================================================================

def load_model_dims(json_path: str) -> dict:
    """Load model dimension configs from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def is_moe_model(dims: dict) -> bool:
    """A model is MoE if it has routed experts > 0."""
    return dims.get("num_routed_experts", 0) > 0


def apply_model_dims(args: argparse.Namespace, dims: dict) -> None:
    """
    Map model_dims.json fields to argparse namespace.

    JSON field                → argparse field              → Megatron TransformerConfig field
    ─────────────────────────────────────────────────────────────────────────────────────────────
    n_layers                  → num_layers                  → num_layers
    d_model                   → hidden_size                 → hidden_size
    expert_dim                → ffn_hidden_size             → ffn_hidden_size (dense) or moe_ffn_hidden_size (MoE)
    n_heads                   → num_attention_heads         → num_attention_heads
    n_kv_heads                → num_query_groups            → num_query_groups
    vocab_size                → vocab_size                  → (passed to GPTModel)
    num_routed_experts        → num_moe_experts             → num_moe_experts (None for dense)
    top_k                     → moe_router_topk             → moe_router_topk
    num_shared_experts        → (used to compute shared expert size)

    For dense models (num_routed_experts == 0):
      - ffn_hidden_size = expert_dim  (the single "expert" IS the MLP)
      - num_moe_experts = None
      - moe_router_topk = 0

    For MoE models (num_routed_experts > 0):
      - moe_ffn_hidden_size = expert_dim  (per-expert FFN size)
      - ffn_hidden_size = expert_dim      (Megatron needs this set too)
      - num_moe_experts = num_routed_experts
      - moe_router_topk = top_k
    """
    moe = is_moe_model(dims)

    # Only apply JSON values if user didn't explicitly override on CLI
    # (argparse defaults are None for these so we can detect overrides)
    def set_if_default(attr, value):
        if getattr(args, attr) is None:
            setattr(args, attr, value)

    set_if_default("num_layers", dims["n_layers"])
    set_if_default("hidden_size", dims["d_model"])
    set_if_default("ffn_hidden_size", dims["expert_dim"])
    set_if_default("num_attention_heads", dims["n_heads"])
    set_if_default("num_query_groups", dims["n_kv_heads"])
    set_if_default("vocab_size", dims["vocab_size"])

    # MoE fields
    if moe:
        set_if_default("num_moe_experts", dims["num_routed_experts"])
        set_if_default("moe_router_topk", dims["top_k"])
        set_if_default("moe_ffn_hidden_size", dims["expert_dim"])
        # shared experts: Megatron uses moe_shared_expert_intermediate_size
        if dims.get("num_shared_experts", 0) > 0:
            # shared expert FFN size = num_shared_experts * expert_dim
            shared_size = dims["num_shared_experts"] * dims["expert_dim"]
            set_if_default("moe_shared_expert_intermediate_size", shared_size)
    else:
        # Dense model — no MoE
        args.num_moe_experts = None
        args.moe_router_topk = 0

    # Store for printing
    args._is_moe = moe
    args._model_name = args.model


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Megatron Core training with configurable memory optimization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Constraint summary (enforced by TransformerConfig.__post_init__):
  • --cpu-offloading and --fine-grained-activation-offloading are MUTUALLY EXCLUSIVE.
  • --fine-grained-activation-offloading requires --offload-modules (≥1 module).
  • --fine-grained-activation-offloading IS compatible with --recompute-granularity selective.
  • --recompute-granularity full requires --recompute-method and --recompute-num-layers.
  • --optimizer-cpu-offload is independent of all activation/weight strategies.
  • MoE-only recompute modules (moe, moe_act) require a MoE model config.
  • MoE-only offload modules (expert_fc1, moe_act) require a MoE model config.
""",
    )

    # ----- Model selection -----
    model_g = p.add_argument_group(
        "Model selection",
        description=(
            "Load architecture from a JSON config file. "
            "Use --model to select a named config. "
            "Individual arch flags (--num-layers, etc.) override JSON values."
        ),
    )
    model_g.add_argument("--model", type=str, default=None,
                         help="Model name to load from --model-dims JSON file")
    model_g.add_argument("--model-dims", type=str, default="model_dims.json",
                         help="Path to model dimensions JSON file (default: model_dims.json)")

    # ----- Model architecture (overrides for JSON values) -----
    # Defaults are None so we can detect whether user explicitly set them
    arch = p.add_argument_group("Model architecture (overrides JSON values)")
    arch.add_argument("--num-layers", type=int, default=None)
    arch.add_argument("--hidden-size", type=int, default=None)
    arch.add_argument("--ffn-hidden-size", type=int, default=None)
    arch.add_argument("--num-attention-heads", type=int, default=None)
    arch.add_argument("--num-query-groups", type=int, default=None)
    arch.add_argument("--seq-length", type=int, default=4096,
                      help="Sequence length (default: 4096)")
    arch.add_argument("--vocab-size", type=int, default=None)
    arch.add_argument("--rotary-base", type=float, default=500000,
                      help="RoPE base frequency (default: 500000)")

    # ----- MoE architecture (overrides for JSON values) -----
    moe_g = p.add_argument_group("MoE architecture (auto-set from JSON, overridable)")
    moe_g.add_argument("--num-moe-experts", type=int, default=None,
                       help="Number of routed experts (None = dense model)")
    moe_g.add_argument("--moe-router-topk", type=int, default=None,
                       help="Top-K experts per token")
    moe_g.add_argument("--moe-ffn-hidden-size", type=int, default=None,
                       help="Per-expert FFN hidden size")
    moe_g.add_argument("--moe-shared-expert-intermediate-size", type=int, default=None,
                       help="Shared expert intermediate size (0 = no shared expert)")
    moe_g.add_argument("--moe-grouped-gemm", action="store_true", default=False,
                       help="Enable grouped GEMM for MoE experts")

    # ----- TE layer-level CPU offloading -----
    ### NOTE: --cpu_offloading-weights should have NO impact; deprecated
    ### cpu-offloading = True implies offloading all activations; not compatible with any recompuation settings...
    te_offload = p.add_argument_group(
        "TE layer-level CPU offloading",
        description=(
            "Bulk offloading via TE cpu_offload context. "
            "MUTUALLY EXCLUSIVE with --fine-grained-activation-offloading."
        ),
    )
    te_offload.add_argument("--cpu-offloading", action="store_true", default=False)
    te_offload.add_argument("--cpu-offloading-num-layers", type=int, default=None)
    te_offload.add_argument("--cpu-offloading-activations", action="store_true", default=False)
    te_offload.add_argument("--cpu-offloading-weights", action="store_true", default=False)
    te_offload.add_argument("--no-cpu-offloading-double-buffering", action="store_true", default=False)

    # ----- Fine-grained activation offloading -----
    fg_offload = p.add_argument_group(
        "Fine-grained activation offloading",
        description=(
            "Module-level offloading. "
            "MUTUALLY EXCLUSIVE with --cpu-offloading. "
            "COMPATIBLE with --recompute-granularity selective."
        ),
    )
    fg_offload.add_argument("--fine-grained-activation-offloading", action="store_true", default=True)
    fg_offload.add_argument("--offload-modules", nargs="+", default=["qkv_linear", "core_attn", "attn_proj"],
        choices=["attn_norm", "qkv_linear", "core_attn", "attn_proj",
                 "mlp_norm", "expert_fc1", "moe_act"],
        help=(
            "Submodule inputs to offload to CPU. "
            "Dense modules: attn_norm, qkv_linear, core_attn, attn_proj, mlp_norm. "
            "MoE-only modules: expert_fc1, moe_act. "
            "Default: auto-selected based on dense vs MoE model."
        ),
    )

    # ----- Activation recomputation -----
    recomp = p.add_argument_group("Activation recomputation (checkpointing)")
    recomp.add_argument(
        "--recompute-granularity", type=str, default="selective",
        choices=["selective", "full"],
    )
    recomp.add_argument(
        "--recompute-modules", nargs="+", default=["core_attn", "mlp", "layernorm"],
        choices=["core_attn", "mlp", "moe", "layernorm", "moe_act", "mla_up_proj"],
        help=(
            "Submodules to recompute (with --recompute-granularity selective). "
            "Dense-safe: core_attn, mlp, layernorm. "
            "MoE-only: moe, moe_act. "
            "MLA-only: mla_up_proj. "
            "Default: auto-selected based on dense vs MoE model."
        ),
    )
    recomp.add_argument("--recompute-method", type=str, default=None, choices=["uniform", "block"])
    recomp.add_argument("--recompute-num-layers", type=int, default=None)

    # ----- Optimizer -----
    optim = p.add_argument_group("Optimizer")
    optim.add_argument("--lr", type=float, default=3e-4)
    optim.add_argument("--min-lr", type=float, default=3e-5)
    optim.add_argument("--weight-decay", type=float, default=0.1)
    optim.add_argument("--adam-beta1", type=float, default=0.9)
    optim.add_argument("--adam-beta2", type=float, default=0.95)
    optim.add_argument("--adam-eps", type=float, default=1e-8)
    optim.add_argument("--clip-grad", type=float, default=1.0)

    # ----- Optimizer CPU offloading -----
    optim_offload = p.add_argument_group("Optimizer CPU offloading")
    optim_offload.add_argument("--no-optimizer-cpu-offload", action="store_true", default=False)
    optim_offload.add_argument("--optimizer-offload-fraction", type=float, default=1.0)
    optim_offload.add_argument("--no-overlap-cpu-optimizer", action="store_true", default=False)
    optim_offload.add_argument("--no-precision-aware-optimizer", action="store_true", default=False)

    # ----- Training -----
    train_g = p.add_argument_group("Training")
    train_g.add_argument("--micro-batch-size", type=int, default=1)
    train_g.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train_g.add_argument("--num-iters", type=int, default=100)
    train_g.add_argument("--log-interval", type=int, default=1)

    args = p.parse_args()

    # ---- Load model dims from JSON ----
    if args.model is not None:
        if not os.path.exists(args.model_dims):
            p.error(f"Model dims file not found: {args.model_dims}")
        all_dims = load_model_dims(args.model_dims)
        if args.model not in all_dims:
            available = ", ".join(sorted(all_dims.keys()))
            p.error(f"Model '{args.model}' not found in {args.model_dims}. Available: {available}")
        apply_model_dims(args, all_dims[args.model])
    else:
        args._is_moe = (args.num_moe_experts is not None and args.num_moe_experts > 0)
        args._model_name = "custom"

    # ---- Fill remaining None arch values with dense Llama defaults ----
    arch_defaults = dict(
        num_layers=12, hidden_size=4096, ffn_hidden_size=14336,
        num_attention_heads=32, num_query_groups=8, vocab_size=128256,
    )
    for attr, default in arch_defaults.items():
        if getattr(args, attr) is None:
            setattr(args, attr, default)

    # ---- Auto-select offload/recompute modules based on dense vs MoE ----
    if args.fine_grained_activation_offloading and args.offload_modules is None:
        if args._is_moe:
            args.offload_modules = ["qkv_linear", "core_attn", "attn_proj", "expert_fc1"]
        else:
            args.offload_modules = ["qkv_linear", "core_attn", "attn_proj"]

    if args.recompute_granularity == "selective" and args.recompute_modules is None:
        if args._is_moe:
            args.recompute_modules = ["core_attn", "layernorm", "moe_act"]
        else:
            args.recompute_modules = ["core_attn", "layernorm"]

    # ---- Derived defaults ----
    if args.cpu_offloading_num_layers is None:
        args.cpu_offloading_num_layers = args.num_layers - 1

    # ---- Validation ----
    if args.cpu_offloading and args.fine_grained_activation_offloading:
        p.error("--cpu-offloading and --fine-grained-activation-offloading are mutually exclusive.")

    if args.fine_grained_activation_offloading and not args.offload_modules:
        p.error("--fine-grained-activation-offloading requires --offload-modules with ≥1 module.")

    if args.recompute_granularity == "full":
        if args.recompute_method is None:
            p.error("--recompute-granularity full requires --recompute-method")
        if args.recompute_num_layers is None:
            p.error("--recompute-granularity full requires --recompute-num-layers")

    if args.cpu_offloading and not (args.cpu_offloading_activations or args.cpu_offloading_weights):
        args.cpu_offloading_activations = True
        args.cpu_offloading_weights = True

    # Warn about MoE modules on dense models
    if not args._is_moe:
        moe_only_recompute = {"moe", "moe_act"}
        moe_only_offload = {"expert_fc1", "moe_act"}
        if args.recompute_modules:
            bad = set(args.recompute_modules) & moe_only_recompute
            if bad:
                p.error(f"--recompute-modules {bad} require a MoE model (num_routed_experts > 0)")
        if args.offload_modules:
            bad = set(args.offload_modules) & moe_only_offload
            if bad:
                p.error(f"--offload-modules {bad} require a MoE model (num_routed_experts > 0)")

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

    # ----- 2. Megatron Core imports -----
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
    recomp_kwargs = dict(recompute_granularity=args.recompute_granularity)
    if args.recompute_granularity == "selective":
        recomp_kwargs["recompute_modules"] = args.recompute_modules
    elif args.recompute_granularity == "full":
        recomp_kwargs["recompute_method"] = args.recompute_method
        recomp_kwargs["recompute_num_layers"] = args.recompute_num_layers

    # MoE kwargs (only set if MoE model)
    moe_kwargs = {}
    if args._is_moe:
        moe_kwargs["num_moe_experts"] = args.num_moe_experts
        moe_kwargs["moe_router_topk"] = args.moe_router_topk
        if args.moe_ffn_hidden_size is not None:
            moe_kwargs["moe_ffn_hidden_size"] = args.moe_ffn_hidden_size
        if args.moe_shared_expert_intermediate_size is not None and args.moe_shared_expert_intermediate_size > 0:
            moe_kwargs["moe_shared_expert_intermediate_size"] = args.moe_shared_expert_intermediate_size
        if args.moe_grouped_gemm:
            moe_kwargs["moe_grouped_gemm"] = True
        # MoE models need a router load balancing type
        moe_kwargs.setdefault("moe_router_load_balancing_type", "aux_loss")
        moe_kwargs.setdefault("moe_aux_loss_coeff", 1e-2)

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

        # --- Memory optimization ---
        **te_kwargs,
        **fg_kwargs,
        **recomp_kwargs,

        # --- MoE ---
        **moe_kwargs,
    )

    # ----- 5. Build model -----
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.num_moe_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
    )

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
    use_dist_optim = not args.no_optimizer_cpu_offload
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
        lr=args.lr, min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2, adam_eps=args.adam_eps,
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
    optimizer = get_megatron_optimizer(config=optimizer_config, model_chunks=[model])

    # ----- 8. Print config summary -----
    if rank == 0:
        print("=" * 70)
        model_type = "MoE" if args._is_moe else "Dense"
        print(f"Training: {args._model_name} ({model_type})")
        print("=" * 70)
        print(f"  Layers:              {args.num_layers}")
        print(f"  Hidden size:         {args.hidden_size}")
        print(f"  FFN hidden size:     {args.ffn_hidden_size}")
        print(f"  Query heads:         {args.num_attention_heads}")
        print(f"  KV heads:            {args.num_query_groups}")
        print(f"  Head dim:            {args.hidden_size // args.num_attention_heads}")
        print(f"  Vocab size:          {args.vocab_size}")
        print(f"  Sequence length:     {args.seq_length}")
        if args._is_moe:
            print(f"  Routed experts:      {args.num_moe_experts}")
            print(f"  Top-K:               {args.moe_router_topk}")
            if args.moe_ffn_hidden_size:
                print(f"  Expert FFN size:     {args.moe_ffn_hidden_size}")
            if args.moe_shared_expert_intermediate_size:
                print(f"  Shared expert size:  {args.moe_shared_expert_intermediate_size}")
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

    model.train()

    for iteration in range(1, args.num_iters + 1):
        iter_start = time.perf_counter()
        optimizer.zero_grad()
        accumulated_loss = 0.0

        for _ in range(args.gradient_accumulation_steps):
            tokens, position_ids, labels = get_dummy_batch()
            output = model(
                input_ids=tokens, position_ids=position_ids,
                attention_mask=None, labels=labels,
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
        tokens_per_step = args.micro_batch_size * args.gradient_accumulation_steps * args.seq_length
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
                f"GPU mem: {gpu_gb:.2f}/{gpu_res_gb:.2f} GB | "
                f"Update: {'yes' if update_successful else 'skipped (nan/inf)'}"
            )

    if rank == 0:
        print("\nTraining complete.")

    # ----- Cleanup -----
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()