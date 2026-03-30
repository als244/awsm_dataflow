#!/usr/bin/env python3
"""
NeMo 2.0 Megatron Core training with Chunked Cross Entropy.

Supports both dense and MoE architectures loaded from a model_dims.json file.
All offloading and recomputation options are exposed as command-line arguments.

Examples:
  torchrun --nproc_per_node=1 train_nemo2.py --model llama3_8B --cross-entropy-chunk-size 1024
"""

import argparse
import json
import os
import ctypes
import torch

# --- NeMo 2.0 Imports ---
import nemo.lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig

_cudart = ctypes.CDLL('libcudart.so')

def start_profile():
    return _cudart.cudaProfilerStart()
    
def stop_profile():
    return _cudart.cudaProfilerStop()


# ===========================================================================
# Model dims loading
# ===========================================================================

def load_model_dims(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)

def is_moe_model(dims: dict) -> bool:
    return dims.get("num_routed_experts", 0) > 0

def apply_model_dims(args: argparse.Namespace, dims: dict) -> None:
    moe = is_moe_model(dims)

    def set_if_default(attr, value):
        if getattr(args, attr) is None:
            setattr(args, attr, value)

    set_if_default("num_layers", dims["n_layers"])
    set_if_default("hidden_size", dims["d_model"])
    set_if_default("ffn_hidden_size", dims["expert_dim"])
    set_if_default("num_attention_heads", dims["n_heads"])
    set_if_default("num_query_groups", dims["n_kv_heads"])
    set_if_default("vocab_size", dims["vocab_size"])

    if moe:
        set_if_default("num_moe_experts", dims["num_routed_experts"])
        set_if_default("moe_router_topk", dims["top_k"])
        set_if_default("moe_ffn_hidden_size", dims["expert_dim"])
        if dims.get("num_shared_experts", 0) > 0:
            shared_size = dims["num_shared_experts"] * dims["expert_dim"]
            set_if_default("moe_shared_expert_intermediate_size", shared_size)
    else:
        args.num_moe_experts = None
        args.moe_router_topk = 0

    args._is_moe = moe
    args._model_name = args.model


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    # ----- Training -----
    train_g = p.add_argument_group("Training")
    train_g.add_argument("--micro-batch-size", type=int, default=1)
    train_g.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train_g.add_argument("--num-iters", type=int, default=5)
    train_g.add_argument("--log-interval", type=int, default=1)
    train_g.add_argument("--cross-entropy-chunk-size", type=int, default=1024)

    # ----- Model selection -----
    model_g = p.add_argument_group("Model selection")
    model_g.add_argument("--model", type=str, default=None)
    model_g.add_argument("--model-dims", type=str, default="model_dims.json")

    # ----- Model architecture -----
    arch = p.add_argument_group("Model architecture")
    arch.add_argument("--seq-length", type=int, default=4096)
    arch.add_argument("--num-layers", type=int, default=None)
    arch.add_argument("--hidden-size", type=int, default=None)
    arch.add_argument("--ffn-hidden-size", type=int, default=None)
    arch.add_argument("--num-attention-heads", type=int, default=None)
    arch.add_argument("--num-query-groups", type=int, default=None)
    arch.add_argument("--vocab-size", type=int, default=128256)
    arch.add_argument("--rotary-base", type=float, default=500000)

    # ----- MoE architecture -----
    moe_g = p.add_argument_group("MoE architecture")
    moe_g.add_argument("--num-moe-experts", type=int, default=None)
    moe_g.add_argument("--moe-router-topk", type=int, default=None)
    moe_g.add_argument("--moe-ffn-hidden-size", type=int, default=None)
    moe_g.add_argument("--moe-shared-expert-intermediate-size", type=int, default=None)
    moe_g.add_argument("--moe-grouped-gemm", action="store_true", default=False)

    # ----- TE layer-level CPU offloading -----
    te_offload = p.add_argument_group("TE layer-level CPU offloading")
    te_offload.add_argument("--cpu-offloading", action="store_true", default=False)
    te_offload.add_argument("--cpu-offloading-num-layers", type=int, default=None)
    te_offload.add_argument("--cpu-offloading-activations", action="store_true", default=False)
    te_offload.add_argument("--cpu-offloading-weights", action="store_true", default=False)
    te_offload.add_argument("--no-cpu-offloading-double-buffering", action="store_true", default=False)

    # ----- Fine-grained activation offloading -----
    fg_offload = p.add_argument_group("Fine-grained activation offloading")
    fg_offload.add_argument("--fine-grained-activation-offloading", action="store_true", default=True)
    fg_offload.add_argument("--offload-modules", nargs="+", default=None)

    # ----- Activation recomputation -----
    recomp = p.add_argument_group("Activation recomputation")
    recomp.add_argument("--recompute-granularity", type=str, default="selective", choices=["selective", "full"])
    recomp.add_argument("--recompute-modules", nargs="+", default=None)
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

    args = p.parse_args()

    # ---- Load model dims from JSON ----
    if args.model is not None:
        if not os.path.exists(args.model_dims):
            p.error(f"Model dims file not found: {args.model_dims}")
        all_dims = load_model_dims(args.model_dims)
        apply_model_dims(args, all_dims[args.model])
    else:
        args._is_moe = (args.num_moe_experts is not None and args.num_moe_experts > 0)
        args._model_name = "custom"

    # ---- Fill remaining None arch values with defaults ----
    arch_defaults = dict(
        num_layers=12, hidden_size=4096, ffn_hidden_size=14336,
        num_attention_heads=32, num_query_groups=8, vocab_size=128256,
    )
    for attr, default in arch_defaults.items():
        if getattr(args, attr) is None:
            setattr(args, attr, default)

    if args.fine_grained_activation_offloading and args.offload_modules is None:
        args.offload_modules = ["qkv_linear", "core_attn", "attn_proj", "expert_fc1"] if args._is_moe else ["qkv_linear", "core_attn", "attn_proj"]

    if args.recompute_granularity == "selective" and args.recompute_modules is None:
        args.recompute_modules = ["core_attn", "layernorm", "moe", "moe_act"] if args._is_moe else ["core_attn", "mlp", "layernorm"]

    if args.cpu_offloading_num_layers is None:
        args.cpu_offloading_num_layers = args.num_layers - 1

    if args._is_moe and args.recompute_modules and "moe_act" in args.recompute_modules:
        if not args.moe_grouped_gemm:
            args.moe_grouped_gemm = True

    return args


# ===========================================================================
# Main NeMo 2.0 Execution
# ===========================================================================
def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("=" * 70, flush=True)
        model_type = "MoE" if args._is_moe else "Dense"
        print(f"Training NeMo 2.0: {args._model_name} ({model_type})", flush=True)
        print(f"Chunked CE Size: {args.cross_entropy_chunk_size}", flush=True)
        print("=" * 70, flush=True)

    # --- 1. Model Configuration ---
    config_kwargs = dict(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_query_groups,
        seq_length=args.seq_length,
        normalization="RMSNorm",
        position_embedding_type="rope",
        rotary_base=args.rotary_base,
        make_vocab_size_divisible_by=128,
        
        # Memory & Checkpointing mappings (FIXED TO NATIVE MEGATRON NAMES)
        recompute_granularity=args.recompute_granularity if args.recompute_granularity else None,
        recompute_method=args.recompute_method,
        recompute_num_layers=args.recompute_num_layers,
        
        # Offloading
        cpu_offloading=args.cpu_offloading,
        cpu_offloading_num_layers=args.cpu_offloading_num_layers,
        cpu_offloading_activations=args.cpu_offloading_activations,
        cpu_offloading_weights=args.cpu_offloading_weights,
        
        fine_grained_activation_offloading=args.fine_grained_activation_offloading,
        offload_modules=args.offload_modules if args.fine_grained_activation_offloading else [],
    )

    if args._is_moe:
        config_kwargs.update({
            "num_moe_experts": args.num_moe_experts,
            "moe_router_topk": args.moe_router_topk,
            "moe_grouped_gemm": args.moe_grouped_gemm,
            "moe_router_load_balancing_type": "aux_loss",
            "moe_aux_loss_coeff": 1e-2,
        })
        if args.moe_ffn_hidden_size:
            config_kwargs["moe_ffn_hidden_size"] = args.moe_ffn_hidden_size
        if args.moe_shared_expert_intermediate_size:
            config_kwargs["moe_shared_expert_intermediate_size"] = args.moe_shared_expert_intermediate_size

    # Initialize standard GPTConfig
    gpt_config = llm.GPTConfig(**config_kwargs)

    # Optional Chunked Cross Entropy parameter override if supported
    if hasattr(gpt_config, 'cross_entropy_chunk_size'):
        gpt_config.cross_entropy_chunk_size = args.cross_entropy_chunk_size

    # --- 2. Data Module ---
    # FIXED: Removed 'vocab_size' from init arguments
    data = llm.MockDataModule(
        seq_length=args.seq_length,
        global_batch_size=args.micro_batch_size * args.gradient_accumulation_steps,
        micro_batch_size=args.micro_batch_size,
    )

    # --- 3. Tokenizer Wrapper ---
    # We must intercept NeMo's dummy tokenizer to report the true vocab size 
    # so that memory benchmarks match your target architecture.
    class DummyTokenizerWrapper:
        def __init__(self, wrapped, target_vocab_size):
            self.wrapped = wrapped
            self.vocab_size = target_vocab_size
            
        def __getattr__(self, item):
            return getattr(self.wrapped, item)
            
    custom_tokenizer = DummyTokenizerWrapper(data.tokenizer, args.vocab_size)

    # --- 4. Optimizer Setup ---
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=args.lr,
        weight_decay=args.weight_decay,
        bf16=not args.no_precision_aware_optimizer,
        use_distributed_optimizer=not args.no_optimizer_cpu_offload,
    )
    optim = nl.MegatronOptimizerModule(config=opt_config)

    # --- 5. Initialize Model ---
    model = llm.GPTModel(gpt_config, tokenizer=custom_tokenizer)

    # --- 6. Trainer & Strategy ---
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
    )

    trainer = nl.Trainer(
        devices=int(os.environ.get("WORLD_SIZE", 1)),
        num_nodes=1,
        accelerator="gpu",
        max_steps=args.num_iters,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        val_check_interval=args.num_iters + 1, 
        log_every_n_steps=args.log_interval,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        enable_model_summary=False,
    )
    
    nemo_logger = nl.NeMoLogger(log_dir="./nemo_experiments")

    # --- 7. Train ---
    start_profile()
    
    # NeMo 2.0's single unified train command
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer='data',
        optim=optim,
    )
    
    stop_profile()

    if local_rank == 0:
        print("\nTraining complete.", flush=True)

if __name__ == "__main__":
    main()