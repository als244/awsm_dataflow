import argparse
import json
import pickle
import ctypes
import sys
import os
import torch
import numpy as np
import time

from awsm_transformer import TransformerLayer, TransformerMoELayer, TransformerEmbed, TransformerHead
from awsm_transformer.utils import *
from active_model import ActiveModel
from working_set import determine_working_set_config
from sequence import Sequence
from sequence_pool import SequencePool
from dashboard.dashboard_logger import DashboardLogger
# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="AdaWS Transformer Training Benchmarking Script")
parser.add_argument("--seq_len",                   type=int,   default=8192,          help="Sequence length")
parser.add_argument("--seqs_per_step",             type=int,   default=64,            help="Sequences per step")
parser.add_argument("--max_steps",                 type=int,   default=10,            help="Max training steps (0 = unlimited)")
parser.add_argument("--max_gpu_mem_gib",            type=float, default=0,          help="Max GPU memory in GiB (0 = detect available capacity)")
parser.add_argument("--max_host_mem_gib",           type=float, default=0,          help="Max host memory in GiB (0 = detect available capacity)")
parser.add_argument("--min_chunk_size",            type=int,   default=None,          help="Minimum chunk size")
parser.add_argument("--max_chunk_size",            type=int,   default=None,          help="Maximum chunk size")
parser.add_argument("--max_tokens_per_round_limit",type=int,   default=None,          help="Max tokens per round limit")
parser.add_argument("--use_muon",                  type=lambda x: x.lower() != 'false', default=True, help="Use Muon optimizer (default: True, pass --use_muon false to disable)")
parser.add_argument("--model_choice",              type=str,   default="llama3_8B",   help="Model choice key from model_dims.json")
parser.add_argument("--run_name",                  type=str,   default="default_run", help="Run name")
parser.add_argument("--force_saved_act_level",     type=int,   default=None,          help="Force saved activation level")
parser.add_argument("--device_id",                 type=int,   default=0,             help="Device ID")
parser.add_argument("--dashboard_port",            type=int,   default=8300,          help="Dashboard port")
args = parser.parse_args()

SEQ_LEN                    = args.seq_len
SEQS_PER_STEP              = args.seqs_per_step
MAX_STEPS                  = args.max_steps if args.max_steps != 0 else None
MAX_GPU_MEM_GIB            = args.max_gpu_mem_gib if args.max_gpu_mem_gib != 0 else None
MAX_HOST_MEM_GIB           = args.max_host_mem_gib if args.max_host_mem_gib != 0 else None
USE_MUON                   = args.use_muon
MODEL_CHOICE               = args.model_choice
RUN_NAME                   = args.run_name
DEVICE_ID                  = args.device_id
DASHBOARD_PORT             = args.dashboard_port

### hidden internal args from README
FORCE_SAVED_ACT_LEVEL      = args.force_saved_act_level
MIN_CHUNK_SIZE             = args.min_chunk_size
MAX_CHUNK_SIZE             = args.max_chunk_size
MAX_TOKENS_PER_ROUND_LIMIT = args.max_tokens_per_round_limit

torch.cuda.set_device(DEVICE_ID)

# ---------------------------------------------------------------------------
# Fixed constants
# ---------------------------------------------------------------------------

TO_PROFILE_BACKEND      = True
TO_PROFILE_TORCH_MEMORY = False
TO_SAVE_ERROR           = False

MAX_TOKENS_PER_STEP = SEQ_LEN * SEQS_PER_STEP
MIN_SEQ_LEN         = SEQ_LEN
MAX_SEQ_LEN         = SEQ_LEN
SAVE_FINAL          = False

INIT_MODEL_PATH        = f"init_models/init_{MODEL_CHOICE}"
SAVE_MODEL_PATH        = f"model_ckpts/my_{MODEL_CHOICE}_awsm_{RUN_NAME}"
SAVE_CHECKPOINT_FREQ   = 0

RAND_SEED = 42
torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.cuda.manual_seed(RAND_SEED)

# ---------------------------------------------------------------------------
# Model dims & config
# ---------------------------------------------------------------------------

all_model_dims = json.load(open("model_dims.json"))
model_dims     = all_model_dims[MODEL_CHOICE]

opt_choice = "Muon" if USE_MUON else "AdamW"

training_config = {
    "master_weight_dtype": "bfloat16",
    "grad_dtype":          "bfloat16",
    "opt_choice":          opt_choice,
    "opt_dtype":           "bfloat16",
}

device       = torch.device(f"cuda:{DEVICE_ID}")
local_device = device

model_hyperparams = {
    "rms_norm_eps":      1e-5,
    "position_angles":   torch.tensor([500000.0], dtype=torch.float32, device=device),
    "window_size_left":  -1,
    "window_size_right": -1,
    "load_bal_coeff":    0.01,
}

TOTAL_TOKENS    = 2e9
est_total_steps = TOTAL_TOKENS / MAX_TOKENS_PER_STEP

opt_hyperparams = {
    "lr":              0,
    "max_lr":          3e-4,
    "warmup_pct":      0.1,
    "cooldown_pct":    0.2,
    "final_lr":        1e-5,
    "est_total_steps": est_total_steps,
    "beta1":           0.95,
    "beta2":           0.98,
    "eps":             1e-8,
    "weight_decay":    0.01,
    "step_num":        0,
}

# ---------------------------------------------------------------------------
# Sequence pool
# ---------------------------------------------------------------------------

### TODO: this is extremely wasteful for host memory usage, no need to load in billions of tokens/additional metadata created per sequence
### initially; instead should have background thread that refreshes loading in shards...
print(f"Creating sequences", flush=True)

train_seq_pool = SequencePool(vocab_size=model_dims["vocab_size"], min_seq_len=MIN_SEQ_LEN, max_seq_len=MAX_SEQ_LEN)
train_seq_pool.add_random_sequences(MAX_STEPS * SEQS_PER_STEP if MAX_STEPS is not None else SEQS_PER_STEP * 1000, SEQ_LEN)

# ---------------------------------------------------------------------------
# Memory limits
# ---------------------------------------------------------------------------

max_gpu_mem_bytes  = int(MAX_GPU_MEM_GIB  * (1 << 30)) if MAX_GPU_MEM_GIB  is not None else None
max_host_mem_bytes = int(MAX_HOST_MEM_GIB * (1 << 30)) if MAX_HOST_MEM_GIB is not None else None

# ---------------------------------------------------------------------------
# Working set config
# ---------------------------------------------------------------------------

working_set_config, chosen_hardware_env = determine_working_set_config(
    model_dims,
    MAX_SEQ_LEN,
    MAX_TOKENS_PER_STEP,
    training_config=training_config,
    device_id=DEVICE_ID,
    verbose=True,
    fixed_seq_len=True,
    min_chunk_size=MIN_CHUNK_SIZE,
    max_chunk_size=MAX_CHUNK_SIZE,
    max_tokens_per_round_limit=MAX_TOKENS_PER_ROUND_LIMIT,
    max_gpu_mem_bytes=max_gpu_mem_bytes,
    max_host_mem_bytes=max_host_mem_bytes,
)

## record this here for dashboard logging
working_set_config["force_saved_act_level"] = FORCE_SAVED_ACT_LEVEL

print("-------- Working Set Config --------")
print(working_set_config)
print("\n\n\n")

print("-------- Chosen Hardware Env --------")
print(chosen_hardware_env)
print("\n\n\n")

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

local_config = {
    "local_rank":      0,
    "local_layer_ids": list(range(model_dims["n_layers"])),
}


dashboard = DashboardLogger(
    url=f"http://localhost:{DASHBOARD_PORT}",
    run_id=RUN_NAME,
    run_name=f"{RUN_NAME}",
    model=MODEL_CHOICE,
    run_dir=SAVE_MODEL_PATH,
    config={
        "working_set_config": working_set_config,
        "init_model_path": INIT_MODEL_PATH,
        "model_dims": model_dims,
        "training_config": training_config,
        "model_hyperparams": model_hyperparams,
        "opt_hyperparams": opt_hyperparams,
        "local_config": local_config,
        "chosen_hardware_env": chosen_hardware_env,
    }
)

embed_layer = TransformerEmbed(model_dims, model_hyperparams)
head_layer  = TransformerHead(model_dims, model_hyperparams)

if model_dims["num_routed_experts"] > 0:
    secondary_compute_stream = torch.cuda.Stream(device=device)
    _nvtxlib = ctypes.CDLL('libnvToolsExt.so')
    _nvtxlib.nvtxNameCuStreamA(secondary_compute_stream.cuda_stream, b"Secondary Compute")
    model_layers = [
        TransformerMoELayer(layer_id, model_dims, model_hyperparams, is_muon=USE_MUON, secondary_compute_stream=secondary_compute_stream)
        for layer_id in range(model_dims["n_layers"])
    ]
else:
    model_layers = [
        TransformerLayer(layer_id, model_dims, model_hyperparams, is_muon=USE_MUON)
        for layer_id in range(model_dims["n_layers"])
    ]

chunk_metadata_func = model_layers[0].make_chunk_metadata

active_model = ActiveModel(
    INIT_MODEL_PATH, model_layers, working_set_config, local_config,
    chosen_hardware_env, chunk_metadata_func,
    embed_layer=embed_layer, head_layer=head_layer, local_device=local_device,
    force_saved_act_level=FORCE_SAVED_ACT_LEVEL,
)

print(f"Initializing model and saving model to {INIT_MODEL_PATH}", flush=True)
active_model.initialize(save_path=INIT_MODEL_PATH)

print(f"Loading model from {INIT_MODEL_PATH}", flush=True)
ret = active_model.load(INIT_MODEL_PATH)
if ret != 0:
    print("Failed to load model, exiting...", flush=True)
    sys.exit(ret)

# ---------------------------------------------------------------------------
# Profiling setup
# ---------------------------------------------------------------------------

torch.cuda.empty_cache()

if TO_PROFILE_BACKEND:
    active_model.start_profile()

if TO_PROFILE_TORCH_MEMORY:
    torch.cuda.memory._record_memory_history(max_entries=1000000)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
os.makedirs(f"{SAVE_MODEL_PATH}/train_seqs", exist_ok=True)

if model_dims["num_routed_experts"] > 0:
    os.makedirs(f"{SAVE_MODEL_PATH}/expert_hists", exist_ok=True)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

step_stats        = {}
loss_smoothed     = None
LOSS_THRESHOLD    = 3.28
total_tokens      = 0
total_seqs        = 0
total_flops_cost  = 0
train_start_time  = time.time()
SMOOTH_DECAY      = 0.95
SAVE_STEPS        = []

while loss_smoothed is None or loss_smoothed > LOSS_THRESHOLD:
    opt_hyperparams["step_num"] += 1
    step_num = opt_hyperparams["step_num"]

    if MAX_STEPS is not None and step_num > MAX_STEPS:
        break

    opt_hyperparams["lr"] = get_lr(
        step_num,
        max_lr=opt_hyperparams["max_lr"],
        warmup_pct=opt_hyperparams["warmup_pct"],
        cooldown_pct=opt_hyperparams["cooldown_pct"],
        final_lr=opt_hyperparams["final_lr"],
        est_total_steps=opt_hyperparams["est_total_steps"],
    )

    start_time    = time.time()
    cur_step_stats = {"step_start_time": start_time}
    cur_step_stats["step_num"] = step_num
    cur_step_stats["lr"] = opt_hyperparams["lr"]

    train_seqs = train_seq_pool.get_sequences(max_token_count=MAX_TOKENS_PER_STEP)

    if len(train_seqs) == 0:
        print("No more sequences...!", flush=True)
        break

    step_tokens = sum([len(s) for s in train_seqs])
    print(f"\n[Step {step_num}] {len(train_seqs)} Sequences ({step_tokens} Tokens, Avg. Length: {step_tokens / len(train_seqs):.2f})", flush=True)

    cur_step_stats["step_num_seqs"]      = len(train_seqs)
    cur_step_stats["step_tokens"]        = step_tokens
    cur_step_stats["step_avg_seq_len"]   = step_tokens / len(train_seqs)

    total_tokens += step_tokens
    cur_step_stats["total_tokens"] = total_tokens
    total_seqs   += len(train_seqs)
    cur_step_stats["total_seqs"]   = total_seqs

    active_model.fwd_bwd(train_seqs, verbose=False, loss_scale_factor=1.0 / step_tokens, total_tokens_per_step=step_tokens)

    step_loss = sum(s.per_token_loss.sum() for s in train_seqs)
    avg_loss  = step_loss / step_tokens

    loss_smoothed = avg_loss if loss_smoothed is None else SMOOTH_DECAY * loss_smoothed + (1 - SMOOTH_DECAY) * avg_loss

    print(f"\tAvg. Loss --- {avg_loss:.4f}", flush=True)
    cur_step_stats["avg_loss"] = avg_loss

    if avg_loss == 100:
        print(f"ERROR: Average loss is 100!", flush=True)
        break

    ret        = active_model.step(opt_hyperparams)
    error_step = (ret != 0 and TO_SAVE_ERROR)

    end_time       = time.time()
    step_duration  = end_time - start_time
    tokens_per_sec = step_tokens / step_duration

    step_flops  = sum(get_model_flops_per_sequence(len(s), model_dims) for s in train_seqs)
    step_flops += get_opt_flops(model_dims, is_muon=USE_MUON)
    total_flops_cost += step_flops
    tflops_per_sec   = (step_flops / step_duration) / 1e12
    
    max_alloc = torch.cuda.max_memory_allocated()
    max_reserve = torch.cuda.max_memory_reserved() 


    print(
        f"\n\tStep Time: {step_duration:.2f}sec\n"
        f"\tThroughput --- {tokens_per_sec:.2f} Tokens/sec, {tflops_per_sec:.2f} Effective TFLOPS, Max Alloc/Reserve {max_alloc / (1 << 30):.2f}/{max_reserve / (1 << 30):.2f} GiB\n\n"
        f"Smoothed Loss: {loss_smoothed:.4f}\n"
        f"\tOverall Tokens Processed: {total_tokens / 1e6:.2f}M, "
        f"Overall Sequences Processed: {total_seqs / 1e3:.2f}k, "
        f"Overall Time: {(end_time - train_start_time) / 60:.2f}min\n\n",
        flush=True,
    )

    cur_step_stats.update({
        "loss_smoothed":          loss_smoothed,
        "step_tokens_per_sec":    tokens_per_sec,
        "step_end_time":          end_time,
        "step_duration":          step_duration,
        "step_flops_cost":        step_flops,
        "step_throughput_tflops": tflops_per_sec,
        "total_train_time":       end_time - train_start_time,
        "total_flops_cost":       total_flops_cost,
        "total_throughput_tflops": (total_flops_cost / (end_time - train_start_time)) / 1e12,
        "total_tokens_per_sec":   total_tokens / (end_time - train_start_time),
    })

    cur_step_stats["max_memory_allocated"] = max_alloc
    cur_step_stats["max_memory_reserved"] = max_reserve

    step_stats[step_num] = cur_step_stats
    dashboard.log(cur_step_stats)

    """
    pickle.dump(train_seqs, open(f"{SAVE_MODEL_PATH}/train_seqs/step_{step_num}.pkl", "wb"))

    if model_dims["num_routed_experts"] > 0:
        all_expert_hist = {layer_id: model_layers[layer_id].expert_hist for layer_id in local_config["local_layer_ids"]}
        pickle.dump(all_expert_hist, open(f"{SAVE_MODEL_PATH}/expert_hists/step_{step_num}.pkl", "wb"))
    """

    step_save_path = f"{SAVE_MODEL_PATH}/step_{step_num}"
    if (step_num in SAVE_STEPS) or (SAVE_CHECKPOINT_FREQ > 0 and step_num % SAVE_CHECKPOINT_FREQ == 0) or error_step:
        if error_step:
            step_save_path += "_error"
        active_model.save(step_save_path, save_opt_state=True, save_gradients=True)

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

print(f"\n\n\nSaving step stats to step_stats.pkl", flush=True)
pickle.dump(step_stats, open(f"{SAVE_MODEL_PATH}/step_stats.pkl", "wb"))

if SAVE_FINAL:
    save_path = f"{SAVE_MODEL_PATH}/step_{opt_hyperparams['step_num']}"
    active_model.save(save_path, save_opt_state=True, save_gradients=True)

torch.cuda.synchronize()
dashboard.close()

print(f"Cleaning up and destroying model...\n")
active_model.destroy()

if TO_PROFILE_BACKEND:
    active_model.stop_profile()

if TO_PROFILE_TORCH_MEMORY:
    torch_mem_profile_path = f"{SAVE_MODEL_PATH}/profiling/torch_memory_snapshot.pkl"
    print(
        f"Dumping torch memory profiling snapshot to: '{torch_mem_profile_path}'. "
        f"Can be loaded into `https://docs.pytorch.org/memory_viz` for analysis.\n"
        f"\tThis may take a while and consume significant host memory if ran for at least a minute..."
    )
    os.makedirs(f"{SAVE_MODEL_PATH}/profiling", exist_ok=True)
    torch.cuda.memory._dump_snapshot(torch_mem_profile_path)
    torch.cuda.memory._record_memory_history(enabled=None)
