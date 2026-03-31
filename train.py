from awsm_transformer import TransformerLayer, TransformerMoELayer, TransformerEmbed, TransformerHead
from awsm_transformer.utils import *
from active_model import ActiveModel
from working_set import determine_working_set_config
import json
import pickle
import ctypes
import sys
import os
import torch
from sequence import Sequence
from sequence_pool import SequencePool
import numpy as np
import time
from dashboard.dashboard_logger import DashboardLogger

all_model_dims = json.load(open("model_dims.json"))

DASHBOARD_PORT = 8300

MAX_GPU_MEM_GIB = None
MAX_HOST_MEM_GIB = None

FORCE_SAVED_ACT_LEVEL = None

TO_PROFILE_BACKEND = False
TO_PROFILE_TORCH_MEMORY = False
MAX_STEPS = None
TO_SAVE_ERROR = True

DEVICE_ID = 0
MAX_TOKENS_PER_STEP = 524288
MIN_CHUNK_SIZE = None
MIN_SEQ_LEN = 256
MAX_SEQ_LEN = 2048
USE_MUON = True
SAVE_FINAL = False

LOSS_THRESHOLD = None

MODEL_CHOICE = "olmoe_7Bx1B"

INIT_MODEL_PATH = f"init_models/init_{MODEL_CHOICE}"

RUN_NAME = f"{MODEL_CHOICE}_gbs_{MAX_TOKENS_PER_STEP}_minchunk_{MIN_CHUNK_SIZE}_maxseq_{MAX_SEQ_LEN}_maxgpumem_{MAX_GPU_MEM_GIB}_maxhostmem_{MAX_HOST_MEM_GIB}_usemuon_{USE_MUON}"

model_dims = all_model_dims[MODEL_CHOICE]


SAVE_MODEL_PATH = f"fineweb_ckpts/{RUN_NAME}"

SAVE_CHECKPOINT_FREQ = 0

RAND_SEED = 42
torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.cuda.manual_seed(RAND_SEED)

if USE_MUON:
    opt_choice = "Muon"
else:
    opt_choice = "AdamW"

training_config = {
    "master_weight_dtype": "bfloat16",
    "grad_dtype": "bfloat16",
    "opt_choice":  opt_choice,
    "opt_dtype": "bfloat16"
}

print("-------- Training Config --------")
print(training_config)
print("\n\n\n")


device = torch.device(f"cuda:{DEVICE_ID}")
local_device = device

model_hyperparams = {
    "rms_norm_eps": 1e-5,
    "position_angles": torch.tensor([10000.0], dtype=torch.float32, device=device),
    "window_size_left": -1,
    "window_size_right": -1,
    "load_bal_coeff": 0.01
}

print("-------- Model Hyperparams --------")
print(model_hyperparams)
print("\n\n\n")



print(f"Creating sequences", flush=True)

TOKENS_PER_SHARD = 100e6
NUM_SHARDS = 100

train_seq_pool = SequencePool(
    vocab_size=model_dims["vocab_size"],
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN,
    shard_path_pattern="fineweb10B/fineweb_train_{shard_index:06d}.bin",
    num_shards=NUM_SHARDS,
    min_tokens_threshold=TOKENS_PER_SHARD * 2,
)

train_seq_pool.prefetch_initial_shards(num_shards=min(2, NUM_SHARDS))

TOTAL_TOKENS = TOKENS_PER_SHARD * NUM_SHARDS
est_total_steps = TOTAL_TOKENS / MAX_TOKENS_PER_STEP



opt_hyperparams = {
    "lr": 0,
    "max_lr": 1e-3,
    "warmup_pct": 0.05,
    "cooldown_pct": 0.2,
    "final_lr": 1e-5,
    "est_total_steps": est_total_steps,
    "beta1": 0.95,
    "beta2": 0.98,
    "eps": 1e-8,
    "weight_decay": 0,
    "step_num": 0,
}

print("-------- Opt Hyperparams --------")
print(opt_hyperparams)
print("\n\n\n")






MAX_GPU_MEM_BYTES = None
MAX_HOST_MEM_BYTES = None

if MAX_GPU_MEM_GIB is not None:
    MAX_GPU_MEM_BYTES = MAX_GPU_MEM_GIB * 1024 * 1024 * 1024
if MAX_HOST_MEM_GIB is not None:
    MAX_HOST_MEM_BYTES = MAX_HOST_MEM_GIB * 1024 * 1024 * 1024


working_set_config, chosen_hardware_env = determine_working_set_config(model_dims, MAX_SEQ_LEN, MAX_TOKENS_PER_STEP, training_config=training_config, device_id=DEVICE_ID, min_chunk_size=MIN_CHUNK_SIZE, verbose=True, max_gpu_mem_bytes=MAX_GPU_MEM_BYTES, max_host_mem_bytes=MAX_HOST_MEM_BYTES)


## record this here for dashboard logging
working_set_config["force_saved_act_level"] = FORCE_SAVED_ACT_LEVEL

print("-------- Working Set Config --------")
print(working_set_config)
print("\n\n\n")

print("-------- Chosen Hardware Env --------")
print(chosen_hardware_env)
print("\n\n\n")


local_config = {
    "local_rank": 0,
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
head_layer = TransformerHead(model_dims, model_hyperparams)

if model_dims["num_routed_experts"] > 0:
    secondary_compute_stream = torch.cuda.Stream(device=device)
    try:
        _nvtxlib = ctypes.CDLL('libnvToolsExt.so')
        _nvtxlib.nvtxNameCuStreamA(secondary_compute_stream.cuda_stream, b"Secondary Compute")
    except Exception as e:
        pass
    model_layers = [TransformerMoELayer(layer_id, model_dims, model_hyperparams, is_muon=USE_MUON, secondary_compute_stream=secondary_compute_stream) for layer_id in range(model_dims["n_layers"])]
else:
    model_layers = [TransformerLayer(layer_id, model_dims, model_hyperparams, is_muon=USE_MUON) for layer_id in range(model_dims["n_layers"])]


chunk_metadata_func = model_layers[0].make_chunk_metadata

active_model = ActiveModel(INIT_MODEL_PATH, model_layers, working_set_config, local_config, chosen_hardware_env, chunk_metadata_func, embed_layer=embed_layer, head_layer=head_layer, local_device=local_device, force_saved_act_level=FORCE_SAVED_ACT_LEVEL)

print(f"Initializing model and saving model to {INIT_MODEL_PATH}", flush=True)



active_model.initialize(save_path=INIT_MODEL_PATH)

print(f"Loading model from {INIT_MODEL_PATH}", flush=True)
ret = active_model.load(INIT_MODEL_PATH)
if ret != 0:
    print("Failed to load model, exiting...", flush=True)
    sys.exit(ret)








torch.cuda.empty_cache()



if TO_PROFILE_BACKEND:
    active_model.start_profile()
#print(f"Running forward-backward pass", flush=True)



step_stats = {}

loss_smoothed = None

total_tokens = 0
total_seqs = 0
total_flops_cost = 0

train_start_time = time.time()

SMOOTH_DECAY = 0.95

if not os.path.exists(f"{SAVE_MODEL_PATH}"):
    os.makedirs(f"{SAVE_MODEL_PATH}")

if not os.path.exists(f"{SAVE_MODEL_PATH}/train_seqs"):
    os.makedirs(f"{SAVE_MODEL_PATH}/train_seqs")

if model_dims["num_routed_experts"] > 0:
    if not os.path.exists(f"{SAVE_MODEL_PATH}/expert_hists"):
        os.makedirs(f"{SAVE_MODEL_PATH}/expert_hists")


if TO_PROFILE_TORCH_MEMORY:
    torch.cuda.memory._record_memory_history(max_entries=1000000)

SAVE_STEPS = []


while LOSS_THRESHOLD is None or loss_smoothed is None or loss_smoothed > LOSS_THRESHOLD:
    opt_hyperparams["step_num"] += 1
    step_num = opt_hyperparams["step_num"]

    if MAX_STEPS is not None and step_num > MAX_STEPS:
        break

    opt_hyperparams["lr"] = get_lr(step_num, max_lr=opt_hyperparams["max_lr"], warmup_pct=opt_hyperparams["warmup_pct"], cooldown_pct=opt_hyperparams["cooldown_pct"], final_lr=opt_hyperparams["final_lr"], est_total_steps=opt_hyperparams["est_total_steps"])

    start_time = time.time()

    cur_step_stats = {}

    cur_step_stats["step_start_time"] = start_time
    cur_step_stats["step_num"] = step_num 
    cur_step_stats["lr"] = opt_hyperparams["lr"]

    
    train_seqs = train_seq_pool.get_sequences(max_token_count=MAX_TOKENS_PER_STEP)

    if len(train_seqs) == 0:
        print("No more sequences...!", flush=True)
        break

    step_tokens = sum([len(s) for s in train_seqs])
    print(f"\n[Step {step_num}] {len(train_seqs)} Sequences ({step_tokens} Tokens, Avg. Length: {step_tokens / len(train_seqs):.2f})", flush=True)

    cur_step_stats["step_num_seqs"] = len(train_seqs)
    cur_step_stats["step_tokens"] = step_tokens
    cur_step_stats["step_avg_seq_len"] = step_tokens / len(train_seqs)

    total_tokens += step_tokens
    cur_step_stats["total_tokens"] = total_tokens
    total_seqs += len(train_seqs)
    cur_step_stats["total_seqs"] = total_seqs
    

    
    active_model.fwd_bwd(train_seqs, verbose=False, loss_scale_factor=1.0 / step_tokens, total_tokens_per_step=step_tokens)

    step_loss = 0.0
    #for s in train_seqs:
    #    print(f"Sequence {s.seq_id} --- Avg. Loss: {s.per_token_loss.mean()}", flush=True)
    for s in train_seqs:
        step_loss += s.per_token_loss.sum()
    avg_loss = step_loss / step_tokens

    if loss_smoothed is None:
        loss_smoothed = avg_loss
    else:
        loss_smoothed = SMOOTH_DECAY * loss_smoothed + (1 - SMOOTH_DECAY) * avg_loss
    
    print(f"\tAvg. Loss --- {avg_loss:.4f}", flush=True)
    
    cur_step_stats["avg_loss"] = avg_loss

    if avg_loss == 100:
        print(f"ERROR: Average loss is 100!", flush=True)
        break
    
    ret = active_model.step(opt_hyperparams)
    
    error_step = False
    if ret != 0 and TO_SAVE_ERROR:
        error_step = True

        
    end_time = time.time()

    step_duration = end_time - start_time

    tokens_per_sec = step_tokens / step_duration

    step_flops = 0
    for s in train_seqs:
        step_flops += get_model_flops_per_sequence(len(s), model_dims)

    opt_flops = get_opt_flops(model_dims, is_muon=USE_MUON)

    step_flops += opt_flops

    total_flops_cost += step_flops

    tflops_per_sec = (step_flops / step_duration) / 1e12

    print(f"\tThroughput --- {tokens_per_sec:.2f} Tokens/sec, {tflops_per_sec:.2f} Effective TFLOPS\n\nSmoothed Loss: {loss_smoothed:.4f}\n\tOverall Tokens Processed: {total_tokens / 1e6:.2f}M, Overall Sequences Processed: {total_seqs / 1e3:.2f}k, Overall Time: {(end_time - train_start_time) / 60:.2f}min\n\n", flush=True)

    cur_step_stats["loss_smoothed"] = loss_smoothed
    cur_step_stats["step_tokens_per_sec"] = tokens_per_sec
    cur_step_stats["step_end_time"] = end_time
    cur_step_stats["step_duration"] = step_duration
    cur_step_stats["step_flops_cost"] = step_flops
    cur_step_stats["step_throughput_tflops"] = tflops_per_sec
    cur_step_stats["total_train_time"] = end_time - train_start_time
    cur_step_stats["total_flops_cost"] = total_flops_cost
    cur_step_stats["total_throughput_tflops"] = (cur_step_stats["total_flops_cost"] / cur_step_stats["total_train_time"]) / 1e12
    cur_step_stats["total_tokens_per_sec"] = cur_step_stats["total_tokens"] / cur_step_stats["total_train_time"]

    cur_step_stats["max_memory_allocated"] = torch.cuda.max_memory_allocated()
    cur_step_stats["max_memory_reserved"] = torch.cuda.max_memory_reserved()
    

    step_stats[step_num] = cur_step_stats
    dashboard.log(cur_step_stats)

    # Save sequences and expert histograms to disk for analysis later...
    pickle.dump(train_seqs, open(f"{SAVE_MODEL_PATH}/train_seqs/step_{step_num}.pkl", "wb"))

    if model_dims["num_routed_experts"] > 0:
        all_expert_hist = {}
        for layer_id in local_config["local_layer_ids"]:
            all_expert_hist[layer_id] = model_layers[layer_id].expert_hist
        pickle.dump(all_expert_hist, open(f"{SAVE_MODEL_PATH}/expert_hists/step_{step_num}.pkl", "wb"))

    ## Saving full training step of step gradients for analysis
    step_save_path = f"{SAVE_MODEL_PATH}/step_{step_num}"
    if ((step_num in SAVE_STEPS) or ((SAVE_CHECKPOINT_FREQ > 0) and (step_num % SAVE_CHECKPOINT_FREQ == 0))) or error_step:
        if error_step:
            step_save_path += "_error"
        active_model.save(step_save_path, save_opt_state=True, save_gradients=True)

print(f"\n\n\nSaving step stats to step_stats.pkl", flush=True)
pickle.dump(step_stats, open(f"{SAVE_MODEL_PATH}/step_stats.pkl", "wb"))

dashboard.close()

if SAVE_FINAL:
    save_path = f"{SAVE_MODEL_PATH}/step_{opt_hyperparams['step_num']}"
    active_model.save(save_path, save_opt_state=True, save_gradients=True)

torch.cuda.synchronize()

print(f"Cleaning up and destroying model...\n")
train_seq_pool.stop()
active_model.destroy()

if TO_PROFILE_BACKEND:
    print("Stopping backend profiling...")
    active_model.stop_profile()

if TO_PROFILE_TORCH_MEMORY:
    torch_mem_profile_path = f"{SAVE_MODEL_PATH}/profiling/torch_memory_snapshot.pkl"
    print(f"Dumping torch memory profiling snapshot to: '{torch_mem_profile_path}'. Can be loaded into `https://docs.pytorch.org/memory_viz` for analsys.\n\tThis may take a while and consume significant host memory if ran for at least a minute...")
    if not os.path.exists(f"{SAVE_MODEL_PATH}/profiling"):
        os.makedirs(f"{SAVE_MODEL_PATH}/profiling")
    torch.cuda.memory._dump_snapshot(torch_mem_profile_path)
    torch.cuda.memory._record_memory_history(enabled=None) # Stop recording
    # # then after saved down => load .pkl into: https://docs.pytorch.org/memory_viz
    # ## should look flat: besides small temporary memory, everything is allocated initially during activate_model.load()

