# train.py

os.environ["DS_SKIP_CUDA_CHECK"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "pinned_use_cuda_host_register:True,expandable_segments:True"


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import deepspeed
import argparse
import time

import psutil
import os

import ctypes

import json

from model_quack import Model, ModelArgs

from select_bins import select_bins

_cudart = ctypes.CDLL('libcudart.so')

process = psutil.Process(os.getpid())
# Initialize peak host memory usage tracker
peak_host_mem_gb = 0.0

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="DeepSpeed Training")
parser.add_argument('--zero_stage', type=int, default=None)
parser.add_argument('--save_act_layer_frac', type=float, default=0, help="Fraction of layer activatons to avoid recomputation and leave on device, deafult is 0 (full layer-wise checkpointing)")
parser.add_argument('--offload_act', action='store_true', help="Offload act checkpoints to cpu (blocking, hurts perf)")
### This has terrible performance, every bwd layer triggers blocking, paged data transfer of muon state to device and then to host causing severe idleness...
parser.add_argument('--use_muon', action='store_true', help="Use muon optimizer, otherwise AdamW")
parser.add_argument('--model_name', type=str, required=True, help="Key in model_dims.json (e.g. llama3_8B)")
parser.add_argument('--seq_len', type=int, default=512, help='Sequence length for training')
parser.add_argument('--seqs_per_batch', type=int, default=1)
parser.add_argument('--grad_accum_steps', type=int, default=1)
parser.add_argument('--num_steps', type=int, default=3)
parser.add_argument('--local_rank', type=int, default=0, help='Local rank')
# === CHANGE: Add deepspeed_config argument ===
# DeepSpeed launcher will automatically provide this argument.
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

grad_accum_steps = args.grad_accum_steps
seqs_per_batch = args.seqs_per_batch
num_steps = args.num_steps
zero_stage = args.zero_stage
save_act_layer_frac = args.save_act_layer_frac
offload_act = args.offload_act
use_muon = args.use_muon

global_steps = grad_accum_steps * num_steps

MODEL_DIMS_FILE = "model_dims.json"

def load_model_args(model_name: str) -> ModelArgs:
    """Load ModelArgs for the given model name from model_dims.json."""

    DTYPE_MAP = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "none": None,
    }

    with open(MODEL_DIMS_FILE, 'r') as f:
        all_configs = json.load(f)

    if model_name not in all_configs:
        available = ", ".join(all_configs.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    config = all_configs[model_name]

    # Flatten the nested "datatypes" dict into top-level keys if present
    if "datatypes" in config:
        dt = config.pop("datatypes")
        for key, val in dt.items():
            # Map to torch dtypes; keep as-is if not recognised (ModelArgs
            # can decide what to do with it)
            config[key + "_dtype"] = DTYPE_MAP.get(val, val)

    # Also handle any remaining loose dtype strings (legacy flat format)
    for key in list(config.keys()):
        if isinstance(config[key], str) and config[key] in DTYPE_MAP:
            config[key] = DTYPE_MAP[config[key]]

    return ModelArgs(**config)

model_args = load_model_args(args.model_name)

SEED = 42

torch.manual_seed(SEED)

# --- Model & Training Configuration ---

epochs = 1
learning_rate = 1e-8

AdamOptSettings = { "type": "AdamW", "params": { "lr": learning_rate, "betas": [0.9, 0.999], "weight_decay": 1e-4, "eps": 1e-8}}
MuonOptSettings = { "type": "Muon", "params": { "lr": learning_rate, "momentum": 0.9, "weight_decay": 0.0, "muon_lr": 0.001}}

if use_muon:
    opt_settings = MuonOptSettings
else:
    opt_settings = AdamOptSettings

# --- DeepSpeed Configuration with Logging ---
# The ds_config can be loaded from a JSON file specified by --deepspeed_config
# For simplicity, we define it here.
ds_config = {
    "train_micro_batch_size_per_gpu": seqs_per_batch,
    "gradient_accumulation_steps": grad_accum_steps,
    "optimizer": opt_settings,
    "bf16": { "enabled": True, "bf16_optimizer_states": True},
    #"wall_clock_breakdown": True,
    #"steps_per_print": 1,

    ## configuring this below...
    #"activation_checkpointing": { "cpu_checkpointing": True, "partition_activations": True },
}

if zero_stage and zero_stage != 0:
    ### only supported in zero stage 1,2, or 3
    ds_config["bf16"]["bf16_master_weights_and_grads"] = True
    ds_config['optimizer']['fp32_optimizer_states'] = False
    if zero_stage == 1:
        ds_config['zero_optimization'] = {"stage": 1, "offload_optimizer": {"device": "cpu", "pin_memory": True}, "reduce_scatter": False}
    elif zero_stage == 2:
        ds_config['zero_optimization'] = {"stage": 2, "offload_optimizer": {"device": "cpu", "pin_memory": True}, "reduce_scatter": False}
    elif zero_stage == 3:
        ds_config['zero_optimization'] = {"stage": 3, "offload_optimizer": {"device": "cpu", "pin_memory": True}, "offload_param": {"device": "cpu", "pin_memory": True}, "reduce_scatter": False}
    else:
        print(f"Error. Zero Stage must be None, 1, 2, or 3...")
        exit(1)



def get_dummy_dataset(seq_length=512, total_tokens=2**24):
    """Returns a TensorDataset."""
    print("Creating dummy dataset...", flush=True)
    num_samples = total_tokens // seq_length
    source_data = torch.randint(0, model_args.vocab_size, (num_samples, seq_length))
    target_data = torch.roll(source_data, shifts=-1, dims=1)
    target_data[:, -1] = -100
    dataset = TensorDataset(source_data, target_data)
    print("Dummy dataset created.")
    return dataset

# --- Initialization ---
print("Initializing model and DeepSpeed...", flush=True)
model = Model(model_args)

for name, param in model.named_parameters():
    if "norm" in name or name == "tok_embeddings.weight" or name == "output.weight":
        param.use_muon = False
    else:
        if use_muon:
            param.use_muon = True
        else:
            param.use_muon = False

# === CHANGE: Call the new function to get the dataset ===
dummy_dataset = get_dummy_dataset(seq_length=args.seq_len)

print("Initializing DeepSpeed...", flush=True)

print(f"Using deepspeed config:\n {ds_config}", flush=True)

# === CHANGE: Pass the dataset to training_data ===
model_engine, optimizer, training_dataloader, _ = deepspeed.initialize(
    args=args, # Pass the full args object
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
    training_data=dummy_dataset # Pass the Dataset here
)

act_layers_saved = select_bins(model_args.n_layers, save_act_layer_frac)

recompute_layers = []
for i in range(model_args.n_layers):
    if i not in act_layers_saved:
        recompute_layers.append(i)

num_checkpoints = len(recompute_layers)

print(f"Number of checkpointed layers: {num_checkpoints}", flush=True)
deepspeed.checkpointing.configure(
        mpu_=None,
        partition_activations=True,
        checkpoint_in_cpu=offload_act,
        contiguous_checkpointing=True,
        num_checkpoints=num_checkpoints
)

current_host_mem_gb = process.memory_info().rss / (1024 ** 3)
# Update peak if current is higher
peak_host_mem_gb = max(peak_host_mem_gb, current_host_mem_gb)



torch.cuda.empty_cache()

# --- Training Loop with Throughput Calculation ---
print(f"Starting training with sequence length: {args.seq_len}...", flush=True)

ret = _cudart.cudaProfilerStart()

start_time = time.time()
total_tokens = 0

num_steps = 0

step_throughputs = []

for epoch in range(epochs):
    # The training_dataloader is now the one created by DeepSpeed
    for i, (inputs, labels) in enumerate(training_dataloader):
        inputs = inputs.to(model_engine.device)
        labels = labels.to(model_engine.device)
        
        loss = model_engine(inputs, labels, save_act_layer_frac=save_act_layer_frac)
        #loss = criterion(outputs.view(-1, model_args.vocab_size), labels.view(-1))

        current_host_mem_gb = process.memory_info().rss / (1024 ** 3)
        # Update peak if current is higher
        peak_host_mem_gb = max(peak_host_mem_gb, current_host_mem_gb)

        model_engine.backward(loss)

        current_host_mem_gb = process.memory_info().rss / (1024 ** 3)
        # Update peak if current is higher
        peak_host_mem_gb = max(peak_host_mem_gb, current_host_mem_gb)

        model_engine.step()

        current_host_mem_gb = process.memory_info().rss / (1024 ** 3)
        # Update peak if current is higher
        peak_host_mem_gb = max(peak_host_mem_gb, current_host_mem_gb)

        num_steps += 1

        actual_bs = model_engine.train_micro_batch_size_per_gpu()
        total_tokens += actual_bs * args.seq_len
        
        if (i + 1) % grad_accum_steps == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time > 0:
                tokens_per_sec = total_tokens / elapsed_time
                step_throughputs.append(tokens_per_sec)
                max_reserve = torch.cuda.max_memory_reserved()
                max_alloc = torch.cuda.max_memory_allocated()
                print(
                    f"\n\nEpoch: {epoch+1}, Step: {i+1}, BS: {actual_bs}, Loss: {loss.item():.4f} | "
                    f"Step total_tokens: {total_tokens}, Step total_time = {elapsed_time}, Tok/sec: {tokens_per_sec:.2f} | Memory Max Alloc/Reserve: {max_alloc / (1 << 30)}/{max_reserve / (1 << 30)} GiB\n\n",
                    flush=True
                )
                start_time = time.time()
                total_tokens = 0


        if num_steps == global_steps:
            break

ret = _cudart.cudaProfilerStop()

peak_mem_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3

print(f"\n\n\nTraining complete! ✅\n\tThroughput: {step_throughputs[-1]} Tok/sec\n\tPeak Host Memory Reserved: {peak_host_mem_gb:.2f} GB\n\tPeak Device Memory Reserved: {peak_mem_reserved_gb:.2f} GB\n\n\n")
