# train.py (Updated - Dense Model Training)

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import argparse
import time

import psutil
import os

import ctypes

import json
import pickle
import sys

from model import Transformer, ModelArgs, SeqlensInfo

import sequence

from model import SAVED_ACT_GRADS


device = torch.device("cuda:0")


all_model_dims = json.load(open("../model_dims.json"))


MODEL_CHOICE = "llama3_8B"

model_dims = all_model_dims[MODEL_CHOICE]


model_args = ModelArgs(
    vocab_size=model_dims["vocab_size"],
    n_layers=n_layers,
    dim=model_dims["embed_dim"],
    head_dim=model_dims["head_dim"],
    n_heads=model_dims["n_heads"],
    n_kv_heads=model_dims["n_kv_heads"],
    expert_dim=model_dims["expert_dim"],
)

model = Transformer(model_args).to(device)

INIT_MODEL_PATH = f"../init_models/init_{MODEL_CHOICE}"

TRAIN_SEQ_PATH = f"{INIT_MODEL_PATH}/train_seqs"

model.load_model_weights(INIT_MODEL_PATH)


train_seqs = []


### USING CONSTANT HYPERPARAMS FOR SIMPLICITY
opt_hyperparams = {
    "lr": 3e-4,
    "beta1": 0.95,
    "beta2": 0.98,
    "eps": 1e-8,
    "weight_decay": 0.0,
    "step_num": 0,
}

optimizer = optim.AdamW(
    model.parameters(),
    lr=opt_hyperparams["lr"],
    betas=(opt_hyperparams["beta1"], opt_hyperparams["beta2"]),
    eps=opt_hyperparams["eps"],
    weight_decay=opt_hyperparams["weight_decay"]
)

criterion = torch.nn.CrossEntropyLoss()

NUM_STEPS = 100

for step_num in range(1, NUM_STEPS + 1):
    train_seqs.append(pickle.load(open(f"{TRAIN_SEQ_PATH}/step_{step_num}.pkl", "rb")))


step_num = 1

SAVE_STEPS = [1, 2, 3, 4, 5, 10, 20, 50, 100]

SAVE_PATH = "../checkpoints/dense_model"

MAX_TOKENS_PER_BATCH = 8192

for step_seqs in train_seqs:
    
    # First, build batches that respect the token limit
    batches = []
    current_batch = []
    current_batch_tokens = 0
    
    for seq in step_seqs:
        seq_len = len(seq)
        
        # If adding this sequence would exceed limit, start a new batch
        if current_batch_tokens + seq_len > MAX_TOKENS_PER_BATCH and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_batch_tokens = 0
        
        current_batch.append(seq)
        current_batch_tokens += seq_len
    
    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)
    
    num_batches = len(batches)
    optimizer.zero_grad()
    
    total_loss = 0.0
    total_tokens = 0
    
    for batch_idx, batch_seqs in enumerate(batches):
        cur_batch_seqlens = []
        input_tokens = []
        target_tokens = []
        
        for seq in batch_seqs:
            cur_batch_seqlens.append(len(seq))
            input_tokens += [t for t in seq.tokens]
            target_tokens += [t for t in seq.targets]
        
        batch_num_tokens = len(input_tokens)
        cur_batch_seqlens_np = np.array(cur_batch_seqlens)
        
        input_tokens = torch.tensor(input_tokens, device=device).long()
        target_tokens = torch.tensor(target_tokens, device=device).long()
        
        seqlens_info = SeqlensInfo(cur_batch_seqlens_np, device)
        
        output = model(input_tokens, seqlens_info, step_num)
        
        loss = criterion(output, target_tokens)
        
        # Scale loss by the proportion of tokens in this batch
        # This ensures proper gradient averaging across the full step
        total_tokens += batch_num_tokens
        total_loss += loss.item() * batch_num_tokens
        
        # Scale gradients for accumulation
        scaled_loss = loss / num_batches
        scaled_loss.backward()
    
    # Step the optimizer after all batches are processed
    optimizer.step()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    print(f"Step {step_num}: Loss: {avg_loss:.6f} ({num_batches} batches, {total_tokens} tokens)", flush=True)


    if step_num in SAVE_STEPS:
        ## save model weights, gradients, and optimizer state!
        
        # 1. Ensure the directory exists
        os.makedirs(SAVE_PATH, exist_ok=True)

        # 2. Manually collect gradients (state_dict does not save these)
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Detach and move to CPU to save GPU memory and storage space
                grads[name] = param.grad.detach().cpu()

        # 3. Create the checkpoint dictionary
        checkpoint = {
            "step_num": step_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "gradients": grads,
            "loss": loss.item()
        }

        # 4. Save to disk
        save_file = f"{SAVE_PATH}/step_{step_num}.pt"
        torch.save(checkpoint, save_file)
        
        print(f"--> Saved checkpoint, optimizer, and gradients to {save_file}", flush=True)

    step_num += 1
