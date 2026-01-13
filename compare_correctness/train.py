# train.py (Corrected)

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

from model import Transformer, ModelArgs, SeqlensInfo

import sequence

from model import SAVED_ACT_GRADS


n_layers = 16

device = torch.device("cuda:0")

model_dims = {
    "vocab_size": 50257,
    "embed_dim": 2048,
    "n_heads": 16,
    "n_kv_heads": 4,
    "head_dim": 128,
    "expert_dim": 8192
}

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

model.load_model_weights("../my_models/init_1B")


train_seqs = []

opt_hyperparams = {
    "lr": 1e-4,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.0,
    "step_num": 0,
}

optimizer = optim.AdamW(
    model.parameters(), # Pass all model parameters to the optimizer
    lr=opt_hyperparams["lr"],
    betas=(opt_hyperparams["beta1"], opt_hyperparams["beta2"]), # (beta1, beta2)
    eps=opt_hyperparams["eps"],           # epsilon
    weight_decay=opt_hyperparams["weight_decay"]
)

criterion = torch.nn.CrossEntropyLoss()

NUM_STEPS = 100

for step_num in range(1, NUM_STEPS + 1):
    train_seqs.append(pickle.load(open(f"../train_seqs/step_{step_num}.pkl", "rb")))


step_num = 1

for step_seqs in train_seqs:

    cur_step_seqlens = []
    input_tokens = []
    target_tokens = []
    for seq in step_seqs:
        cur_step_seqlens.append(len(seq))
        input_tokens += [t for t in seq.tokens]
        target_tokens += [t for t in seq.targets]

    cur_step_seqlens_np = np.array(cur_step_seqlens)
    
    input_tokens = torch.tensor(input_tokens, device=device).long()
    target_tokens = torch.tensor(target_tokens, device=device).long()

    seqlens_info = SeqlensInfo(cur_step_seqlens_np, device)

    output = model(input_tokens, seqlens_info, step_num)

    loss = criterion(output, target_tokens)

    print(f"Step {step_num}: Loss: {loss.item()}", flush=True)

    optimizer.zero_grad()
    loss.backward()

    # head_grad_proj = model.output.weight.grad
    # if head_grad_proj is not None:
    #     print(f"Head grad proj (norm): {head_grad_proj.norm()}", flush=True)
    
    # head_attn_norm_grad = model.norm.weight.grad
    # if head_attn_norm_grad is not None:
    #     print(f"Head attn norm grad (norm): {head_attn_norm_grad.norm()}", flush=True)
   

    # final_layer_grad = model.layers[-1].attention_norm.weight.grad
    # if final_layer_grad is not None:
    #     print(f"Final layer g_attn_norm (norm): {final_layer_grad.norm()}", flush=True)

    # final_layer_grad = model.layers[-1].attention.wq.weight.grad
    # if final_layer_grad is not None:
    #     print(f"Final layer g_q (norm): {final_layer_grad.norm()}", flush=True)

    # final_layer_grad = model.layers[-1].attention.wk.weight.grad
    # if final_layer_grad is not None:
    #     print(f"Final layer g_k (norm): {final_layer_grad.norm()}", flush=True)

    # final_layer_grad = model.layers[-1].attention.wv.weight.grad
    # if final_layer_grad is not None:
    #     print(f"Final layer g_v (norm): {final_layer_grad.norm()}", flush=True)
    
    # final_layer_grad = model.layers[-1].attention.wo.weight.grad
    # if final_layer_grad is not None:
    #     print(f"Final layer g_o (norm): {final_layer_grad.norm()}", flush=True)

    # final_layer_grad = model.layers[-1].ffn_norm.weight.grad
    # if final_layer_grad is not None:
    #     print(f"Final layer g_ffn_norm (norm): {final_layer_grad.norm()}", flush=True)

    # final_layer_grad = model.layers[-1].feed_forward.w1.weight.grad
    # if final_layer_grad is not None:
    #     print(f"Final layer g_1 (norm): {final_layer_grad.norm()}", flush=True)

    # final_layer_grad = model.layers[-1].feed_forward.w3.weight.grad
    # if final_layer_grad is not None:
    #     print(f"Final layer g_3 (norm): {final_layer_grad.norm()}", flush=True)

    # final_layer_grad = model.layers[-1].feed_forward.w2.weight.grad
    # if final_layer_grad is not None:
    #     print(f"Final layer g_2 (norm): {final_layer_grad.norm()}", flush=True)

    
    # for name, tensor in SAVED_ACT_GRADS.items():
    #     torch.save(tensor, f"saved_act_grads_{name}.pt")

    

    # for layer_id, layer in enumerate(model.layers):
    #     attention_norm_grad = layer.attention_norm.weight.grad
    #     if attention_norm_grad is not None:
    #         print(f"L{layer_id}: g_attn_norm (norm): {attention_norm_grad.norm()}", flush=True)
    #     else:
    #         print(f"L{layer_id}: g_attn_norm (norm): None", flush=True)

    # for layer_id, layer in enumerate(model.layers):
    #     # Access the gradient of the attention_norm weight
    #     grad = layer.attention_norm.weight.grad
        
    #     if grad is not None:
    #         # Calculate norm (default is L2/Euclidean)
    #         # .item() converts the 1-element tensor to a standard Python float for cleaner printing
    #         print(f"L{layer_id}: g_attn_norm (norm): {grad.norm()}", flush=True)
    #     else:
    #         print(f"L{layer_id}: g_attn_norm (norm): None", flush=True)


    optimizer.step()

    if step_num == 100:
        break

    step_num += 1
    