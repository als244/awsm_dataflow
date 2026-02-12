"""
Example usage of awsm_attention.FlashAttentionHelper
"""

import torch
from awsm_attention import FlashAttentionHelper

# --- Setup ------------------------------------------------------------------
device = torch.device("cuda:0")
dtype = torch.bfloat16

# Create helper (loads libattentionwrapper.so, auto-detects GPU arch)
helper = FlashAttentionHelper(device=device)
print(f"GPU arch: SM{helper.arch}, SM count: {helper.sm_count}")

# --- Fake data for 2 sequences: lengths 128 and 256 -----------------------
seq_lens = [128, 256]
num_seqs = len(seq_lens)
total_tokens = sum(seq_lens)
max_seqlen = max(seq_lens)

n_q_heads = 32
n_kv_heads = 8   # GQA
head_dim = 128

# Build offset arrays (cumsum with leading 0)
offsets = [0]
for s in seq_lens:
    offsets.append(offsets[-1] + s)

q_seq_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
k_seq_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
q_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
k_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)

# Allocate tensors
q = torch.randn(total_tokens, n_q_heads, head_dim, dtype=dtype, device=device)
k = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)
v = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)
out = torch.empty(total_tokens, n_q_heads, head_dim, dtype=dtype, device=device)
softmax_lse = torch.empty(n_q_heads, total_tokens, dtype=torch.float32, device=device)

# --- Forward ----------------------------------------------------------------
helper.forward(
    q, k, v, out, softmax_lse,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    causal=True,
)
torch.cuda.synchronize()
print(f"Forward done. out shape: {out.shape}, lse shape: {softmax_lse.shape}")

# --- Backward ---------------------------------------------------------------
dout = torch.randn_like(out)
dq = torch.empty_like(q)
dk = torch.empty_like(k)
dv = torch.empty_like(v)

helper.backward(
    dout, q, k, v, out, softmax_lse,
    dq, dk, dv,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    causal=True,
)
torch.cuda.synchronize()
print(f"Backward done. dq shape: {dq.shape}, dk shape: {dk.shape}, dv shape: {dv.shape}")