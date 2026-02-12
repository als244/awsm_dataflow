# awsm_attention

Python wrapper around a C/CUDA flash attention library, supporting both Flash Attention 2 and Flash Attention 3 with automatic GPU architecture detection.

## Overview

This package provides a `FlashAttentionHelper` class that:

- Loads the native `libattentionwrapper.so` (which in turn links against `libflash2.so` and `libflash3.so`)
- Auto-detects GPU architecture (SM80/86/89/90) and SM count
- Accepts PyTorch tensors directly — no manual pointer juggling
- Allocates and frees GPU workspace automatically on each forward/backward call
- Uses the current CUDA stream from PyTorch

## Directory Structure

```
attention_helper/
├── Makefile                        # Top-level Makefile (existing)
├── setup.py                        # Builds native libs + installs Python package
├── pyproject.toml                  # Build system config for pip
├── README.md                       # This file
│
├── include/                        # C headers (existing)
│   └── attention_helper.h
│
├── src/                            # C source (existing)
│   └── attention_helper.c
│
├── flash2/                         # Flash Attention 2 (existing, own Makefile)
│   ├── Makefile
│   ├── include/
│   │   └── flash2_wrapper.h
│   └── lib/                        # libflash2.so built here
│
├── flash3/                         # Flash Attention 3 (existing, own Makefile)
│   ├── Makefile
│   ├── include/
│   │   └── flash3_wrapper.h
│   └── lib/                        # libflash3.so built here
│
├── objs/                           # Build artifacts (existing)
├── lib/                            # libattentionwrapper.so built here (existing)
│
├── awsm_attention/                 # Python package
│   ├── __init__.py                 # Exports FlashAttentionHelper
│   ├── attention.py                # Main wrapper class
│   └── lib/                        # .so files copied here at build time
│       └── __init__.py             # Empty (makes this a Python subpackage)
│
└── example.py                      # Usage example
```

## Prerequisites

- CUDA toolkit installed (default expected at `/usr/local/cuda`)
- A supported NVIDIA GPU: SM80 (A100), SM86 (A10/A40), SM89 (L4/L40), or SM90 (H100)
- `nvcc` available on `PATH`
- Python >= 3.8
- PyTorch with CUDA support

## Installation

From the `attention_helper/` directory:

```bash
# Standard install — builds native libs, installs package
pip install -v .

# Editable install — useful during development
pip install -v -e .

# Limit parallel build jobs if memory is tight
ATTENTION_BUILD_JOBS=4 pip install -v -e .
```

This will:

1. Run `make lib/libattentionwrapper.so`, which builds `libflash2.so`, `libflash3.so`, and `libattentionwrapper.so`
2. Copy all three `.so` files into `awsm_attention/lib/`
3. Install the `awsm_attention` Python package

### Verify installation

```bash
python -c "from awsm_attention import FlashAttentionHelper; print('Import OK')"
```

## Usage

```python
import torch
from awsm_attention import FlashAttentionHelper

device = torch.device("cuda:0")
dtype = torch.bfloat16

# Initialize (loads library, detects GPU arch)
helper = FlashAttentionHelper(device=device)

# Sequence setup — 2 sequences of lengths 128 and 256
seq_lens = [128, 256]
total_tokens = sum(seq_lens)

offsets = [0]
for s in seq_lens:
    offsets.append(offsets[-1] + s)

q_seq_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
k_seq_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
q_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
k_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)

# Allocate tensors
n_q_heads, n_kv_heads, head_dim = 32, 8, 128
q = torch.randn(total_tokens, n_q_heads, head_dim, dtype=dtype, device=device)
k = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)
v = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)
out = torch.empty(total_tokens, n_q_heads, head_dim, dtype=dtype, device=device)
softmax_lse = torch.empty(total_tokens, n_q_heads, dtype=torch.float32, device=device)

# Forward pass
helper.forward(
    q, k, v, out, softmax_lse,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q=max(seq_lens),
    max_seqlen_k=max(seq_lens),
    causal=True,
)

# Backward pass
dout = torch.randn_like(out)
dq = torch.empty_like(q)
dk = torch.empty_like(k)
dv = torch.empty_like(v)

helper.backward(
    dout, q, k, v, out, softmax_lse,
    dq, dk, dv,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q=max(seq_lens),
    max_seqlen_k=max(seq_lens),
    causal=True,
)
```

## API Reference

### `FlashAttentionHelper(lib_path=None, device=None)`

**Parameters:**

- `lib_path` (str, optional): Explicit path to `libattentionwrapper.so`. If not provided, the library is located automatically from the installed package. Can also be set via the `AWSM_ATTENTION_LIB` environment variable.
- `device` (torch.device, optional): CUDA device to use for architecture detection. Defaults to `torch.cuda.current_device()`.

**Attributes:**

- `arch` (int): GPU architecture (e.g., 90 for H100, 80 for A100)
- `sm_count` (int): Number of streaming multiprocessors

### `helper.forward(q, k, v, out, softmax_lse, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen_q, max_seqlen_k, causal=True)`

Runs the flash attention forward pass. Results are written into `out` and `softmax_lse` in-place.

**Tensor shapes:**

| Tensor | Shape | Dtype |
|---|---|---|
| `q` | `(total_q, n_q_heads, head_dim)` | bf16 / fp16 / fp32 |
| `k` | `(total_k, n_kv_heads, head_dim)` | bf16 / fp16 / fp32 |
| `v` | `(total_k, n_kv_heads, head_dim)` | bf16 / fp16 / fp32 |
| `out` | `(total_q, n_q_heads, head_dim)` | same as q |
| `softmax_lse` | `(total_q, n_q_heads)` | float32 |
| `q_seq_offsets` | `(num_seqs + 1,)` | int32 |
| `k_seq_offsets` | `(num_seqs + 1,)` | int32 |
| `q_seq_lens` | `(num_seqs,)` | int32 |
| `k_seq_lens` | `(num_seqs,)` | int32 |

### `helper.backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen_q, max_seqlen_k, causal=True)`

Runs the flash attention backward pass. Gradients are written into `dq`, `dk`, `dv` in-place.

All tensors have the same shape/dtype requirements as forward, plus:

| Tensor | Shape | Dtype |
|---|---|---|
| `dout` | `(total_q, n_q_heads, head_dim)` | same as q |
| `dq` | `(total_q, n_q_heads, head_dim)` | same as q |
| `dk` | `(total_k, n_kv_heads, head_dim)` | same as k |
| `dv` | `(total_k, n_kv_heads, head_dim)` | same as v |

### `helper.get_workspace_size(...)`

Query the required workspace size in bytes without running a kernel. Useful for pre-allocating memory.

## Configuration

### Environment Variables

| Variable | Description |
|---|---|
| `AWSM_ATTENTION_LIB` | Override path to `libattentionwrapper.so` |
| `ATTENTION_BUILD_JOBS` | Number of parallel make jobs during build (default: 8) |

### Build Flags

Override at install time:

```bash
CFLAGS="-O3 -fPIC" NVCC_FLAGS="-O4 --use_fast_math" pip install .

# Limit parallel build jobs (default is 8)
ATTENTION_BUILD_JOBS=4 pip install -v -e .
```

## Troubleshooting

**`FileNotFoundError: Cannot find libattentionwrapper.so`**

The native library wasn't built or copied correctly. Try reinstalling:

```bash
pip install --force-reinstall --no-cache-dir .
```

Or point to the library manually:

```bash
export AWSM_ATTENTION_LIB=/path/to/attention_helper/lib/libattentionwrapper.so
```

**`make` fails during install**

The CUDA toolkit or `nvcc` may not be on your PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**`RuntimeError: flash_attention_fwd failed with error code ...`**

Check that your GPU is one of the supported architectures (SM80, SM86, SM89, SM90) and that all input tensors are contiguous, on the same CUDA device, and have the correct dtypes.

**nvcc intermittent build failures**

The Makefile already retries flash2/flash3 builds up to 20 times. If builds still fail, check disk space and available memory — `nvcc` can be resource-hungry.