# Matmul Dispatcher

A high-performance, lightweight Python wrapper for NVIDIA's `cublasLt` library.

This project bypasses the standard PyTorch linear layer overhead by dispatching matrix multiplications directly via **Ctypes**. It implements a C++ **heuristic caching strategy** to minimize the overhead of finding the optimal cuBLAS algorithm for recurring matrix shapes.

## Features

* **Low Overhead:** Uses Ctypes to call C-API directly, avoiding Python C-Extension bloat.
* **Heuristic Caching:** Caches `cublasLt` algorithm descriptors based on matrix geometry.
* **Fuzzy Matching:** Groups similar matrix shapes (e.g., batch size 1020 and 1024) to reduce cache misses and setup time.
* **Persistent Workspace:** Manages a single GPU workspace buffer to avoid repeated allocations.

## Prerequisites

Before installing, ensure you have the following in your environment:

1.  **NVIDIA CUDA Toolkit** (standard installation usually puts this in `/usr/local/cuda`).
2.  **CMake** (Version 3.18 or higher).
3.  **Python 3.x** with **PyTorch** installed.

## Installation

Cloning the repository and navigating to the root folder:

```bash
git clone <your-repo-url>
cd matmul_dispatcher
```

### Option 1: Standard Installation (Recommended)

This compiles the C++ backend and installs the Python package into your current environment.

```bash
pip install .
```

* *What this does:* It runs `setup.py`, invokes `cmake` to build `libmatmul_dispatcher.so`, and installs the package so you can `import matmul_dispatcher` from anywhere.

### Option 2: Editable / Developer Mode

If you plan to modify the Python wrapper or the C++ code frequently:

```bash
pip install -e .
```

* *Note:* If you modify `src/dispatch.cpp`, you must run `pip install -e .` again (or run `cmake` manually) to recompile the shared object file. Changes to Python files will reflect immediately.

### Option 3: Manual Build (For C++ Debugging)

If you want to build the shared library without installing the Python package:

```bash
cmake .
make -j4
```

This will generate `matmul_dispatcher/libmatmul_dispatcher.so`. You can then run scripts in the root directory.

## Usage

```python
import torch
from matmul_dispatcher import CublasLtDispatcher

# 1. Initialize the dispatcher
# round_multiple=32 means shapes like 1020 and 1024 share the same heuristic
dispatcher = CublasLtDispatcher(round_multiple=32)

# 2. Prepare Data (Must be on GPU)
M, N, K = 4096, 4096, 4096
a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)

# 3. Dispatch
# You must provide a CUDA stream (torch.cuda.current_stream() is standard)
stream = torch.cuda.current_stream()

c = dispatcher.matmul(stream, a, b)

# 4. View Performance Stats
# Check how many times we hit the cache vs ran the heuristic
stats = dispatcher.get_stats()
print("Performance Stats:", stats)
```

## Troubleshooting

**`FileNotFoundError: Could not find libmatmul_dispatcher.so`**
* Ensure you ran `pip install .` or `cmake . && make`.
* Check that the build process finished successfully.

**`ImportError: No module named 'matmul_dispatcher'`**
* Ensure you are in the correct environment.
* If you didn't install via pip, ensure you are in the project root directory.

**CMake Error: `CUDAToolkit not found`**
* Ensure `nvcc` is in your PATH (`which nvcc`).
* If installed in a non-standard location, export the path: `export CUDACXX=/path/to/nvcc`.

## Project Structure

```text
matmul_dispatcher/
├── CMakeLists.txt           # Build configuration finding CUDA
├── setup.py                 # Python installer wrapper around CMake
├── src/
│   ├── dispatch.h           # Pure C interface
│   └── dispatch.cpp         # C++ implementation & caching logic
└── matmul_dispatcher/       # Python Package
    └── __init__.py          # Ctypes wrapper & Entry point
```