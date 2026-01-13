import torch
import ctypes
import time
import os
import sys

# --- 1. Load the Shared Library ---

_LIB_NAME = "libmatmul_dispatcher.so"
_LIB_PATH = os.path.join(os.path.dirname(__file__), _LIB_NAME)

# Windows fallback check
if sys.platform == 'win32':
    _win_path = os.path.join(os.path.dirname(__file__), "libmatmul_dispatcher.dll")
    if os.path.exists(_win_path):
        _LIB_PATH = _win_path

# Strict check
if not os.path.exists(_LIB_PATH):
    raise FileNotFoundError(
        f"Could not find shared library at: {_LIB_PATH}\n"
        "If you installed via 'pip install -e .', ensure setup.py ran successfully "
        "and copied the .so file to the source folder."
    )

try:
    _lib = ctypes.CDLL(_LIB_PATH)
except OSError as e:
    raise OSError(
        f"Found library at {_LIB_PATH} but failed to load it.\n"
        f"System Error: {e}\n"
        "Check CUDA dependencies (libcublasLt, libcudart) are in your LD_LIBRARY_PATH."
    )

# --- 2. Define C Argument Types ---

_lib.create_dispatcher.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.create_dispatcher.restype = ctypes.c_void_p

_lib.destroy_dispatcher.argtypes = [ctypes.c_void_p]
_lib.destroy_dispatcher.restype = None

_lib.get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
_lib.get_stats.restype = None

_lib.dispatch_matmul.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # M, N, K
    ctypes.c_void_p, ctypes.c_int,             # ptr_A, lda
    ctypes.c_void_p, ctypes.c_int,             # ptr_B, ldb
    ctypes.c_void_p, ctypes.c_int,             # ptr_C, ldc
    ctypes.c_void_p, ctypes.c_int,             # ptr_D, ldd
    ctypes.c_void_p, ctypes.c_size_t,          # Workspace
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, # Dtypes
    ctypes.c_float, ctypes.c_float,            # Alpha, Beta
    ctypes.c_int, ctypes.c_int                 # TransA, TransB
]
_lib.dispatch_matmul.restype = ctypes.c_int


# --- 3. Helper for Layout Extraction ---

def _get_layout(t):
    """
    Analyzes tensor strides to return (ptr, ld, trans_flag).
    Raises error if memory layout is unsupported (non-strided/irregular).
    """
    if t.dim() != 2:
        raise ValueError("Only 2D tensors supported for matmul dispatcher.")
        
    stride_row, stride_col = t.stride()
    
    # CASE 1: Row Major (Standard)
    # The stride of the last dimension is 1. 
    # The 'Leading Dimension' (stride between rows) is stride_row.
    if stride_col == 1:
        return t.data_ptr(), stride_row, 0
    
    # CASE 2: Column Major (Transposed)
    # The stride of the first dimension is 1.
    # Physically, this is stored as [Cols, Rows] row-major.
    # The 'Leading Dimension' is stride_col.
    elif stride_row == 1:
        return t.data_ptr(), stride_col, 1
        
    else:
        raise ValueError(
            f"Tensor memory layout not supported (Strides: {t.stride()}). "
            "Must be Row-Major (stride[:,-1]==1) or Col-Major (stride[:,0]==1)."
        )


# --- 4. Python Wrapper Class ---

class CublasLtDispatcher:
    def __init__(self, round_multiple=32, max_algos=5):
        self.round_multiple = round_multiple
        self.max_algos = max_algos
        self.ctx = _lib.create_dispatcher(round_multiple, max_algos)
        self.workspace_size = 64 * 1024 * 1024 
        self.workspace = torch.empty(self.workspace_size, dtype=torch.uint8, device='cuda')
        self.ws_ptr = self.workspace.data_ptr()
        self.fp16 = torch.float16
        self.bf16 = torch.bfloat16
        self.total_python_ns = 0

    def __del__(self):
        if hasattr(self, 'ctx') and self.ctx:
            _lib.destroy_dispatcher(self.ctx)
            self.ctx = None

    def get_stats(self):
        stats_array = (ctypes.c_uint64 * 7)()
        _lib.get_stats(self.ctx, stats_array)
        calls = stats_array[1]
        if calls == 0: return {
            "matmuls_called": 0,
            "algos_saved": 0,
            "algo_hits": 0,
            "avg_wrapper_overhead_us": 0,
            "avg_cpp_total_us": 0,
            "breakdown": {"driver_submit_us": 0, "cpp_hash_logic_us": 0}
        }
        
        total_cpp_ns = stats_array[3]
        total_driver_ns = stats_array[6]
        
        return {
            "matmuls_called": calls,
            "algos_saved": stats_array[0],
            "algo_hits": stats_array[2],
            "avg_wrapper_overhead_us": (self.total_python_ns / calls / 1000.0) - (total_cpp_ns / calls / 1000.0),
            "avg_cpp_total_us": total_cpp_ns / calls / 1000.0,
            "breakdown": {
                "driver_submit_us": total_driver_ns / calls / 1000.0,
                "cpp_hash_logic_us": (total_cpp_ns - total_driver_ns) / calls / 1000.0
            }
        }

    def matmul(self, stream_ptr, A, B, C=None, D=None, alpha=1.0, beta=0.0):

        ## D = alpha * A @ B + beta * C

        t0 = time.perf_counter_ns()

        # 1. Inspect Inputs (No Copies Created)
        ptr_A, ld_A, trans_a = _get_layout(A)
        ptr_B, ld_B, trans_b = _get_layout(B)

        # 2. Determine Logical Dimensions M, N, K
        M, K = A.shape
        K_B, N = B.shape
        
        if K != K_B:
            raise ValueError(f"Shape Mismatch: A is {A.shape}, B is {B.shape}. Inner dims {K} vs {K_B} must match.")
            
        # 3. Output Handling (D)
        if D is None:
            D = torch.empty((M, N), dtype=A.dtype, device=A.device)
        
        ptr_D, ld_D, trans_d = _get_layout(D)
        
        if trans_d != 0:
             raise ValueError("Output tensor D must be Row-Major for this dispatcher.")

        # 4. Optional C Handling
        if C is not None:
            ptr_C, ld_C, trans_c = _get_layout(C)
            if trans_c != 0:
                raise ValueError("Bias/Add tensor C must be Row-Major.")
        else:
            ptr_C = 0
            ld_C = N

        if M == 0 or N == 0 or K == 0:
            raise ValueError(f"All dimensions must be non-zero. Got M={M}, N={N}, K={K}")

        dt_enum = 1 if A.dtype == self.fp16 else (2 if A.dtype == self.bf16 else 0)

        # 5. Dispatch
        ret = _lib.dispatch_matmul(
            self.ctx, ctypes.c_void_p(stream_ptr), M, N, K,
            ctypes.c_void_p(ptr_A), ld_A,
            ctypes.c_void_p(ptr_B), ld_B,
            ctypes.c_void_p(ptr_C), ld_C,
            ctypes.c_void_p(ptr_D), ld_D,
            ctypes.c_void_p(self.ws_ptr), self.workspace_size,
            dt_enum, dt_enum, dt_enum, dt_enum, 0, alpha, beta, 
            trans_a, trans_b
        )

        if ret != 0:
            raise RuntimeError("Failed to dispatch matmul")
        
        self.total_python_ns += (time.perf_counter_ns() - t0)
        return D