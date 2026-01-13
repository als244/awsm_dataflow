import torch
import time
import sys
from matmul_dispatcher import CublasLtDispatcher

# --- Formatting ---
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def log_pass(msg):
    print(f"{GREEN}[PASS]{RESET} {msg}")

def log_fail(msg):
    print(f"{RED}[FAIL]{RESET} {msg}")

def get_tensors(M, N, K, dtype, trans_a, trans_b, device='cuda'):
    if trans_a:
        A = torch.randn((K, M), device=device, dtype=dtype).transpose(-1, -2)
    else:
        A = torch.randn((M, K), device=device, dtype=dtype)

    if trans_b:
        B = torch.randn((N, K), device=device, dtype=dtype).transpose(-1, -2)
    else:
        B = torch.randn((K, N), device=device, dtype=dtype)
        
    return A, B

def verify_result(ref, out, dtype, msg):
    atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    
    if torch.allclose(ref, out, atol=atol, rtol=rtol):
        log_pass(msg)
        return True
    else:
        diff = (ref - out).abs().max().item()
        log_fail(f"{msg} | Max Diff: {diff}")
        return False

# --- Tests ---

def test_basic_correctness(dispatcher):
    print(f"\n{YELLOW}--- 1. Basic Correctness (Standard, Transpose, Bias) ---{RESET}")
    
    dtypes = [torch.float16, torch.bfloat16]
    shape = (128, 128, 128)
    M, N, K = shape
    
    # Get current stream
    stream = torch.cuda.current_stream()
    
    for dt in dtypes:
        # A. Standard A @ B
        A, B = get_tensors(M, N, K, dt, False, False)
        out = dispatcher.matmul(stream, A, B)
        ref = torch.matmul(A, B)
        verify_result(ref, out, dt, f"Standard {dt}")

        # B. Transposed A.T @ B
        A_t, B = get_tensors(M, N, K, dt, True, False)
        out = dispatcher.matmul(stream, A_t, B)
        ref = torch.matmul(A_t, B)
        verify_result(ref, out, dt, f"TransA   {dt}")

        # C. Bias (C) Support: D = AB + C
        C = torch.randn((M, N), device='cuda', dtype=dt)
        ref = torch.addmm(C, A, B, beta=1.0, alpha=1.0) 
        out = dispatcher.matmul(stream, A, B, C=C, beta=1.0)
        verify_result(ref, out, dt, f"Bias Add {dt}")

def test_explicit_output(dispatcher):
    print(f"\n{YELLOW}--- 2. Explicit Output D (Pre-allocation) ---{RESET}")
    M, N, K = 1024, 1024, 1024
    dt = torch.float16
    A = torch.randn((M, K), device='cuda', dtype=dt)
    B = torch.randn((K, N), device='cuda', dtype=dt)
    
    stream = torch.cuda.current_stream()
    
    # Pre-allocate D with zeros
    D = torch.zeros((M, N), device='cuda', dtype=dt)
    original_ptr = D.data_ptr()
    
    # Run Dispatcher
    ret_D = dispatcher.matmul(stream, A, B, D=D)
    
    # Check 1: Did it return the same object?
    if ret_D.data_ptr() == original_ptr:
        log_pass("Returned tensor matches input pointer")
    else:
        log_fail("Dispatcher allocated new tensor instead of using D")

    # Check 2: Correctness
    ref = torch.matmul(A, B)
    verify_result(ref, D, dt, "Pre-allocated Result Correctness")

def test_inplace_accumulation(dispatcher):
    print(f"\n{YELLOW}--- 3. In-Place Accumulation (D = AB + D) ---{RESET}")
    M, N, K = 512, 512, 512
    dt = torch.float16
    
    stream = torch.cuda.current_stream()
    
    A = torch.randn((M, K), device='cuda', dtype=dt)
    B = torch.randn((K, N), device='cuda', dtype=dt)
    D = torch.randn((M, N), device='cuda', dtype=dt)
    
    D_clone = D.clone()
    
    # Reference: D = 1.0*D + 1.0*(A@B)
    ref = torch.addmm(D_clone, A, B, beta=1.0, alpha=1.0)
    
    # Dispatch: Pass D as both C (source of bias) and D (output)
    dispatcher.matmul(stream, A, B, C=D, D=D, beta=1.0, alpha=1.0)
    
    verify_result(ref, D, dt, "In-Place Accumulation Correctness")

def test_rounding_and_caching(dispatcher):
    print(f"\n{YELLOW}--- 4. Cache Rounding & Stats ---{RESET}")
    
    M1, N, K = 1000, 1024, 1024
    M2       = 1001 # Different M, but rounds to 1024
    dt = torch.float16
    
    stream = torch.cuda.current_stream()
    
    A1 = torch.randn((M1, K), device='cuda', dtype=dt)
    B  = torch.randn((K, N), device='cuda', dtype=dt)
    A2 = torch.randn((M2, K), device='cuda', dtype=dt)
    
    # 1. Warm up
    dispatcher.matmul(stream, torch.randn(32,32,device='cuda',dtype=dt), 
                      torch.randn(32,32,device='cuda',dtype=dt))
    
    print(">> Run 1: Shape (1000, 1024, 1024) -> Should be Cold Start")
    dispatcher.matmul(stream, A1, B)
    stats1 = dispatcher.get_stats()
    print(f"   Stats: {stats1}")
    
    # Check Algos Saved (Assuming warmup created 1, run 1 created 1 -> total 2)
    # We just care that it's > 0
    if stats1['algos_saved'] > 0:
        log_pass("Algo saved correctly")
    
    print(">> Run 2: Shape (1001, 1024, 1024) -> Should be CACHE HIT (due to rounding)")
    dispatcher.matmul(stream, A2, B)
    stats2 = dispatcher.get_stats()
    print(f"   Stats: {stats2}")

    # Logic: 
    # algos_saved should remain SAME (reused the old key)
    # algo_hits should increase
    if stats2['algos_saved'] == stats1['algos_saved'] and stats2['algo_hits'] > stats1['algo_hits']:
        log_pass("Rounding Logic Worked! (Reused algo for slightly different shape)")
    else:
        log_fail("Rounding Logic Failed (Created new algo or didn't hit)")

def test_performance(dispatcher):
    print(f"\n{YELLOW}--- 5. Overhead Analysis ---{RESET}")
    M, N, K = 128, 128, 128
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    D = torch.empty(M, N, device='cuda', dtype=torch.float16)
    
    # Use integer stream for max perf
    stream_ptr = torch.cuda.current_stream().cuda_stream
    
    # Warmup
    for _ in range(100):
        dispatcher.matmul(stream_ptr, A, B, D=D)
        
    start = time.perf_counter_ns()
    for _ in range(1000):
        dispatcher.matmul(stream_ptr, A, B, D=D)
    end = time.perf_counter_ns()
    
    avg_us = (end - start) / 1000 / 1000.0
    print(f"Total Wall Time per Call: {avg_us:.3f} us")
    
    stats = dispatcher.get_stats()
    print("Internal Stats Breakdown:")
    print(f"  > Python Wrapper:  {stats['avg_wrapper_overhead_us']:.3f} us")
    print(f"  > C++ Total:       {stats['avg_cpp_total_us']:.3f} us")
    print(f"  \t> BLAS Submission Latency:  {stats['breakdown']['driver_submit_us']:.3f} us")
    print(f"  \t> C++ Hash Logic:       {stats['breakdown']['cpp_hash_logic_us']:.3f} us")
    
def main():
    try:
        # Initialize dispatcher once
        dispatcher = CublasLtDispatcher(round_multiple=32)
    except Exception as e:
        log_fail(f"Failed to load dispatcher: {e}")
        return

    test_basic_correctness(dispatcher)
    test_explicit_output(dispatcher)
    test_inplace_accumulation(dispatcher)
    
    # Re-init for clean stats
    dispatcher = CublasLtDispatcher(round_multiple=32)
    test_rounding_and_caching(dispatcher)
    
    # Performance test
    dispatcher = CublasLtDispatcher(round_multiple=32)
    test_performance(dispatcher)

if __name__ == "__main__":
    main()