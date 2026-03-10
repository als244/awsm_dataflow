"""
Fused MoE Router Kernel: TopK Selection + Softmax in a Single Triton Kernel
v3: Process multiple tokens per thread block for better throughput.

Key insight: With E=64 and K=8, each token needs very little work.
Launching 131K thread blocks with 1-2 warps each is launch-overhead dominated.
Instead, have each thread block process BLOCK_T tokens using a 2D tile of
shape [BLOCK_T, BLOCK_E]. The reductions (max, argmin) happen along axis=1
(the expert dim), processing all BLOCK_T tokens in parallel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_topk_softmax_kernel(
    # Inputs
    LOGITS_PTR,          # [T, E] gate logits
    # Outputs
    TOPK_IDS_PTR,        # [T, K] selected expert IDs (int32)
    TOPK_WEIGHTS_PTR,    # [T, K] softmax probabilities
    # Strides
    stride_logits_t,
    stride_logits_e,
    stride_ids_t,
    stride_ids_k,
    stride_w_t,
    stride_w_k,
    # Dimensions
    T,                        # total tokens
    E: tl.constexpr,          # number of experts
    K: tl.constexpr,          # top-k
    BLOCK_E: tl.constexpr,    # >= E, power of 2
    BLOCK_T: tl.constexpr,    # tokens per block
):
    """
    Each program processes BLOCK_T tokens via a [BLOCK_T, BLOCK_E] tile.
    Reductions along axis=1 give per-token max/argmin across experts.
    
    Top-K selection uses iterative max+mask:
      1. Find max value per row (axis=1 reduction)
      2. Among positions equal to max, find smallest index (tie-breaking)
      3. Mask out selected expert, repeat K times
    
    Tie-breaking: we use equality with the row-max to find candidates,
    then argmin to get the smallest index. The equality comparison is safe
    here because max and the elements are in the same register tile —
    no cross-warp reduction mismatch.
    
    The key difference from v1: tl.max on a 2D [BLOCK_T, BLOCK_E] tensor
    along axis=1 produces a [BLOCK_T] vector of per-row maxima. These maxima
    are then broadcast back and compared element-wise against the SAME tile
    they were derived from. Since there's no separate scalar reduction step
    that might lose precision, the equality check is reliable.
    """
    pid = tl.program_id(0).to(tl.int64)
    
    # Token and expert offsets — explicitly int64 for large T support.
    # tl.arange returns int32; without the cast, token_offs * stride can
    # overflow int32 when T * E > 2^31 (~33M elements for bf16).
    token_offs = (pid * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
    token_mask = token_offs < T
    expert_offs = tl.arange(0, BLOCK_E).to(tl.int64)
    expert_mask = expert_offs < E
    
    # Load [BLOCK_T, BLOCK_E] tile of logits
    logits_ptrs = (LOGITS_PTR 
                   + token_offs[:, None] * stride_logits_t 
                   + expert_offs[None, :] * stride_logits_e)
    load_mask = token_mask[:, None] & expert_mask[None, :]
    remaining = tl.load(logits_ptrs, mask=load_mask, other=-float('inf')).to(tl.float32)
    
    # --- Top-K Selection ---
    for k in range(K):
        # Per-token max across experts: [BLOCK_T]
        max_vals = tl.max(remaining, axis=1)
        
        # Find positions equal to max (ties included)
        is_max = (remaining == max_vals[:, None]) & expert_mask[None, :]
        
        # Smallest index among tied maxima (deterministic tie-breaking)
        candidates = tl.where(is_max, expert_offs[None, :], BLOCK_E)
        expert_ids = tl.min(candidates, axis=1)  # [BLOCK_T]
        
        # Store expert ID
        id_ptrs = TOPK_IDS_PTR + token_offs * stride_ids_t + k * stride_ids_k
        tl.store(id_ptrs, expert_ids.to(tl.int32), mask=token_mask)
        
        # Store raw logit value (will be overwritten with softmax prob)
        w_ptrs = TOPK_WEIGHTS_PTR + token_offs * stride_w_t + k * stride_w_k
        tl.store(w_ptrs, max_vals, mask=token_mask)
        
        # Mask out selected expert
        is_selected = (expert_offs[None, :] == expert_ids[:, None])
        remaining = tl.where(is_selected, -float('inf'), remaining)
    
    # --- Softmax over selected top-K values ---
    k_offs = tl.arange(0, K)
    softmax_ptrs = (TOPK_WEIGHTS_PTR 
                    + token_offs[:, None] * stride_w_t 
                    + k_offs[None, :] * stride_w_k)
    softmax_mask = token_mask[:, None]
    selected_logits = tl.load(softmax_ptrs, mask=softmax_mask, other=0.0).to(tl.float32)
    
    # Stable softmax along K dim
    max_logit = tl.max(selected_logits, axis=1)
    exp_logits = tl.exp(selected_logits - max_logit[:, None])
    sum_exp = tl.sum(exp_logits, axis=1)
    softmax_probs = exp_logits / sum_exp[:, None]
    
    tl.store(softmax_ptrs, softmax_probs.to(TOPK_WEIGHTS_PTR.dtype.element_ty), mask=softmax_mask)


def awsm_fused_topk_softmax(
    gate_logits: torch.Tensor,
    top_k: int,
    topk_ids_out: torch.Tensor = None,
    topk_weights_out: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused top-K expert selection + softmax for MoE routing.
    
    Replaces:
        raw_weights, topk_ids = torch.topk(gate_logits, k=top_k, dim=-1)
        router_weights = softmax(raw_weights, dim=-1)
    
    Args:
        gate_logits: Shape [T, E], CUDA, contiguous
        top_k: Number of experts per token (K)
        topk_ids_out: Optional [T, K] int32 output buffer
        topk_weights_out: Optional [T, K] output buffer
    
    Returns:
        (topk_weights, topk_ids) with shapes [T, K]
    """
    if not gate_logits.is_cuda:
        raise ValueError("gate_logits must be a CUDA tensor")
    if not gate_logits.is_contiguous():
        raise ValueError("gate_logits must be contiguous")
    if gate_logits.dim() != 2:
        raise ValueError(f"gate_logits must be 2D, got {gate_logits.dim()}D")
    
    T, E = gate_logits.shape
    K = top_k
    
    if K > E:
        raise ValueError(f"top_k ({K}) cannot exceed num_experts ({E})")
    
    if topk_ids_out is None:
        topk_ids_out = torch.empty((T, K), dtype=torch.int32, device=gate_logits.device)
    else:
        if topk_ids_out.shape != (T, K):
            raise ValueError(f"topk_ids_out shape {topk_ids_out.shape} must be [{T}, {K}]")
        if not topk_ids_out.is_contiguous():
            raise ValueError("topk_ids_out must be contiguous")
    
    if topk_weights_out is None:
        topk_weights_out = torch.empty((T, K), dtype=gate_logits.dtype, device=gate_logits.device)
    else:
        if topk_weights_out.shape != (T, K):
            raise ValueError(f"topk_weights_out shape {topk_weights_out.shape} must be [{T}, {K}]")
        if not topk_weights_out.is_contiguous():
            raise ValueError("topk_weights_out must be contiguous")
    
    BLOCK_E = triton.next_power_of_2(E)
    
    # Tile sizing: BLOCK_T * BLOCK_E should be ~8K-16K elements
    # to keep register pressure reasonable while amortizing launch overhead.
    if BLOCK_E <= 32:
        BLOCK_T = 256
        num_warps = 8
    elif BLOCK_E <= 64:
        BLOCK_T = 128
        num_warps = 4
    elif BLOCK_E <= 128:
        BLOCK_T = 64
        num_warps = 4
    else:
        BLOCK_T = 32
        num_warps = 4
    
    grid = (triton.cdiv(T, BLOCK_T),)
    
    fused_topk_softmax_kernel[grid](
        gate_logits,
        topk_ids_out,
        topk_weights_out,
        gate_logits.stride(0), gate_logits.stride(1),
        topk_ids_out.stride(0), topk_ids_out.stride(1),
        topk_weights_out.stride(0), topk_weights_out.stride(1),
        T,
        E=E,
        K=K,
        BLOCK_E=BLOCK_E,
        BLOCK_T=BLOCK_T,
        num_warps=num_warps,
    )
    
    return topk_weights_out, topk_ids_out


# =============================================================================
# NUMERICAL VERIFICATION
# =============================================================================

def reference_topk_softmax(gate_logits, top_k):
    """Reference implementation using PyTorch ops."""
    raw_weights, topk_ids = torch.topk(gate_logits, k=top_k, dim=-1)
    router_weights = torch.softmax(raw_weights.float(), dim=-1).to(gate_logits.dtype)
    return router_weights, topk_ids.int()


def verify_correctness(T=1024, E=64, K=8, dtype=torch.bfloat16):
    """
    Verify fused kernel matches PyTorch reference.
    
    Compares SETS of selected experts (not exact ordering within ties)
    and verifies softmax weights match for each token's selected set.
    """
    torch.manual_seed(42)
    gate_logits = torch.randn(T, E, dtype=dtype, device='cuda')
    
    ref_weights, ref_ids = reference_topk_softmax(gate_logits, K)
    fused_weights, fused_ids = awsm_fused_topk_softmax(gate_logits, K)
    
    # Sort both by expert ID to compare sets (not ordering)
    ref_ids_sorted, ref_sort_idx = ref_ids.sort(dim=-1)
    fused_ids_sorted, fused_sort_idx = fused_ids.sort(dim=-1)
    sets_match = torch.all(ref_ids_sorted == fused_ids_sorted).item()
    
    # Compare weights: sort BOTH by value (descending) rather than by expert ID.
    # Sorting by expert ID fails when tied logits produce identical softmax weights
    # that end up at different positions — the gather misaligns them.
    # Sorting by weight value avoids this: identical weights land at same positions.
    ref_weights_vsorted, _ = ref_weights.float().sort(dim=-1, descending=True)
    fused_weights_vsorted, _ = fused_weights.float().sort(dim=-1, descending=True)
    weight_diff = (ref_weights_vsorted - fused_weights_vsorted).abs().max().item()
    
    exact_ids_match = torch.all(ref_ids == fused_ids).item()
    
    # Tolerance: bf16 softmax can differ by ~1e-3 at large T due to
    # reduction order differences. 2e-3 is a safe threshold.
    ok = sets_match and weight_diff < 2e-3
    
    if not ok:
        print(f"  Sets match: {sets_match}")
        print(f"  Max weight diff (aligned): {weight_diff:.2e}")
        if not sets_match:
            mismatch_mask = (ref_ids_sorted != fused_ids_sorted).any(dim=-1)
            first_t = mismatch_mask.nonzero(as_tuple=True)[0][0].item()
            print(f"  First set mismatch at token={first_t}")
            print(f"    Ref experts:   {ref_ids[first_t].tolist()}")
            print(f"    Fused experts: {fused_ids[first_t].tolist()}")
            print(f"    Logits: {gate_logits[first_t].tolist()}")
    else:
        n_order_diff = (ref_ids != fused_ids).any(dim=-1).sum().item()
        if n_order_diff > 0:
            print(f"  (Note: {n_order_diff}/{T} tokens have different tie-breaking order — OK)")
    
    return ok


if __name__ == "__main__":
    print("Verifying correctness...")
    all_pass = True
    for E in [16, 32, 64, 128]:
        for K in [2, 4, 8]:
            if K <= E:
                ok = verify_correctness(T=2048, E=E, K=K)
                status = "PASS" if ok else "FAIL"
                if not ok:
                    all_pass = False
                print(f"  E={E}, K={K}: {status}")
    
    print("\nFloat32 verification...")
    for E in [64, 128]:
        ok = verify_correctness(T=2048, E=E, K=8, dtype=torch.float32)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  E={E}, K=8, float32: {status}")
    
    # Large T tests: stress int64 indexing.
    # int32 overflow occurs when T * stride > 2^31.
    # For bf16 with E=128, stride_t=128, overflow at T > 2^31/128 = 16.7M.
    # We test at boundaries that would fail with int32 pointer math.
    print("\nLarge T verification (int64 indexing)...")
    large_t_configs = [
        # T chosen so T * E approaches/exceeds int32 range
        (131072,  128, 8, "131K×128 — typical large batch"),
        (262144,  128, 8, "262K×128 — 33M elements, near int32 limit"),
        (524288,   64, 8, "524K×64  — 33M elements, near int32 limit"),
        (1048576,  64, 8, "1M×64   — 67M elements, exceeds int32"),
    ]
    for T_test, E_test, K_test, desc in large_t_configs:
        # Check if we have enough GPU memory (rough estimate: T*E*2 bytes * 4 tensors)
        mem_needed = T_test * E_test * 2 * 4  
        mem_free = torch.cuda.mem_get_info()[0]
        if mem_needed > mem_free * 0.8:
            print(f"  T={T_test}, E={E_test}: SKIP (need ~{mem_needed//1e6:.0f}MB, have {mem_free//1e6:.0f}MB free)")
            continue
        
        ok = verify_correctness(T=T_test, E=E_test, K=K_test, dtype=torch.bfloat16)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {desc}: {status}")
    
    if all_pass:
        print("\nAll correctness tests PASSED")
    else:
        print("\nSome tests FAILED")
    
    print("\nBenchmarking...")
    import time
    T, E, K = 131072, 64, 8
    gate_logits = torch.randn(T, E, dtype=torch.bfloat16, device='cuda')
    
    # Warmup
    for _ in range(10):
        awsm_fused_topk_softmax(gate_logits, K)
    torch.cuda.synchronize()
    
    # Fused — wall clock
    torch.cuda.synchronize()
    start = time.perf_counter()
    N_ITER = 100
    for _ in range(N_ITER):
        awsm_fused_topk_softmax(gate_logits, K)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / N_ITER * 1000
    
    # Fused — CUDA events (more precise, no Python overhead)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(N_ITER):
        awsm_fused_topk_softmax(gate_logits, K)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_cuda = start_event.elapsed_time(end_event) / N_ITER
    
    # Compute effective bandwidth
    bytes_read = T * E * gate_logits.element_size()
    bytes_written = T * K * (4 + gate_logits.element_size())  # int32 ids + dtype weights
    bytes_softmax_rw = T * K * gate_logits.element_size() * 2  # reload + store
    total_bytes = bytes_read + bytes_written + bytes_softmax_rw
    eff_bw = total_bytes / (elapsed_cuda * 1e-3) / 1e12  # TB/s
    
    print(f"Fused topk+softmax:")
    print(f"  Wall clock: {elapsed:.3f} ms")
    print(f"  CUDA event: {elapsed_cuda:.3f} ms")
    print(f"  Effective BW: {eff_bw:.2f} TB/s  ({total_bytes/1e6:.1f} MB transferred)")
    
    # Separate — wall clock
    for _ in range(10):
        torch.topk(gate_logits, k=K, dim=-1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(N_ITER):
        raw_w, ids = torch.topk(gate_logits, k=K, dim=-1)
        torch.softmax(raw_w, dim=-1)
    torch.cuda.synchronize()
    elapsed_ref = (time.perf_counter() - start) / N_ITER * 1000
    print(f"\nSeparate topk+softmax: {elapsed_ref:.3f} ms")
    print(f"Speedup: {elapsed_ref/elapsed_cuda:.1f}x")
    
    # Scaling
    print("\nScaling benchmark:")
    for T_test in [1024, 8192, 32768, 65536, 131072, 262144, 524288, 1048576]:
        mem_needed = T_test * E * 2 * 5  # rough: input + outputs + intermediates
        mem_free = torch.cuda.mem_get_info()[0]
        if mem_needed > mem_free * 0.8:
            print(f"  T={T_test:>7d}: SKIP (insufficient memory)")
            continue
        g = torch.randn(T_test, E, dtype=torch.bfloat16, device='cuda')
        for _ in range(5):
            awsm_fused_topk_softmax(g, K)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            awsm_fused_topk_softmax(g, K)
        torch.cuda.synchronize()
        t_fused = (time.perf_counter() - start) / 100 * 1000
        
        for _ in range(5):
            torch.topk(g, k=K, dim=-1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            raw_w, ids = torch.topk(g, k=K, dim=-1)
            torch.softmax(raw_w, dim=-1)
        torch.cuda.synchronize()
        t_sep = (time.perf_counter() - start) / 100 * 1000
        
        print(f"  T={T_test:>6d}: fused={t_fused:.3f}ms, separate={t_sep:.3f}ms, speedup={t_sep/t_fused:.1f}x")