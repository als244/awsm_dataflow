"""
MoE (Mixture of Experts) Operations - Triton Kernels and Wrappers

This module provides high-performance Triton kernels for Mixture of Experts layers,
including sorting, scattering, gathering, and SwiGLU activation operations.

Key Concepts:
- "Original order": Tokens indexed by [token_id, expert_k] where k is the k-th expert choice
- "Sorted order": Tokens grouped by expert, indexed by a flat position in [0, T*K)
- `index_mapping[t, k]` maps original position (t, k) -> sorted position

Shape Conventions:
- T: Number of tokens
- K: Top-K experts per token  
- D: Model dimension
- F: Expert hidden dimension (expert_dim)
- E: Number of experts
"""

import torch
import triton
import triton.language as tl
import math

# ============================================================================
# TRITON KERNELS
# ============================================================================

@triton.jit
def swiglu_fwd_weighted_kernel(
    X_PTR,      # Packed Input (T x 2F)
    W_PTR,      # Router Weights (T)
    OUT_PTR,    # Output (T x F)
    stride_x_t, 
    stride_out_t, 
    F: tl.constexpr,         
    BLOCK_SIZE: tl.constexpr
):
    """
    SwiGLU forward with router weight scaling.
    
    IMPORTANT: Input X is packed as [x3, x1] where:
      - x3 (value) is in the FIRST half:  X[:, :F]
      - x1 (gate) is in the SECOND half:  X[:, F:]
    
    Computes: out = w * (SiLU(x1) * x3)
    """
    pid_t = tl.program_id(0).to(tl.int64) 
    pid_f = tl.program_id(1).to(tl.int64)
    
    offs_f = pid_f * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_f = offs_f < F

    w = tl.load(W_PTR + pid_t).to(tl.float32)

    row_start_ptr = X_PTR + (pid_t * stride_x_t)
    
    # FIXED: x3 (value) is first half, x1 (gate) is second half
    x3_ptrs = row_start_ptr + offs_f          # value: first half [0:F]
    x1_ptrs = row_start_ptr + F + offs_f      # gate: second half [F:2F]
    
    x3 = tl.load(x3_ptrs, mask=mask_f)  # value
    x1 = tl.load(x1_ptrs, mask=mask_f)  # gate

    x1_f32 = x1.to(tl.float32)
    x3_f32 = x3.to(tl.float32)
    s = tl.sigmoid(x1_f32)

    swiglu_out = (x1_f32 * s) * x3_f32
    
    out = swiglu_out * w

    out_ptrs = OUT_PTR + (pid_t * stride_out_t) + offs_f
    tl.store(out_ptrs, out.to(OUT_PTR.dtype.element_ty), mask=mask_f)


@triton.jit
def swiglu_bwd_weighted_kernel(
    DX_PTR, DW_PTR, DOUT_PTR, X_PTR, W_PTR,
    stride_dx_t, stride_dout_t, stride_x_t,
    FWD_ACT_PTR, stride_fwd_act_t,
    STORE_FWD_ACT: tl.constexpr,
    F: tl.int32, 
    BLOCK_SIZE: tl.constexpr
):
    """
    SwiGLU backward with router weight gradient.
    
    IMPORTANT: Input X is packed as [x3, x1] where:
      - x3 (value) is in the FIRST half:  X[:, :F]
      - x1 (gate) is in the SECOND half:  X[:, F:]
    
    Output DX is packed the same way: [dx3, dx1]
    """
    pid_t = tl.program_id(0).to(tl.int64) 
    
    w_val = tl.load(W_PTR + pid_t)
    
    row_dout_ptr = DOUT_PTR + (pid_t * stride_dout_t)
    row_x_ptr    = X_PTR + (pid_t * stride_x_t)
    row_dx_ptr   = DX_PTR + (pid_t * stride_dx_t)
    
    dw_acc = 0.0

    for off in range(0, F, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < F

        dout_chunk = tl.load(row_dout_ptr + offs, mask=mask, other=0.0)
        
        # FIXED: x3 (value) is first half, x1 (gate) is second half
        x3_chunk = tl.load(row_x_ptr + offs, mask=mask, other=0.0)      # value: first half
        x1_chunk = tl.load(row_x_ptr + F + offs, mask=mask, other=0.0)  # gate: second half

        x1_f32 = x1_chunk.to(tl.float32)
        x3_f32 = x3_chunk.to(tl.float32)
        
        s = tl.sigmoid(x1_f32) 
        silu = x1_f32 * s
        
        fwd_out_unweighted = silu * x3_f32

        fwd_out_scaled = fwd_out_unweighted * w_val.to(tl.float32)
        
        if STORE_FWD_ACT:
            row_fwd_ptr = FWD_ACT_PTR + (pid_t * stride_fwd_act_t)
            tl.store(row_fwd_ptr + offs, fwd_out_scaled.to(FWD_ACT_PTR.dtype.element_ty), mask=mask)

        chunk_dot = tl.sum(dout_chunk.to(tl.float32) * fwd_out_unweighted)
        dw_acc += chunk_dot

        dout_scaled = dout_chunk.to(tl.float32) * w_val.to(tl.float32)

        # Gradient for x3 (value): d/dx3 = dout * silu
        dx3 = dout_scaled * silu
        
        # Gradient for x1 (gate): d/dx1 = dout * x3 * d(silu)/d(x1)
        dsilu_dx1 = s + (silu * (1.0 - s))
        dx1 = dout_scaled * x3_f32 * dsilu_dx1

        # FIXED: Store dx3 in first half, dx1 in second half
        tl.store(row_dx_ptr + offs, dx3.to(DX_PTR.dtype.element_ty), mask=mask)      # dx3: first half
        tl.store(row_dx_ptr + F + offs, dx1.to(DX_PTR.dtype.element_ty), mask=mask)  # dx1: second half

    tl.store(DW_PTR + pid_t, dw_acc.to(DW_PTR.dtype.element_ty))


@triton.jit
def swiglu_bwd_prescaled_kernel(
    DX_PTR, DW_PTR, DOUT_SCALED_PTR, X_PTR, W_PTR,
    stride_dx_t, stride_dout_t, stride_x_t,
    FWD_ACT_PTR, stride_fwd_act_t,
    STORE_FWD_ACT: tl.constexpr,
    F: tl.int32, 
    BLOCK_SIZE: tl.constexpr
):
    """
    SwiGLU backward when upstream gradient is already scaled by router probability.
    
    IMPORTANT: Input X is packed as [x3, x1] where:
      - x3 (value) is in the FIRST half:  X[:, :F]
      - x1 (gate) is in the SECOND half:  X[:, F:]
    
    Output DX is packed the same way: [dx3, dx1]
    """
    pid_t = tl.program_id(0).to(tl.int64) 
    
    w_val = tl.load(W_PTR + pid_t).to(tl.float32)
    w_val_inv = 1.0 / tl.maximum(1e-8, w_val)
    
    row_dout_ptr = DOUT_SCALED_PTR + (pid_t * stride_dout_t)
    row_x_ptr    = X_PTR + (pid_t * stride_x_t)
    row_dx_ptr   = DX_PTR + (pid_t * stride_dx_t)
    
    dw_acc = 0.0

    for off in range(0, F, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < F

        dout_scaled_chunk = tl.load(row_dout_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        
        # FIXED: x3 (value) is first half, x1 (gate) is second half
        x3_chunk = tl.load(row_x_ptr + offs, mask=mask, other=0.0)      # value: first half
        x1_chunk = tl.load(row_x_ptr + F + offs, mask=mask, other=0.0)  # gate: second half

        x1_f32 = x1_chunk.to(tl.float32)
        x3_f32 = x3_chunk.to(tl.float32)
        
        s = tl.sigmoid(x1_f32) 
        silu = x1_f32 * s
        fwd_out_unweighted = silu * x3_f32
        
        if STORE_FWD_ACT:
            row_fwd_ptr = FWD_ACT_PTR + (pid_t * stride_fwd_act_t)
            tl.store(row_fwd_ptr + offs, fwd_out_unweighted.to(FWD_ACT_PTR.dtype.element_ty), mask=mask)

        dout_unscaled_chunk = dout_scaled_chunk * w_val_inv
        chunk_dot = tl.sum(dout_unscaled_chunk * fwd_out_unweighted)
        dw_acc += chunk_dot

        # Gradient for x3 (value): d/dx3 = dout_scaled * silu
        dx3 = dout_scaled_chunk * silu
        
        # Gradient for x1 (gate): d/dx1 = dout_scaled * x3 * d(silu)/d(x1)
        dsilu_dx1 = s + (silu * (1.0 - s))
        dx1 = dout_scaled_chunk * x3_f32 * dsilu_dx1

        # FIXED: Store dx3 in first half, dx1 in second half
        tl.store(row_dx_ptr + offs, dx3.to(DX_PTR.dtype.element_ty), mask=mask)      # dx3: first half
        tl.store(row_dx_ptr + F + offs, dx1.to(DX_PTR.dtype.element_ty), mask=mask)  # dx1: second half

    dw_final = dw_acc

    tl.store(DW_PTR + pid_t, dw_final.to(DW_PTR.dtype.element_ty))


@triton.jit
def moe_router_gate_bwd_kernel(
    DLOGITS_PTR, DPROBS_PTR, PROBS_PTR, INDICES_PTR, EXPERTS_PTR,
    stride_dlogits_t, stride_probs_t, stride_indices_t, stride_experts_t,
    T: tl.constexpr, K: tl.constexpr, E: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    
    k_offs = tl.arange(0, BLOCK_SIZE)
    mask_k = k_offs < K
    
    indices_ptr = INDICES_PTR + (pid * stride_indices_t) + k_offs
    dprob_indices = tl.load(indices_ptr, mask=mask_k, other=0).to(tl.int64)
    
    dp_val = tl.load(DPROBS_PTR + dprob_indices, mask=mask_k, other=0.0).to(tl.float32)
    
    probs_ptr = PROBS_PTR + (pid * stride_probs_t) + k_offs
    p_val = tl.load(probs_ptr, mask=mask_k, other=0.0).to(tl.float32)
    
    dot_prod = tl.sum(dp_val * p_val)
    dz_val = p_val * (dp_val - dot_prod)
    
    experts_ptr = EXPERTS_PTR + (pid * stride_experts_t) + k_offs
    expert_ids = tl.load(experts_ptr, mask=mask_k, other=0).to(tl.int64)
    
    dlogits_row_start = DLOGITS_PTR + (pid * stride_dlogits_t)
    out_ptrs = dlogits_row_start + expert_ids
    
    mask_safe = mask_k & (expert_ids >= 0) & (expert_ids < E)
    
    current_val = tl.load(out_ptrs, mask=mask_safe, other=0.0)
    tl.store(out_ptrs, current_val + dz_val.to(DLOGITS_PTR.dtype.element_ty), mask=mask_safe)


@triton.jit
def moe_count_kernel(
    TOPK_IDS_PTR, count_out_ptr, N_TOKENS, NUM_EXPERTS, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    offs_token = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_token = offs_token < N_TOKENS
    
    expert_ids = tl.load(TOPK_IDS_PTR + offs_token, mask=mask_token, other=0).to(tl.int64)
    
    row_start_ptr = count_out_ptr + (pid * NUM_EXPERTS)
    tl.atomic_add(row_start_ptr + expert_ids, 1, mask=mask_token)


@triton.jit
def moe_map_kernel(
    TOPK_IDS_PTR, DEST_IDX_PTR, OFFSETS_PTR, N_TOKENS, NUM_EXPERTS, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    offs_token = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_token = offs_token < N_TOKENS
    
    expert_ids = tl.load(TOPK_IDS_PTR + offs_token, mask=mask_token, other=0).to(tl.int64)
    
    broad_experts = expert_ids[:, None]
    is_same = broad_experts == broad_experts.T
    indices = tl.arange(0, BLOCK_SIZE)
    is_after = indices[:, None] > indices[None, :]
    valid_neighbors = is_same & is_after
    local_ranks = tl.sum(valid_neighbors.to(tl.int32), axis=1)
    
    offs_row_ptr = OFFSETS_PTR + (pid * NUM_EXPERTS)
    base_offsets = tl.load(offs_row_ptr + expert_ids, mask=mask_token, other=0)
    
    tl.store(DEST_IDX_PTR + offs_token, base_offsets + local_ranks, mask=mask_token)


@triton.jit
def moe_copy_counts_kernel(
    GPU_SRC_PTR, CPU_DEST_PTR, NUM_EXPERTS, BLOCK_SIZE: tl.constexpr
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < NUM_EXPERTS
    val = tl.load(GPU_SRC_PTR + offs, mask=mask)
    tl.store(CPU_DEST_PTR + offs, val, mask=mask)


@triton.jit
def moe_scatter_kernel(
    SRC_PTR, DEST_PTR, INDICES_PTR, SCALES_PTR,
    stride_src_m, stride_src_d, 
    stride_dest_m, stride_dest_d, 
    stride_idx_m, stride_idx_k,
    stride_scale_m, stride_scale_k,
    T, D, K: tl.constexpr, 
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    HAS_SCALE: tl.constexpr
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < T
    mask_d = offs_d < D
    
    src_ptrs = SRC_PTR + (offs_m[:, None] * stride_src_m + offs_d[None, :] * stride_src_d)
    src_tile = tl.load(src_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    
    for k in range(K):
        idx_ptrs = INDICES_PTR + (offs_m * stride_idx_m + k * stride_idx_k)
        dest_rows = tl.load(idx_ptrs, mask=mask_m, other=0).to(tl.int64)
        
        val_to_store = src_tile

        if HAS_SCALE:
            scale_ptrs = SCALES_PTR + (offs_m * stride_scale_m + k * stride_scale_k)
            scale_vals = tl.load(scale_ptrs, mask=mask_m, other=1.0).to(tl.float32)
            
            val_to_store = val_to_store.to(tl.float32) * scale_vals[:, None]

        dst_ptrs = DEST_PTR + (dest_rows[:, None] * stride_dest_m + offs_d[None, :] * stride_dest_d)
        tl.store(dst_ptrs, val_to_store.to(DEST_PTR.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def moe_scatter_routing_weights_kernel(
    W_PTR, INDICES_PTR, OUT_PTR, N_ELEMENTS, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEMENTS
    
    w_val = tl.load(W_PTR + offs, mask=mask)
    dest_idx = tl.load(INDICES_PTR + offs, mask=mask).to(tl.int64)
    
    tl.store(OUT_PTR + dest_idx, w_val, mask=mask)


@triton.jit
def moe_gather_kernel(
    EXPERT_OUT_PTR, INDICES_PTR, WEIGHTS_PTR, RESIDUAL_PTR, OUT_PTR,
    stride_exp_m, stride_exp_d, stride_idx_m, stride_idx_k, stride_w_m, stride_w_k,
    stride_res_m, stride_res_d, stride_out_m, stride_out_d,
    T, D, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, 
    USE_WEIGHTS: tl.constexpr, 
    HAS_RESIDUAL: tl.constexpr
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < T
    mask_d = offs_d < D
    
    if HAS_RESIDUAL:
        res_ptrs = RESIDUAL_PTR + (offs_m[:, None] * stride_res_m + offs_d[None, :] * stride_res_d)
        accumulator = tl.load(res_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    for k in range(K):
        idx_ptrs = INDICES_PTR + (offs_m * stride_idx_m + k * stride_idx_k)
        target_rows = tl.load(idx_ptrs, mask=mask_m, other=0).to(tl.int64)
        
        exp_ptrs = EXPERT_OUT_PTR + (target_rows[:, None] * stride_exp_m + offs_d[None, :] * stride_exp_d)
        val = tl.load(exp_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        
        if USE_WEIGHTS:
            w_ptrs = WEIGHTS_PTR + (offs_m * stride_w_m + k * stride_w_k)
            w = tl.load(w_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            accumulator += val * w[:, None]
        else:
            accumulator += val

    out_ptrs = OUT_PTR + (offs_m[:, None] * stride_out_m + offs_d[None, :] * stride_out_d)
    tl.store(out_ptrs, accumulator.to(OUT_PTR.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def load_balance_bwd_kernel(
    LOGITS_PTR, FRAC_PTR, DLOGITS_PTR,
    stride_logits_t, stride_logits_e,
    stride_dlogits_t, stride_dlogits_e,
    T, E: tl.constexpr, SCALE: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    Computes gradient of load balancing loss w.r.t. router logits.
    
    Loss: L = alpha * E * sum_e(f_e * p_bar_e)
    where f_e = fraction of assignments to expert e
          p_bar_e = mean probability for expert e across tokens
    
    Gradient: dL/dz_{t,e} = (alpha * E / T) * p_{t,e} * (f_e - sum_{e'} f_{e'} * p_{t,e'})
    """
    pid = tl.program_id(0).to(tl.int64)
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < E
    
    # Load logits for this token
    logits_ptrs = LOGITS_PTR + pid * stride_logits_t + offs * stride_logits_e
    logits = tl.load(logits_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
    
    # Stable softmax
    m = tl.max(logits, 0)
    exp_logits = tl.exp(logits - m)
    sum_exp = tl.sum(exp_logits, 0)
    p = exp_logits / sum_exp
    
    # Load pre-computed fractions f_e
    f_val = tl.load(FRAC_PTR + offs, mask=mask, other=0.0).to(tl.float32)
    
    # Compute gradient: scale * p * (f - <f, p>)
    dot = tl.sum(f_val * p, 0)
    grad = SCALE * p * (f_val - dot)
    
    # Store gradient
    dlogits_ptrs = DLOGITS_PTR + pid * stride_dlogits_t + offs * stride_dlogits_e
    existing = tl.load(dlogits_ptrs, mask=mask, other=0.0)
    tl.store(dlogits_ptrs, (existing + grad).to(DLOGITS_PTR.dtype.element_ty), mask=mask)


# ============================================================================
# WRAPPER FUNCTIONS
# ============================================================================

def awsm_moe_sort(
    topk_ids: torch.Tensor, 
    num_experts: int,
    indices: torch.Tensor = None,
    expert_counts_gpu: torch.Tensor = None,
    block_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sort token-expert assignments by expert ID for efficient batched expert computation.
    
    Creates a mapping from original [token, k] positions to sorted positions where
    all tokens for expert 0 come first, then expert 1, etc.
    
    Args:
        topk_ids: Expert IDs for each token's top-k choices.
            Shape: [T, K] where T=num_tokens, K=top_k
            Dtype: int32 or int64
            Must be: CUDA tensor
            
        num_experts: Total number of experts (E).
        
        indices: Optional pre-allocated output buffer for index mapping.
            Shape: [T, K]
            Dtype: same as topk_ids
            Must be: CUDA tensor
            If None, will be allocated.
            
        block_size: Triton block size for parallel processing. Default 256.
    
    Returns:
        Tuple of:
            index_mapping: Maps original position to sorted position.
                Shape: [T, K]
                Semantics: index_mapping[t, k] = sorted position for token t's k-th expert
                
            expert_counts: Number of tokens assigned to each expert.
                Shape: [E]
                Dtype: int32
    
    Example:
        # Token 0 -> experts [2, 1], Token 1 -> experts [0, 2]
        topk_ids = torch.tensor([[2, 1], [0, 2]], device='cuda')
        index_mapping, counts = awsm_moe_sort(topk_ids, num_experts=3)
        # counts = [1, 1, 2]  (expert 0 has 1, expert 1 has 1, expert 2 has 2)
        # index_mapping maps each (token, k) to its position in the sorted array
    
    Raises:
        AssertionError: If topk_ids is not on CUDA.
        AssertionError: If provided indices has wrong shape.
    """
    if not topk_ids.is_cuda:
        raise ValueError("topk_ids must be a CUDA tensor")
    
    flat_ids = topk_ids.flatten()
    N_TOKENS = flat_ids.numel()
    
    num_blocks = triton.cdiv(N_TOKENS, block_size)
    block_counts = torch.zeros((num_blocks, num_experts), dtype=torch.int32, device=topk_ids.device)
    
    moe_count_kernel[(num_blocks,)](
        flat_ids, 
        block_counts, 
        N_TOKENS, 
        num_experts, 
        BLOCK_SIZE=block_size
    )

    total_counts = block_counts.sum(dim=0)
    expert_starts = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=topk_ids.device), 
        total_counts.cumsum(0)[:-1]
    ])
    
    block_accum = torch.zeros_like(block_counts)
    block_accum[1:] = block_counts.cumsum(dim=0)[:-1]
    offsets = block_accum + expert_starts.unsqueeze(0)

    if indices is None:
        indices_flat = torch.empty_like(flat_ids)
    else:
        if indices.shape != topk_ids.shape:
            raise ValueError(f"indices shape {indices.shape} must match topk_ids shape {topk_ids.shape}")
        if not indices.is_cuda:
            raise ValueError("indices must be a CUDA tensor")
        indices_flat = indices.flatten()
    
    moe_map_kernel[(num_blocks,)](
        flat_ids, 
        indices_flat, 
        offsets, 
        N_TOKENS, 
        num_experts, 
        BLOCK_SIZE=block_size
    )
    
    sorted_indices = indices_flat.view(topk_ids.shape)

    if expert_counts_gpu is not None:
        expert_counts_gpu.copy_(total_counts)
    return sorted_indices, total_counts


def awsm_copy_expert_counts(
    gpu_counts: torch.Tensor, 
    cpu_buffer: torch.Tensor
) -> None:
    """
    Asynchronously copy expert counts from GPU to pinned CPU memory.
    
    This enables overlapping the CPU-side expert loop setup with GPU computation.
    Caller must synchronize before reading cpu_buffer.
    
    Args:
        gpu_counts: Expert token counts on GPU.
            Shape: [E] where E=num_experts
            Dtype: int32
            Must be: CUDA tensor, contiguous
            
        cpu_buffer: Pre-allocated pinned CPU buffer for receiving counts.
            Shape: [E]
            Dtype: int32
            Must be: CPU tensor, pinned memory (torch.zeros(..., pin_memory=True))
    
    Raises:
        ValueError: If gpu_counts is not on CUDA.
        ValueError: If cpu_buffer is not pinned memory.
    
    Example:
        expert_counts_gpu = ...  # from awsm_moe_sort
        expert_counts_cpu = torch.zeros(num_experts, dtype=torch.int32, pin_memory=True)
        awsm_copy_expert_counts(expert_counts_gpu, expert_counts_cpu)
        torch.cuda.current_stream().synchronize()  # Wait for copy
        # Now safe to read expert_counts_cpu
    """
    if not gpu_counts.is_cuda:
        raise ValueError("gpu_counts must be a CUDA tensor")
    if not cpu_buffer.is_pinned():
        raise ValueError("cpu_buffer must be pinned memory (use pin_memory=True)")
    if gpu_counts.shape != cpu_buffer.shape:
        raise ValueError(f"Shape mismatch: gpu_counts {gpu_counts.shape} vs cpu_buffer {cpu_buffer.shape}")
    
    n_experts = gpu_counts.numel()
    BLOCK_SIZE = 1024 
    
    moe_copy_counts_kernel[(1,)](
        gpu_counts,
        cpu_buffer,
        n_experts,
        BLOCK_SIZE=BLOCK_SIZE
    )


def awsm_moe_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    scales: torch.Tensor = None,
    out: torch.Tensor = None
) -> torch.Tensor:
    """
    Scatter input tokens to sorted expert positions, optionally with scaling.
    
    For each token t and expert choice k, copies x[t] to out[indices[t, k]],
    optionally multiplied by scales[t, k].
    
    Args:
        x: Input token features.
            Shape: [T, D] where T=num_tokens, D=model_dim
            Must be: CUDA tensor, contiguous
            
        indices: Mapping from original to sorted positions (from awsm_moe_sort).
            Shape: [T, K] where K=top_k
            Must be: CUDA tensor, contiguous
            Semantics: indices[t, k] = destination position in output
            
        scales: Optional per-token-expert scaling factors.
            Shape: [T, K] - MUST be in original [token, k] order, NOT sorted order
            Must be: CUDA tensor, contiguous
            If None, no scaling is applied.
            WARNING: Do NOT pass sorted-order weights here.
            
        out: Optional pre-allocated output buffer.
            Shape: [T * K, D]
            Must be: CUDA tensor
            If None, will be allocated.
    
    Returns:
        Scattered token features in sorted expert order.
            Shape: [T * K, D]
            Semantics: out[indices[t, k]] = x[t] * scales[t, k] (if scales provided)
    
    Raises:
        ValueError: If x is not contiguous.
        ValueError: If indices is not contiguous.
        ValueError: If scales is provided but not contiguous.
        ValueError: If scales has wrong number of elements.
    
    Example:
        x = torch.randn(1024, 512, device='cuda')  # 1024 tokens, dim 512
        indices, _ = awsm_moe_sort(topk_ids, num_experts=8)  # [1024, 2]
        scattered = awsm_moe_scatter(x, indices)  # [2048, 512]
    """
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not indices.is_cuda:
        raise ValueError("indices must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not indices.is_contiguous():
        raise ValueError("indices must be contiguous")
    
    T, D = x.shape
    T_idx, K = indices.shape
    
    if T_idx != T:
        raise ValueError(f"indices first dim {T_idx} must match x first dim {T}")
    
    has_scale = scales is not None
    stride_scale_m, stride_scale_k = 0, 0
    
    if has_scale:
        if not scales.is_cuda:
            raise ValueError("scales must be a CUDA tensor")
        if not scales.is_contiguous():
            raise ValueError("scales must be contiguous")
        if scales.numel() != T * K:
            raise ValueError(
                f"scales must have T*K={T*K} elements, got {scales.numel()}. "
                f"Expected shape [T={T}, K={K}] in ORIGINAL order (not sorted)."
            )
        scales = scales.view(T, K)
        stride_scale_m, stride_scale_k = scales.stride(0), scales.stride(1)
    else:
        scales = x  # Dummy pointer
    
    if out is None:
        out = torch.empty((T * K, D), dtype=x.dtype, device=x.device)
    else:
        if out.shape != (T * K, D):
            raise ValueError(f"out shape {out.shape} must be [{T * K}, {D}]")
    
    BLOCK_M = 32
    BLOCK_D = 128
    
    grid = (triton.cdiv(T, BLOCK_M), triton.cdiv(D, BLOCK_D))
    
    moe_scatter_kernel[grid](
        x, out, indices, scales,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        indices.stride(0), indices.stride(1),
        stride_scale_m, stride_scale_k,
        T, D, K, 
        BLOCK_M=BLOCK_M, 
        BLOCK_D=BLOCK_D,
        HAS_SCALE=has_scale
    )
    return out


def awsm_moe_scatter_routing_weights(
    weights: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor = None
) -> torch.Tensor:
    """
    Scatter router probabilities from original order to sorted expert order.
    
    Reorders weights so they align with the sorted token positions for use
    in the expert computation loop.
    
    Args:
        weights: Router probabilities in original [token, k] order.
            Shape: [T, K]
            Must be: CUDA tensor, contiguous
            Semantics: weights[t, k] = probability for token t's k-th expert choice
            
        indices: Mapping from original to sorted positions (from awsm_moe_sort).
            Shape: [T, K]
            Must be: CUDA tensor, contiguous
            
        out: Optional pre-allocated output buffer.
            Shape: [T * K]
            Must be: CUDA tensor
            If None, will be allocated.
    
    Returns:
        Router probabilities in sorted expert order.
            Shape: [T * K]
            Semantics: out[indices[t, k]] = weights[t, k]
            Use: Index with [start:end] slices in expert loop
    
    Raises:
        ValueError: If weights or indices not on CUDA.
        ValueError: If shapes don't match.
    
    Example:
        router_probs = torch.softmax(logits, dim=-1)  # [T, K]
        indices, _ = awsm_moe_sort(topk_ids, num_experts=8)
        probs_sorted = awsm_moe_scatter_routing_weights(router_probs, indices)
        # In expert loop: exp_probs = probs_sorted[start:end]
    """
    if not weights.is_cuda:
        raise ValueError("weights must be a CUDA tensor")
    if not indices.is_cuda:
        raise ValueError("indices must be a CUDA tensor")
    if weights.shape != indices.shape:
        raise ValueError(f"weights shape {weights.shape} must match indices shape {indices.shape}")
    
    w_flat = weights.flatten()
    idx_flat = indices.flatten()
    
    if not w_flat.is_contiguous():
        raise ValueError("weights must be contiguous")
    if not idx_flat.is_contiguous():
        raise ValueError("indices must be contiguous")
    
    N = w_flat.numel()
    if out is None:
        out = torch.empty_like(w_flat)
    else:
        if out.numel() != N:
            raise ValueError(f"out must have {N} elements, got {out.numel()}")
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    moe_scatter_routing_weights_kernel[grid](
        w_flat, 
        idx_flat, 
        out, 
        N, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def awsm_moe_gather(
    expert_out: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor = None,
    residual: torch.Tensor = None,
    out: torch.Tensor = None
) -> torch.Tensor:
    """
    Gather expert outputs back to original token order with optional weighting.
    
    For each token t, sums contributions from its K expert outputs:
        out[t] = sum_k(weights[t, k] * expert_out[indices[t, k]]) + residual[t]
    
    Args:
        expert_out: Expert computation results in sorted order.
            Shape: [T * K, D]
            Must be: CUDA tensor, contiguous
            Semantics: Contains outputs for all token-expert pairs, grouped by expert
            
        indices: Mapping from original to sorted positions (from awsm_moe_sort).
            Shape: [T, K]
            Must be: CUDA tensor, contiguous
            Semantics: indices[t, k] = position in expert_out for token t's k-th expert
            
        weights: Router probabilities in ORIGINAL [token, k] order.
            Shape: [T, K]
            Must be: CUDA tensor, contiguous
            WARNING: Must be original order, NOT sorted order (router_weights_sorted).
                     The kernel indexes as weights[t, k] to match the loop over k.
            If None, outputs are summed without weighting.
            
        residual: Optional residual to add to output.
            Shape: [T, D]
            Must be: CUDA tensor, contiguous
            If None, no residual is added.
            
        out: Optional pre-allocated output buffer.
            Shape: [T, D]
            Must be: CUDA tensor, contiguous
            If None, will be allocated.
    
    Returns:
        Gathered and weighted token outputs.
            Shape: [T, D]
    
    Raises:
        ValueError: If any input is not contiguous.
        ValueError: If shapes are inconsistent.
    
    Example:
        # After expert computation:
        gathered = awsm_moe_gather(
            expert_out=scattered_x,      # [T*K, D] in sorted order  
            indices=index_mapping,       # [T, K] from awsm_moe_sort
            weights=router_weights,      # [T, K] ORIGINAL order, NOT router_weights_sorted!
        )
    """
    if not expert_out.is_cuda:
        raise ValueError("expert_out must be a CUDA tensor")
    if not indices.is_cuda:
        raise ValueError("indices must be a CUDA tensor")
    if not expert_out.is_contiguous():
        raise ValueError("expert_out must be contiguous")
    if not indices.is_contiguous():
        raise ValueError("indices must be contiguous")

    T, K = indices.shape
    _, D = expert_out.shape
    
    # Validate expert_out shape
    expected_expert_out_rows = T * K
    if expert_out.shape[0] != expected_expert_out_rows:
        raise ValueError(
            f"expert_out has {expert_out.shape[0]} rows but expected T*K={expected_expert_out_rows} "
            f"for T={T}, K={K}"
        )

    # Handle Residual
    has_residual = residual is not None
    if has_residual:
        if not residual.is_cuda:
            raise ValueError("residual must be a CUDA tensor")
        if not residual.is_contiguous():
            raise ValueError("residual must be contiguous")
        if residual.shape != (T, D):
            raise ValueError(f"residual shape {residual.shape} must be [{T}, {D}]")
        res_ptr = residual
        stride_res_m, stride_res_d = residual.stride(0), residual.stride(1)
    else:
        res_ptr = expert_out
        stride_res_m, stride_res_d = 0, 0

    # Handle Weights
    use_weights = weights is not None
    if use_weights:
        if not weights.is_cuda:
            raise ValueError("weights must be a CUDA tensor")
        if not weights.is_contiguous():
            raise ValueError("weights must be contiguous")
        if weights.shape != (T, K):
            raise ValueError(
                f"weights shape {weights.shape} must be [{T}, {K}] in ORIGINAL order. "
                f"Do NOT pass router_weights_sorted here - use the original router_weights."
            )
        w_ptr = weights
        stride_w_m, stride_w_k = weights.stride(0), weights.stride(1)
    else:
        w_ptr = expert_out 
        stride_w_m, stride_w_k = 0, 0
    
    # Handle Output
    if out is None:
        out_dtype = residual.dtype if has_residual else expert_out.dtype
        out = torch.empty((T, D), dtype=out_dtype, device=expert_out.device)
    else:
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.shape != (T, D):
            raise ValueError(f"out shape {out.shape} must be [{T}, {D}]")
        
    BLOCK_M = 32
    BLOCK_D = 128
    grid = (triton.cdiv(T, BLOCK_M), triton.cdiv(D, BLOCK_D))
    
    moe_gather_kernel[grid](
        expert_out, 
        indices, 
        w_ptr, 
        res_ptr, 
        out,
        expert_out.stride(0), expert_out.stride(1),
        indices.stride(0), indices.stride(1),
        stride_w_m, stride_w_k,
        stride_res_m, stride_res_d,
        out.stride(0), out.stride(1),
        T, D, K=K, 
        BLOCK_M=BLOCK_M, 
        BLOCK_D=BLOCK_D, 
        USE_WEIGHTS=use_weights,
        HAS_RESIDUAL=has_residual
    )
    return out


def awsm_swiglu_moe_fwd(
    x: torch.Tensor, 
    w: torch.Tensor = None, 
    out: torch.Tensor = None
) -> torch.Tensor:
    """
    Forward pass of SwiGLU activation with optional per-token weighting.
    
    Computes: out = w * SwiGLU(x) = w * (SiLU(x1) * x3)
    
    IMPORTANT: x is packed as [x3, x1] (value first, gate second):
      - x[:, :F] = x3 (value)
      - x[:, F:] = x1 (gate)
    
    Args:
        x: Packed input containing value and gate components.
            Shape: [N, 2*F] where N=num_tokens, F=hidden_dim
            Layout: x[:, :F] = value (x3), x[:, F:] = gate (x1)
            Must be: CUDA tensor
            
        w: Optional per-token scaling weights.
            Shape: [N] or [N, 1]
            Must be: CUDA tensor
            If None, defaults to 1.0 (no scaling).
            
        out: Optional pre-allocated output buffer.
            Shape: [N, F]
            Must be: CUDA tensor
            If None, will be allocated.
    
    Returns:
        SwiGLU activation output, optionally scaled.
            Shape: [N, F]
    """
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    
    if w is not None:
        if not w.is_cuda:
            raise ValueError("w must be a CUDA tensor")
    else:
        w = torch.ones((x.shape[0],), device=x.device, dtype=x.dtype)
    
    T, F2 = x.shape
    if F2 % 2 != 0:
        raise ValueError(f"x last dimension must be even (got {F2}), as it contains packed [value, gate]")
    F = F2 // 2
    
    if out is None:
        out = torch.empty((T, F), device=x.device, dtype=x.dtype)
    else:
        if out.shape != (T, F):
            raise ValueError(f"out shape {out.shape} must be [{T}, {F}]")
    
    BLOCK_SIZE = 1024
    grid = (T, triton.cdiv(F, BLOCK_SIZE))
    
    swiglu_fwd_weighted_kernel[grid](
        x, w, out,
        x.stride(0), out.stride(0),
        F=F,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out



def awsm_swiglu_moe_bwd(
    dout: torch.Tensor, 
    x: torch.Tensor, 
    w: torch.Tensor, 
    dx: torch.Tensor = None,
    dw: torch.Tensor = None,
    fwd_act: torch.Tensor = None 
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass of SwiGLU with router weight gradient computation.
    
    IMPORTANT: x is packed as [x3, x1] (value first, gate second):
      - x[:, :F] = x3 (value)
      - x[:, F:] = x1 (gate)
    
    Output dx is packed the same way: [dx3, dx1]
    
    Args:
        dout: Upstream gradient (UNSCALED by router probability).
            Shape: [N, F]
            
        x: Original packed input from forward pass.
            Shape: [N, 2*F]
            Layout: x[:, :F] = value (x3), x[:, F:] = gate (x1)
            
        w: Per-token router probabilities.
            Shape: [N] or [N, 1]
            
        dx: Optional pre-allocated gradient buffer for x.
            Shape: [N, 2*F]
            
        dw: Optional pre-allocated gradient buffer for w.
            Shape: [N] or [N, 1]
            
        fwd_act: Optional buffer to store w * SwiGLU(x).
            Shape: [N, F]
    
    Returns:
        Tuple of (dx, dw)
    """
    if not dout.is_cuda:
        raise ValueError("dout must be a CUDA tensor")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not w.is_cuda:
        raise ValueError("w must be a CUDA tensor")
    
    T, F = dout.shape
    
    if x.shape[0] != T:
        raise ValueError(f"x first dim {x.shape[0]} must match dout first dim {T}")
    if x.shape[1] != 2 * F:
        raise ValueError(f"x second dim {x.shape[1]} must be 2*F={2*F}")
    if w.numel() != T:
        raise ValueError(f"w must have {T} elements, got {w.numel()}")
    
    if dx is None:
        dx = torch.empty_like(x)
    else:
        if dx.shape != x.shape:
            raise ValueError(f"dx shape {dx.shape} must match x shape {x.shape}")
            
    if dw is None:
        dw = torch.empty((T,), device=x.device, dtype=torch.float32)
    else:
        if dw.numel() != T:
            raise ValueError(f"dw must have {T} elements, got {dw.numel()}")

    store_fwd_act = fwd_act is not None
    if store_fwd_act:
        if fwd_act.shape != (T, F):
            raise ValueError(f"fwd_act shape {fwd_act.shape} must be [{T}, {F}]")
        fwd_act_stride = fwd_act.stride(0)
    else:
        fwd_act = x  # Dummy pointer
        fwd_act_stride = 0

    BLOCK_SIZE = min(1024, triton.next_power_of_2(F))
    
    grid = (T,)
    
    swiglu_bwd_weighted_kernel[grid](
        dx, dw, dout, x, w,
        dx.stride(0), dout.stride(0), x.stride(0),
        fwd_act, fwd_act_stride,
        STORE_FWD_ACT=store_fwd_act,
        F=F,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return dx, dw


def awsm_swiglu_moe_bwd_prescaled(
    dout_scaled: torch.Tensor,
    x: torch.Tensor, 
    w: torch.Tensor, 
    dx: torch.Tensor = None,
    dw: torch.Tensor = None,
    fwd_act: torch.Tensor = None 
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass of SwiGLU when upstream gradient is ALREADY scaled by router probability.
    
    IMPORTANT: x is packed as [x3, x1] (value first, gate second):
      - x[:, :F] = x3 (value)
      - x[:, F:] = x1 (gate)
    
    Output dx is packed the same way: [dx3, dx1]
    
    Args:
        dout_scaled: Upstream gradient, ALREADY SCALED by router probability.
            Shape: [N, F]
            
        x: Original packed input from forward pass.
            Shape: [N, 2*F]
            
        w: Per-token router probabilities.
            Shape: [N] or [N, 1]
            
        dx: Optional pre-allocated gradient buffer for x.
            Shape: [N, 2*F]
            
        dw: Optional pre-allocated gradient buffer for w.
            Shape: [N] or [N, 1]
            
        fwd_act: Optional buffer to store UNWEIGHTED SwiGLU(x).
            Shape: [N, F]
    
    Returns:
        Tuple of (dx, dw)
    """
    if not dout_scaled.is_cuda:
        raise ValueError("dout_scaled must be a CUDA tensor")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not w.is_cuda:
        raise ValueError("w must be a CUDA tensor")
    
    T, F = dout_scaled.shape
    
    if x.shape[0] != T:
        raise ValueError(f"x first dim {x.shape[0]} must match dout_scaled first dim {T}")
    if x.shape[1] != 2 * F:
        raise ValueError(f"x second dim {x.shape[1]} must be 2*F={2*F}")
    if w.numel() != T:
        raise ValueError(f"w must have {T} elements, got {w.numel()}")
    
    if dx is None:
        dx = torch.empty_like(x)
    else:
        if dx.shape != x.shape:
            raise ValueError(f"dx shape {dx.shape} must match x shape {x.shape}")
            
    if dw is None:
        dw = torch.empty((T,), device=x.device, dtype=torch.float32)
    else:
        if dw.numel() != T:
            raise ValueError(f"dw must have {T} elements, got {dw.numel()}")

    store_fwd_act = fwd_act is not None
    if store_fwd_act:
        if fwd_act.shape != (T, F):
            raise ValueError(f"fwd_act shape {fwd_act.shape} must be [{T}, {F}]")
        fwd_act_stride = fwd_act.stride(0)
    else:
        fwd_act = x 
        fwd_act_stride = 0

    BLOCK_SIZE = min(1024, triton.next_power_of_2(F))
    
    grid = (T,)
    
    swiglu_bwd_prescaled_kernel[grid](
        dx, dw, dout_scaled, x, w,
        dx.stride(0), dout_scaled.stride(0), x.stride(0),
        fwd_act, fwd_act_stride,
        STORE_FWD_ACT=store_fwd_act,
        F=F,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return dx, dw


def awsm_moe_router_gate_bwd(
    probs: torch.Tensor,
    dprobs: torch.Tensor,
    indices: torch.Tensor,
    chosen_experts: torch.Tensor,
    dlogits: torch.Tensor = None,
    num_experts: int = None
) -> torch.Tensor:
    """
    Compute router logit gradients from probability gradients through softmax.
    
    Performs two operations:
    1. Unpermutes dprobs from sorted order back to original [T, K] order
    2. Computes softmax backward: dlogits = probs * (dprobs - sum(dprobs * probs))
    
    Args:
        probs: Router probabilities (softmax output) for chosen experts.
            Shape: [T, K]
            Must be: CUDA tensor, contiguous
            Semantics: probs[t, k] = softmax probability for token t's k-th expert choice
            
        dprobs: Gradient w.r.t. router probabilities in SORTED order.
            Shape: [T * K] (flat)
            Must be: CUDA tensor, contiguous
            Semantics: Gradients computed during expert loop, in expert-sorted order
            
        indices: Mapping from original to sorted positions (from awsm_moe_sort).
            Shape: [T, K]
            Must be: CUDA tensor, contiguous
            Used to: Unpermute dprobs back to original order
            
        chosen_experts: Expert IDs chosen by each token.
            Shape: [T, K]
            Must be: CUDA tensor, contiguous
            Semantics: chosen_experts[t, k] = expert ID for token t's k-th choice
            
        dlogits: Optional pre-allocated output buffer.
            Shape: [T, E] where E = num_experts
            Must be: CUDA tensor, contiguous
            If None, will be allocated (requires num_experts).
            
        num_experts: Number of experts (E). Required if dlogits is None.
    
    Returns:
        Gradient w.r.t. router logits.
            Shape: [T, E]
            Semantics: dlogits[t, e] = gradient for token t's logit to expert e
    
    Raises:
        ValueError: If inputs not contiguous.
        ValueError: If shapes don't match.
        ValueError: If num_experts not provided when dlogits is None.
    
    Example:
        # After computing dprobs in expert loop:
        dlogits = awsm_moe_router_gate_bwd(
            probs=router_probs,           # [T, K] original softmax probs
            dprobs=dprobs_sorted,         # [T*K] from expert loop
            indices=index_mapping,        # [T, K] from awsm_moe_sort  
            chosen_experts=selected_experts,
            num_experts=8
        )
    """
    if not probs.is_contiguous():
        raise ValueError("probs must be contiguous")
    if not dprobs.is_contiguous():
        raise ValueError("dprobs must be contiguous")
    if not indices.is_contiguous():
        raise ValueError("indices must be contiguous")
    if not chosen_experts.is_contiguous():
        raise ValueError("chosen_experts must be contiguous")

    T, K = probs.shape
    
    if dprobs.numel() != T * K:
        raise ValueError(f"dprobs must have T*K={T*K} elements, got {dprobs.numel()}")
    if indices.shape != (T, K):
        raise ValueError(f"indices shape {indices.shape} must be [{T}, {K}]")
    if chosen_experts.shape != (T, K):
        raise ValueError(f"chosen_experts shape {chosen_experts.shape} must be [{T}, {K}]")
    
    if dlogits is None:
        if num_experts is None:
            raise ValueError("num_experts must be provided if dlogits is not passed")
        dlogits = torch.zeros((T, num_experts), device=probs.device, dtype=probs.dtype)
    else:
        if dlogits.shape[0] != T:
            raise ValueError(f"dlogits first dim {dlogits.shape[0]} must be T={T}")
        if not dlogits.is_contiguous():
            raise ValueError("dlogits must be contiguous")
    
    E = dlogits.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(K)
    
    grid = (T,)
    
    moe_router_gate_bwd_kernel[grid](
        dlogits, dprobs, probs, indices, chosen_experts,
        dlogits.stride(0), probs.stride(0), indices.stride(0), chosen_experts.stride(0),
        T=T, K=K, E=E,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dlogits


def awsm_load_balance_bwd(
    logits: torch.Tensor,
    expert_counts: torch.Tensor,
    num_experts: int,
    alpha: float = 0.01,
    top_k: int = 1,
    dlogits: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute gradient for auxiliary load balancing loss.
    
    The load balancing loss encourages uniform expert utilization:
        L_balance = alpha * E * sum_e(f_e * p_bar_e)
    where:
        f_e = (tokens assigned to expert e) / (T * top_k) = fraction of assignments
        p_bar_e = (1/T) * sum_t p_{t,e} = mean router probability for expert e
    
    This computes dL_balance/d_logits.
    
    Args:
        logits: Router logits before softmax.
            Shape: [T, E] where T=num_tokens, E=num_experts
            
        expert_counts: Number of tokens assigned to each expert.
            Shape: [E]
            Note: With top-k routing, sum(expert_counts) = T * top_k
            
        num_experts: Number of experts (E).
        
        alpha: Load balancing loss coefficient. Default 0.01.
            Typical values: 0.01 - 0.1
            
        top_k: Number of experts each token is routed to. Default 1.
            Used to correctly normalize expert_counts to fractions.
            
        dlogits: Optional pre-allocated output buffer. If supplied, we accumulate load balance gradient into it.
            Shape: [T, E]
    
    Returns:
        Gradient of load balancing loss w.r.t. logits.
            Shape: [T, E]
    """
    if not logits.is_cuda:
        raise ValueError("logits must be a CUDA tensor")
    if not expert_counts.is_cuda:
        raise ValueError("expert_counts must be a CUDA tensor")
    
    T, E = logits.shape
    if E != num_experts:
        raise ValueError(f"logits has {E} experts but num_experts={num_experts}")
    if expert_counts.numel() != E:
        raise ValueError(f"expert_counts must have {E} elements, got {expert_counts.numel()}")
    
    if dlogits is None:
        dlogits = torch.zeros_like(logits)
    elif dlogits.shape != (T, E):
        raise ValueError(f"dlogits shape {dlogits.shape} must be [{T}, {E}]")
    
    # Compute fractions: f_e = count_e / (T * top_k)
    # Do this in float32 for precision
    fractions = expert_counts.float() / (T * top_k)
    
    # Scale factor: alpha * E / T
    scale = alpha * E / T
    
    BLOCK_SIZE = triton.next_power_of_2(E)
    grid = (T,)
    
    load_balance_bwd_kernel[grid](
        logits, fractions, dlogits,
        logits.stride(0), logits.stride(1),
        dlogits.stride(0), dlogits.stride(1),
        T=T, E=E,
        SCALE=scale,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dlogits