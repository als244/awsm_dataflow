import torch
import triton
import triton.language as tl

@triton.jit
def embedding_bwd_kernel(
    # Pointers
    grad_out_ptr,         
    grad_weight_ptr,      
    sorted_indices_ptr,   
    sorted_tokens_ptr,    
    # Strides
    stride_grad_out_row, stride_grad_out_col,
    stride_gw_row, stride_gw_col,
    # Scale factor
    scale,
    T,
    # Meta-parameters   
    D: tl.constexpr,  
    BLOCK_SIZE_D: tl.constexpr
):
    # --- 1. 64-bit Safe Indexing ---
    pid_seg = tl.program_id(0).to(tl.int64) 
    pid_d = tl.program_id(1).to(tl.int64)

    # --- 2. Early Exit (Skip Duplicates) ---
    curr_token = tl.load(sorted_tokens_ptr + pid_seg)
    
    if pid_seg > 0:
        prev_token = tl.load(sorted_tokens_ptr + pid_seg - 1)
        if prev_token == curr_token:
            return

    # --- 3. Setup Offsets ---
    offs_d = (pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)).to(tl.int64)
    mask_d = offs_d < D
    
    # --- 4. Accumulate ---
    acc = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
    curr_idx_in_sorted = pid_seg
    
    while curr_idx_in_sorted < T:
        should_process = True
        
        if curr_idx_in_sorted > pid_seg:
            check_token = tl.load(sorted_tokens_ptr + curr_idx_in_sorted)
            if check_token != curr_token:
                should_process = False
        
        if should_process:
            orig_row_idx = tl.load(sorted_indices_ptr + curr_idx_in_sorted)
            
            src_offset = (orig_row_idx * stride_grad_out_row) + (offs_d * stride_grad_out_col)
            
            val = tl.load(grad_out_ptr + src_offset, mask=mask_d, other=0.0)
            acc += val
            
            curr_idx_in_sorted += 1
        else:
            curr_idx_in_sorted = tl.full([], T, dtype=tl.int64)

    # --- 5. Apply Scale Factor ---
    acc = acc * scale

    # --- 6. Write Output ---
    token_idx = curr_token.to(tl.int64)
    
    dst_offset = (token_idx * stride_gw_row) + (offs_d * stride_gw_col)
    dst_ptr = grad_weight_ptr + dst_offset
    
    existing_val = tl.load(dst_ptr, mask=mask_d, other=0.0)
    final_val = existing_val + acc
    tl.store(dst_ptr, final_val, mask=mask_d)


def awsm_embedding_bwd(grad_output, indices, grad_weight, scale=1.0):
    """
    Backward pass for scaled embedding lookup.
    
    Args:
        grad_output: Gradient w.r.t. output [T, D]
        indices: Token indices [T]
        grad_weight: Gradient w.r.t. embedding weights [V, D] (accumulated into)
        scale: Optional scale factor applied during forward (default: 1.0)
    
    Returns:
        grad_weight with gradients accumulated
    """
    assert grad_output.dim() == 2
    assert indices.dim() == 1
    assert grad_weight.dim() == 2
    
    T, D = grad_output.shape
    
    # Sort indices to group identical tokens together
    sorted_idx_map = torch.argsort(indices)
    sorted_tokens = indices[sorted_idx_map]
    
    BLOCK_SIZE_D = 1024
    if D < BLOCK_SIZE_D:
        BLOCK_SIZE_D = triton.next_power_of_2(D)

    grid = lambda meta: (
        T, 
        triton.cdiv(D, meta['BLOCK_SIZE_D'])
    )
    
    embedding_bwd_kernel[grid](
        grad_output, grad_weight,
        sorted_idx_map,
        sorted_tokens, 
        grad_output.stride(0), grad_output.stride(1),
        grad_weight.stride(0), grad_weight.stride(1),
        scale,  # Pass scale factor to kernel
        T, 
        D=D,
        BLOCK_SIZE_D=BLOCK_SIZE_D
    )
    
    return grad_weight