import torch
import triton
import triton.language as tl

# Define a map for PyTorch dtypes to Triton dtypes
# (fp8 support completely removed for cleanliness)
DTYPE_MAP = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

@triton.jit
def softmax_kernel(
    in_ptr,
    out_ptr,
    max_idx_ptr,
    max_val_ptr,
    temperature, # NEW: Temperature argument
    N_COLS,
    stride_in_row,
    stride_out_row,
    IN_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    WRITE_MAX_VAL: tl.constexpr,
    WRITE_MAX_IDX: tl.constexpr,
):
    """
    Optimized Triton kernel for online softmax with temperature.
    """
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Create pointers for the row
    in_row_ptr = in_ptr + (pid * stride_in_row)
    out_row_ptr = out_ptr + (pid * stride_out_row)
    
    # Initialize running stats
    m_i = -float('inf')
    l_i = 0.0
    
    if WRITE_MAX_IDX:
        m_i_idx = 0
    
    # Column offsets
    cols = tl.arange(0, BLOCK_SIZE_N)
    
    # --- Pass 1: Calculate max and sum ---
    for start_col in range(0, N_COLS, BLOCK_SIZE_N):
        mask = (start_col + cols) < N_COLS
        x_ptr = in_row_ptr + start_col + cols
        
        # Load and apply temperature
        x = tl.load(x_ptr, mask=mask, other=-float('inf')).to(tl.float32)
        x = x / temperature # NEW: Apply temperature scaling
        
        # Online max update
        m_i_block = tl.max(x, 0)
        m_i_new = tl.maximum(m_i, m_i_block)
        
        # Stabilized exponential sum
        exp_m_diff = tl.exp(m_i - m_i_new)
        l_i = l_i * exp_m_diff
        p = tl.exp(x - m_i_new)
        l_i = l_i + tl.sum(tl.where(mask, p, 0.0), 0)
        
        # Argmax update
        # Note: argmax(x / T) == argmax(x) for T > 0, so this logic is still
        # correct for finding the index of the original max logit.
        if WRITE_MAX_IDX:
            x_for_max = tl.where(mask, x, -float('inf'))
            m_i_block_idx_local = tl.argmax(x_for_max, axis=0)
            m_i_block_idx_global = start_col + m_i_block_idx_local
            m_i_idx = tl.where(m_i_block > m_i, m_i_block_idx_global, m_i_idx)
        
        m_i = m_i_new
    
    
    # Write max index output
    if WRITE_MAX_IDX:
        tl.store(max_idx_ptr + pid, m_i_idx.to(tl.int64))
    
    # Precompute normalization factor
    l_i_inv = 1.0 / l_i
    
    # Write max value output (which is the max of the softmax)
    if WRITE_MAX_VAL:
        # The max value of the softmax output is exp(m_i - m_i) / l_i = 1.0 / l_i
        # This is correct even with temperature, as m_i is max(x/T).
        tl.store(max_val_ptr + pid, l_i_inv)
        
    
    # --- Pass 2: Write normalized output ---
    for start_col in range(0, N_COLS, BLOCK_SIZE_N):
        mask = (start_col + cols) < N_COLS
        in_ptr_block = in_row_ptr + start_col + cols
        
        # Load and apply temperature
        x = tl.load(in_ptr_block, mask=mask, other=0.0).to(tl.float32)
        x = x / temperature # NEW: Apply temperature scaling
        
        # Compute and store softmax
        p = tl.exp(x - m_i)
        out = p * l_i_inv
        
        out_ptr_block = out_row_ptr + start_col + cols
        tl.store(out_ptr_block, out.to(OUT_DTYPE), mask=mask)


def awsm_softmax(
    x: torch.Tensor, 
    out: torch.Tensor = None,
    max_idx_out: torch.Tensor = None,
    max_val_out: torch.Tensor = None,
    temperature: float = 1.0, # NEW: Temperature argument
):
    """
    High-performance online softmax for large rows.
    
    Computes:
    1. out = softmax(x / temperature)
    2. (Optional) max_idx_out = argmax(softmax(x / T), dim=1) (same as argmax(x, dim=1))
    3. (Optional) max_val_out = max(softmax(x / T), dim=1)
    
    Args:
        x (torch.Tensor): Input tensor. Must be 2D and contiguous.
        out (torch.Tensor, optional): Output tensor for softmax. If None,
                                      a new tensor is created.
        max_idx_out (torch.Tensor, optional): Output tensor for argmax. 
                                              If None, this computation is skipped.
        max_val_out (torch.Tensor, optional): Output tensor for max of softmax. 
                                              If None, this computation is skipped.
        temperature (float, optional): Softmax temperature. Defaults to 1.0.
                                              
    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): 
            A tuple containing (out, max_idx_out, max_val_out).
            The 2nd and 3rd elements will be None if they were not provided.
    """
    if x.dim() != 2:
        raise ValueError(f"Input tensor 'x' must be 2D, but got {x.dim()} dims.")
        
    M, N = x.shape
    
    # --- Validate Input ---
    if not x.is_contiguous():
        raise ValueError("Input tensor 'x' must be contiguous.")
    if x.dtype not in DTYPE_MAP:
        raise TypeError(f"Input dtype {x.dtype} not supported. Supported: {DTYPE_MAP.keys()}")
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, but got {temperature}.")

    # --- Prepare Softmax Output Tensor ---
    if out is None:
        out = torch.empty_like(x)
    else:
        if not out.is_contiguous():
            raise ValueError("Output tensor 'out' must be contiguous.")
        if out.shape != x.shape:
            raise ValueError("Output tensor 'out' must have the same shape as 'x'.")
        if out.dtype not in DTYPE_MAP:
            raise TypeError(f"Output dtype {out.dtype} not supported. Supported: {DTYPE_MAP.keys()}")
            
    # --- Prepare Max Value Output Tensor ---
    WRITE_MAX_VAL = (max_val_out is not None)
    if WRITE_MAX_VAL:
        if max_val_out.shape != (M,):
             raise ValueError(f"Output tensor 'max_val_out' must have shape ({M},), got {max_val_out.shape}.")
        if max_val_out.dtype != torch.float32:
            # The max value of the softmax is a float32 probability
            raise ValueError(f"Output tensor 'max_val_out' must have dtype torch.float32, got {max_val_out.dtype}.")
        if not max_val_out.is_contiguous():
            raise ValueError("Output tensor 'max_val_out' must be contiguous.")

    # --- Prepare Max Index Output Tensor ---
    WRITE_MAX_IDX = (max_idx_out is not None)
    if WRITE_MAX_IDX:
        if max_idx_out.shape != (M,):
             raise ValueError(f"Output tensor 'max_idx_out' must have shape ({M},), got {max_idx_out.shape}.")
        if max_idx_out.dtype != torch.int64:
            raise ValueError(f"Output tensor 'max_idx_out' must have dtype torch.int64, got {max_idx_out.dtype}.")
        if not max_idx_out.is_contiguous():
            raise ValueError("Output tensor 'max_idx_out' must be contiguous.")

    # Get Triton dtypes
    IN_DTYPE = DTYPE_MAP[x.dtype]
    OUT_DTYPE = DTYPE_MAP[out.dtype]

    # Grid is 1D, with one program per row
    grid = (M, )

    BLOCK_SIZE_N=8192 
        
    # 2. Pass num_warps as a keyword arg in the brackets
    num_warps=32
    
    softmax_kernel[grid](
        x,
        out,
        max_idx_out, # Pass tensor or None
        max_val_out, # Pass tensor or None
        temperature, # NEW: Pass temperature
        N,
        x.stride(0),
        out.stride(0),
        IN_DTYPE=IN_DTYPE,
        OUT_DTYPE=OUT_DTYPE,
        WRITE_MAX_VAL=WRITE_MAX_VAL, # Pass as constexpr
        WRITE_MAX_IDX=WRITE_MAX_IDX, # Pass as constexpr
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps
    )
    
    return out, max_idx_out, max_val_out