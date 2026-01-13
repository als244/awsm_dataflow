import torch
import triton
import triton.language as tl

DTYPE_MAP = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}




@triton.jit
def cross_entropy_loss_kernel(
    P_ptr,            # Input probabilities: (T, V) - OVERWRITTEN with dZ
    Labels_ptr,       # Input labels: (T,)
    L_ptr,            # Output row-wise loss: (T,)
    Valid_Count_ptr,  # Output valid mask: (T,)
    T,                # Number of rows
    V,                # Vocabulary size
    stride_pt,        # Stride of P along dimension T
    stride_pv,        # Stride of P along dimension V
    MAX_LOSS: tl.constexpr  # Clamp for fwd pass stability
):
    """
    Triton kernel that takes Probabilities (P) as input and does two things:
    1. Forward Pass: Computes Loss = -log(P_correct)
    2. Backward Pass: Computes dZ = P - Y (by modifying P_ptr in-place)
    """
    # 1. Get Program ID
    t = tl.program_id(axis=0)
    
    # 2. Load Label and Check Validity
    label = tl.load(Labels_ptr + t)
    valid_label_mask = (label >= 0) & (label < V)
    
    valid_float = tl.where(valid_label_mask, 1.0, 0.0)
    tl.store(Valid_Count_ptr + t, valid_float)
    
    # 3. Find Pointer to Correct Class
    p_correct_ptr = P_ptr + t * stride_pt + label * stride_pv
    
    # 4. Load P_correct
    p_correct = tl.load(
        p_correct_ptr, mask=valid_label_mask, other=1.0
    ).to(tl.float32) # other=1.0 -> log(1.0) = 0 loss
    
    # 5. FORWARD PASS: Calculate and Store Loss
    loss = -tl.log(p_correct)
    loss = tl.minimum(loss, MAX_LOSS) # Clamp loss
    loss = tl.where(valid_label_mask, loss, 0.0)
    tl.store(L_ptr + t, loss)
    
    # 6. BACKWARD PASS: Compute dZ = P - Y (in-place)
    grad_correct = p_correct - 1.0
    tl.store(p_correct_ptr, grad_correct, mask=valid_label_mask)


# --- 4. The Cross-Entropy Wrapper (Corrected) ---
def awsm_cross_entropy_loss(
    P_in: torch.Tensor,
    labels: torch.Tensor,
    L: torch.Tensor = None,
    Valid_Count_out: torch.Tensor = None,
    max_loss: float = 100.0
):
    """
    High-performance wrapper for the probability cross-entropy kernel.
    
    Computes:
    1. L = -log(P_in[label]) (clamped to max_loss)
    2. dZ_out = P_in - Y (where Y is one-hot label)
    
    NOTE: This function handles 'ignore_index' (labels < 0) by
    zeroing out the corresponding gradient rows in dZ_out.
    """
    if P_in.dim() != 2:
        raise ValueError(f"Input tensor 'P_in' must be 2D, but got {P_in.dim()} dims.")
        
    T, V = P_in.shape
    
    # Validate Input P_in
    if not P_in.is_contiguous():
        raise ValueError("Input tensor 'P_in' must be contiguous.")
    if P_in.dtype not in DTYPE_MAP:
        raise TypeError(f"Input dtype {P_in.dtype} not supported. Supported: {DTYPE_MAP.keys()}")
        
    # Validate Labels
    if labels.dim() != 1:
        raise ValueError(f"Input tensor 'labels' must be 1D, but got {labels.dim()} dims.")
    if labels.shape[0] != T:
        raise ValueError(f"labels.shape[0] ({labels.shape[0]}) must match P_in.shape[0] ({T}).")

    # Prepare Loss Output Tensor
    if L is None:
        L = torch.empty((T,), dtype=torch.float32, device=P_in.device)
    else:
        if L.shape != (T,):
             raise ValueError(f"Output tensor 'L' must have shape ({T},), got {L.shape}.")
        if L.dtype != torch.float32:
            raise ValueError(f"Output tensor 'L' must have dtype torch.float32, got {L.dtype}.")
        if not L.is_contiguous():
            raise ValueError("Output tensor 'L' must be contiguous.")

    # Prepare Valid Count Output Tensor
    if Valid_Count_out is None:
        Valid_Count_out = torch.empty((T,), dtype=torch.float32, device=P_in.device)
    else:
        if Valid_Count_out.shape != (T,):
             raise ValueError(f"Output tensor 'Valid_Count_out' must have shape ({T},), got {Valid_Count_out.shape}.")
        if Valid_Count_out.dtype != torch.float32:
            raise ValueError(f"Output tensor 'Valid_Count_out' must have dtype torch.float32, got {Valid_Count_out.dtype}.")
        if not Valid_Count_out.is_contiguous():
            raise ValueError("Output tensor 'Valid_Count_out' must be contiguous.")
            
    dZ_out = P_in

    # Grid is 1D, with one program per row
    grid = (T, )
    
    cross_entropy_loss_kernel[grid](
        dZ_out,          # This tensor starts as P, becomes dZ
        labels,
        L,
        Valid_Count_out,
        T,
        V,
        dZ_out.stride(0),
        dZ_out.stride(1),
        MAX_LOSS=max_loss,
    )
    
    return dZ_out, L


@triton.jit
def softmax_cross_entropy_kernel(
    X_ptr,            # Input logits: (T, V) - Can be any float type
    Labels_ptr,       # Input labels: (T,)
    L_ptr,            # Output row-wise loss: (T,) - Should be float32
    Valid_Count_ptr,  # Output valid mask: (T,) - Should be float32
    T,                # Number of rows
    V,                # Vocabulary size
    stride_xt,        # Stride of X along dimension T
    stride_xv,        # Stride of X along dimension V
    BLOCK_SIZE_V: tl.constexpr
):
    """
    Triton kernel for stable cross-entropy loss (Stage 1).
    Each program in a 1D grid of size (T,) processes one row.
    """
    # --- 1. Get Program ID and Pointers ---
    t = tl.program_id(axis=0)
    row_ptr = X_ptr + t * stride_xt
    
    # --- 2. Load Label and Check Validity ---
    label = tl.load(Labels_ptr + t)
    valid_label_mask = (label >= 0) & (label < V)
    
    valid_float = tl.where(valid_label_mask, 1.0, 0.0)
    tl.store(Valid_Count_ptr + t, valid_float)
    
    # --- 3. Find Logit for Correct Class ---
    logit_correct_ptr = row_ptr + label * stride_xv
    
    # Load and UPCAST to float32
    logit_correct = tl.load(
        logit_correct_ptr, mask=valid_label_mask, other=0.0
    ).to(tl.float32)
    
    # --- 4. Compute Stable LogSumExp (in float32) ---
    m = tl.full((), -float('inf'), dtype=tl.float32)
    s = tl.zeros((), dtype=tl.float32)
    v_offsets = tl.arange(0, BLOCK_SIZE_V)
    
    for v_start in range(0, V, BLOCK_SIZE_V):
        v_cols = v_start + v_offsets
        v_mask = v_cols < V
        
        # Load and UPCAST to float32
        x = tl.load(
            row_ptr + v_cols * stride_xv, mask=v_mask, other=-float('inf')
        ).to(tl.float32)
        
        # All internal math is float32
        m_new = tl.maximum(m, tl.max(x, 0))
        s_scaled = tl.sum(tl.exp(x - m_new), 0)
        s = s * tl.exp(m - m_new) + s_scaled
        m = m_new
        
    log_sum_exp = m + tl.log(s)
    
    # --- 5. Calculate Loss (in float32) ---
    loss = log_sum_exp - logit_correct
    loss = tl.where(valid_label_mask, loss, 0.0)
    
    # --- 6. Store Row-wise Loss ---
    # L_ptr is expected to be float32
    tl.store(L_ptr + t, loss)
    
    # --- 7. Perform In-place Modification ---
    # Load (and upcast)
    val_to_sub = tl.load(
        logit_correct_ptr, mask=valid_label_mask
    ).to(tl.float32)
    
    # Subtract (in float32)
    val_to_sub = val_to_sub - 1.0
    
    # Store (will DOWNCAST to X_ptr's type)
    tl.store(logit_correct_ptr, val_to_sub, mask=valid_label_mask)


def awsm_softmax_cross_entropy_loss(
    X, 
    labels, 
    L=None,
    Valid_Count=None
):
    """
    PyTorch wrapper for the Triton cross-entropy kernel.
    (This function is identical to the previous version)
    """
    T, V = X.shape
    assert X.is_contiguous(), "Input tensor X must be contiguous"
    assert labels.shape == (T,), f"Labels tensor must have shape ({T},), but got {labels.shape}"
    
    # --- Stage 1: Allocate Outputs (if not provided) ---
    if L is None:
        L = torch.empty((T,), dtype=torch.float32, device=X.device)
    else:
        assert L.shape == (T,), f"L must have shape {(T,)}, but got {L.shape}"
        assert L.device == X.device, "L must be on the same device as X"
        assert L.dtype == torch.float32, "L must be dtype torch.float32"

    if Valid_Count is None:
        Valid_Count = torch.empty((T,), dtype=torch.float32, device=X.device)
    else:
        assert Valid_Count.shape == (T,), f"Valid_Count must have shape {(T,)}, but got {Valid_Count.shape}"
        assert Valid_Count.device == X.device, "Valid_Count must be on the same device as X"
        assert Valid_Count.dtype == torch.float32, "Valid_Count must be dtype torch.float32"
    
    grid = (T,)
    
    # Heuristic for BLOCK_SIZE_V
    if V <= 1024:
        BLOCK_SIZE_V = triton.next_power_of_2(V)
    elif V <= 16384:
        BLOCK_SIZE_V = 1024
    else:
        BLOCK_SIZE_V = 2048
    
    # --- Stage 1: Launch Kernel 1 ---
    cross_entropy_loss_kernel[grid](
        X,
        labels,
        L,
        Valid_Count,
        T,
        V,
        X.stride(0),
        X.stride(1),
        BLOCK_SIZE_V=BLOCK_SIZE_V
    )
    
    # --- Stage 2: Launch Reduction Kernels ---
    # --- FIX: Use torch.sum ---
    # This runs a highly optimized GPU reduction kernel
    total_loss = torch.sum(L)
    num_valid_labels = torch.sum(Valid_Count)
    # --- End of Fix ---
    
    avg_loss = total_loss / (num_valid_labels + 1e-8)
    
    # .item() calls are still correct for getting Python scalars
    return X, L, total_loss.item(), avg_loss.item(), num_valid_labels.item()