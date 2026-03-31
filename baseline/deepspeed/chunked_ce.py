import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# =========================================================================
# Shared Triton Utilities
# =========================================================================
DTYPE_MAP = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

# =========================================================================
# 1. Softmax Kernel & Wrapper
# =========================================================================
@triton.jit
def softmax_kernel(
    in_ptr, out_ptr, max_idx_ptr, max_val_ptr, temperature, N_COLS,
    stride_in_row, stride_out_row, IN_DTYPE: tl.constexpr, OUT_DTYPE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, WRITE_MAX_VAL: tl.constexpr, WRITE_MAX_IDX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    in_row_ptr = in_ptr + (pid * stride_in_row)
    out_row_ptr = out_ptr + (pid * stride_out_row)
    
    m_i = -float('inf')
    l_i = 0.0
    if WRITE_MAX_IDX:
        m_i_idx = 0
    
    cols = tl.arange(0, BLOCK_SIZE_N)
    
    # Pass 1: Calculate max and sum
    for start_col in range(0, N_COLS, BLOCK_SIZE_N):
        mask = (start_col + cols) < N_COLS
        x_ptr = in_row_ptr + start_col + cols
        
        x = tl.load(x_ptr, mask=mask, other=-float('inf')).to(tl.float32)
        x = x / temperature 
        
        m_i_block = tl.max(x, 0)
        m_i_new = tl.maximum(m_i, m_i_block)
        
        exp_m_diff = tl.exp(m_i - m_i_new)
        l_i = l_i * exp_m_diff
        p = tl.exp(x - m_i_new)
        l_i = l_i + tl.sum(tl.where(mask, p, 0.0), 0)
        
        if WRITE_MAX_IDX:
            x_for_max = tl.where(mask, x, -float('inf'))
            m_i_block_idx_local = tl.argmax(x_for_max, axis=0)
            m_i_block_idx_global = start_col + m_i_block_idx_local
            m_i_idx = tl.where(m_i_block > m_i, m_i_block_idx_global, m_i_idx)
        
        m_i = m_i_new
    
    if WRITE_MAX_IDX:
        tl.store(max_idx_ptr + pid, m_i_idx.to(tl.int64))
    
    l_i_inv = 1.0 / l_i
    if WRITE_MAX_VAL:
        tl.store(max_val_ptr + pid, l_i_inv)
        
    # Pass 2: Write normalized output
    for start_col in range(0, N_COLS, BLOCK_SIZE_N):
        mask = (start_col + cols) < N_COLS
        in_ptr_block = in_row_ptr + start_col + cols
        
        x = tl.load(in_ptr_block, mask=mask, other=0.0).to(tl.float32)
        x = x / temperature 
        
        p = tl.exp(x - m_i)
        out = p * l_i_inv
        
        out_ptr_block = out_row_ptr + start_col + cols
        tl.store(out_ptr_block, out.to(OUT_DTYPE), mask=mask)

def awsm_softmax(x: torch.Tensor, out: torch.Tensor = None, max_idx_out: torch.Tensor = None, max_val_out: torch.Tensor = None, temperature: float = 1.0):
    if x.dim() != 2: raise ValueError(f"Input tensor 'x' must be 2D, but got {x.dim()} dims.")
    M, N = x.shape
    if not x.is_contiguous(): raise ValueError("Input tensor 'x' must be contiguous.")
    
    if out is None: out = torch.empty_like(x)
    
    WRITE_MAX_VAL = (max_val_out is not None)
    WRITE_MAX_IDX = (max_idx_out is not None)

    grid = (M, )
    softmax_kernel[grid](
        x, out, max_idx_out, max_val_out, temperature, N,
        x.stride(0), out.stride(0),
        IN_DTYPE=DTYPE_MAP[x.dtype], OUT_DTYPE=DTYPE_MAP[out.dtype],
        WRITE_MAX_VAL=WRITE_MAX_VAL, WRITE_MAX_IDX=WRITE_MAX_IDX, 
        BLOCK_SIZE_N=8192, num_warps=32
    )
    return out, max_idx_out, max_val_out

# =========================================================================
# 2. Cross Entropy Kernel & Wrapper (Probabilities -> Loss & dZ)
# =========================================================================
@triton.jit
def cross_entropy_loss_kernel(
    P_ptr, Labels_ptr, L_ptr, Valid_Count_ptr, T, V, 
    stride_pt, stride_pv, MAX_LOSS: tl.constexpr
):
    t = tl.program_id(axis=0)
    label = tl.load(Labels_ptr + t)
    valid_label_mask = (label >= 0) & (label < V)
    
    valid_float = tl.where(valid_label_mask, 1.0, 0.0)
    tl.store(Valid_Count_ptr + t, valid_float)
    
    p_correct_ptr = P_ptr + t * stride_pt + label * stride_pv
    p_correct = tl.load(p_correct_ptr, mask=valid_label_mask, other=1.0).to(tl.float32) 
    
    # FORWARD: Calculate and Store Loss
    loss = -tl.log(p_correct)
    loss = tl.minimum(loss, MAX_LOSS)
    loss = tl.where(valid_label_mask, loss, 0.0)
    tl.store(L_ptr + t, loss)
    
    # BACKWARD: Compute dZ = P - Y (in-place modification of P_ptr)
    grad_correct = p_correct - 1.0
    tl.store(p_correct_ptr, grad_correct, mask=valid_label_mask)

def awsm_cross_entropy_loss(P_in: torch.Tensor, labels: torch.Tensor, L: torch.Tensor = None, Valid_Count_out: torch.Tensor = None, max_loss: float = 100.0):
    T, V = P_in.shape
    if not P_in.is_contiguous(): raise ValueError("Input tensor 'P_in' must be contiguous.")
        
    if L is None: L = torch.empty((T,), dtype=torch.float32, device=P_in.device)
    if Valid_Count_out is None: Valid_Count_out = torch.empty((T,), dtype=torch.float32, device=P_in.device)
            
    dZ_out = P_in
    grid = (T, )
    
    cross_entropy_loss_kernel[grid](
        dZ_out, labels, L, Valid_Count_out, T, V,
        dZ_out.stride(0), dZ_out.stride(1), MAX_LOSS=max_loss,
    )
    return dZ_out, L

# =========================================================================
# 3. Chunked Cross Entropy Autograd Function
# =========================================================================
class ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, labels, chunk_size=1024, ignore_index=-100):
        ctx.save_for_backward(x, weight, labels)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index

        N = x.size(0)
        total_loss = 0.0
        
        valid_tokens = (labels != ignore_index).sum().item()
        ctx.valid_tokens = valid_tokens

        if valid_tokens == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for i in range(0, N, chunk_size):
            x_chunk = x[i:i+chunk_size]
            labels_chunk = labels[i:i+chunk_size]
            
            with torch.no_grad():
                # F.linear requires contiguous memory for Triton to play nicely
                logits_chunk = F.linear(x_chunk, weight).contiguous()
                
                # Use custom kernels (probs array is overwritten by dZ, but we drop it in forward)
                probs_chunk, _, _ = awsm_softmax(logits_chunk)
                _, L = awsm_cross_entropy_loss(probs_chunk, labels_chunk)
                
                # L naturally ignores valid_labels because the kernel stores 0.0 for them
                total_loss += L.sum().item()

        return torch.tensor(total_loss / valid_tokens, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, labels = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        ignore_index = ctx.ignore_index
        valid_tokens = ctx.valid_tokens

        if valid_tokens == 0:
            return torch.zeros_like(x), torch.zeros_like(weight), None, None, None

        N = x.size(0)
        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight)

        for i in range(0, N, chunk_size):
            x_chunk = x[i:i+chunk_size]
            labels_chunk = labels[i:i+chunk_size]

            with torch.no_grad():
                logits_chunk = F.linear(x_chunk, weight).contiguous()
                
                # 1. Calculate Softmax Probabilities
                probs_chunk, _, _ = awsm_softmax(logits_chunk)
                
                # 2. Compute in-place dZ using probabilities (modifies probs_chunk to P - Y)
                dZ, _ = awsm_cross_entropy_loss(probs_chunk, labels_chunk)
                
                # 3. Handle ignore_index 
                # The Triton CE kernel leaves invalid rows fully intact (i.e., non-zero probability distributions)
                # We MUST manually zero out the ignored rows so they don't impact gradients
                valid_mask = labels_chunk != ignore_index
                dZ[~valid_mask] = 0.0
                
                # Scale gradients based on reduction method
                dZ.mul_(grad_output.item() / valid_tokens)
                
                # 4. Compute final chunked weight/input gradients
                dx[i:i+chunk_size] = torch.matmul(dZ, weight)
                dw.addmm_(dZ.t(), x_chunk)

        return dx, dw, None, None, None

# =========================================================================
# API Wrapper
# =========================================================================
def chunked_linear_cross_entropy(x, weight, labels, chunk_size=1024, ignore_index=-100):
    return ChunkedLinearCrossEntropyFunction.apply(x, weight, labels, chunk_size, ignore_index)

# --- Quick Verification Test ---
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32 
    
    batch_seq_len = 8192
    hidden_dim = 512
    vocab_size = 32000
    
    x = torch.randn(batch_seq_len, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_seq_len,), device=device)
    
    # Inject some ignore indices to test correctness
    labels[:100] = -100 

    # Standard PyTorch method
    logits_std = F.linear(x, weight).float() 
    loss_std = F.cross_entropy(logits_std, labels, ignore_index=-100)
    loss_std.backward()
    
    dx_std = x.grad.clone()
    dw_std = weight.grad.clone()
    
    x.grad.zero_()
    weight.grad.zero_()
    
    # Custom Triton-Chunked method
    loss_chunked = chunked_linear_cross_entropy(x, weight, labels, chunk_size=1024, ignore_index=-100)
    loss_chunked.backward()
    
    dx_chunked = x.grad.clone()
    dw_chunked = weight.grad.clone()
    
    print(f"Standard Loss: {loss_std.item():.6f}")
    print(f"Chunked Loss:  {loss_chunked.item():.6f}")
    print(f"Max diff in dx: {torch.max(torch.abs(dx_std - dx_chunked)).item()}")
    print(f"Max diff in dw: {torch.max(torch.abs(dw_std - dw_chunked)).item()}")