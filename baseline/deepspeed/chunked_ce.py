import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# =========================================================================
# Triton Softmax Kernel & Wrapper
# =========================================================================
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
    temperature,
    N_COLS,
    stride_in_row,
    stride_out_row,
    IN_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    WRITE_MAX_VAL: tl.constexpr,
    WRITE_MAX_IDX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    in_row_ptr = in_ptr + (pid * stride_in_row)
    out_row_ptr = out_ptr + (pid * stride_out_row)
    
    m_i = -float('inf')
    l_i = 0.0
    
    if WRITE_MAX_IDX:
        m_i_idx = 0
    
    cols = tl.arange(0, BLOCK_SIZE_N)
    
    # --- Pass 1: Calculate max and sum ---
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
        
    # --- Pass 2: Write normalized output ---
    for start_col in range(0, N_COLS, BLOCK_SIZE_N):
        mask = (start_col + cols) < N_COLS
        in_ptr_block = in_row_ptr + start_col + cols
        
        x = tl.load(in_ptr_block, mask=mask, other=0.0).to(tl.float32)
        x = x / temperature 
        
        p = tl.exp(x - m_i)
        out = p * l_i_inv
        
        out_ptr_block = out_row_ptr + start_col + cols
        tl.store(out_ptr_block, out.to(OUT_DTYPE), mask=mask)


def awsm_softmax(
    x: torch.Tensor, 
    out: torch.Tensor = None,
    max_idx_out: torch.Tensor = None,
    max_val_out: torch.Tensor = None,
    temperature: float = 1.0, 
):
    if x.dim() != 2:
        raise ValueError(f"Input tensor 'x' must be 2D, but got {x.dim()} dims.")
        
    M, N = x.shape
    
    if not x.is_contiguous():
        raise ValueError("Input tensor 'x' must be contiguous.")
    if x.dtype not in DTYPE_MAP:
        raise TypeError(f"Input dtype {x.dtype} not supported.")
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive.")

    if out is None:
        out = torch.empty_like(x)
    else:
        if not out.is_contiguous():
            raise ValueError("Output tensor 'out' must be contiguous.")
        if out.shape != x.shape:
            raise ValueError("Output tensor 'out' must have the same shape as 'x'.")

    WRITE_MAX_VAL = (max_val_out is not None)
    WRITE_MAX_IDX = (max_idx_out is not None)

    IN_DTYPE = DTYPE_MAP[x.dtype]
    OUT_DTYPE = DTYPE_MAP[out.dtype]

    grid = (M, )
    BLOCK_SIZE_N = 8192 
    num_warps = 32
    
    softmax_kernel[grid](
        x,
        out,
        max_idx_out, 
        max_val_out, 
        temperature, 
        N,
        x.stride(0),
        out.stride(0),
        IN_DTYPE=IN_DTYPE,
        OUT_DTYPE=OUT_DTYPE,
        WRITE_MAX_VAL=WRITE_MAX_VAL, 
        WRITE_MAX_IDX=WRITE_MAX_IDX, 
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps
    )
    
    return out, max_idx_out, max_val_out


# =========================================================================
# Chunked Cross Entropy Autograd Function
# =========================================================================
class ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, labels, chunk_size=4096, ignore_index=-100):
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
                # Native fast projection. Upcast locally for CE stability
                logits_chunk = F.linear(x_chunk, weight).float()
                
                # Using PyTorch CE guarantees stable LogSumExp.
                # Since we don't save logits, memory footprint remains small.
                loss_chunk = F.cross_entropy(
                    logits_chunk, 
                    labels_chunk, 
                    ignore_index=ignore_index, 
                    reduction='sum'
                )
                total_loss += loss_chunk.item()

        total_loss_tensor = torch.tensor(total_loss / valid_tokens, device=x.device, dtype=x.dtype)
        return total_loss_tensor

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
                # 1. Matmul in native dtype (bfloat16 / float16)
                # Call .contiguous() to ensure Triton kernel alignment
                logits_chunk = F.linear(x_chunk, weight).contiguous()
                
                # 2. Use Custom Triton Softmax Kernel 
                # This computes softmax entirely in native dtype (no fp32 intermediate tensors!)
                probs, _, _ = awsm_softmax(logits_chunk)
                
                # 3. Compute gradients in-place to save memory: (probs - true_labels)
                dZ = probs 
                
                valid_mask = labels_chunk != ignore_index
                valid_labels = labels_chunk[valid_mask]
                
                # Subtract 1.0 from true label probabilities
                dZ[valid_mask, valid_labels] -= 1.0
                
                # Zero out gradients for ignored indices
                dZ[~valid_mask] = 0.0
                
                # Scale by grad_output and valid token count
                dZ.mul_(grad_output.item() / valid_tokens)
                
                # 4. Compute dx and dw gradients
                dx[i:i+chunk_size] = torch.matmul(dZ, weight)
                dw.addmm_(dZ.t(), x_chunk)

        return dx, dw, None, None, None


# =========================================================================
# API Wrapper
# =========================================================================
def chunked_linear_cross_entropy(x, weight, labels, chunk_size=4096, ignore_index=-100):
    """
    Computes Linear + Cross Entropy in chunks to save memory during training.
    """
    return ChunkedLinearCrossEntropyFunction.apply(x, weight, labels, chunk_size, ignore_index)

# --- Quick Verification Test ---
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Switch to bfloat16 to verify Triton kernel behavior with lower precision
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32 
    
    batch_seq_len = 8192
    hidden_dim = 512
    vocab_size = 32000
    
    x = torch.randn(batch_seq_len, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_seq_len,), device=device)
    
    # Standard PyTorch method
    logits_std = F.linear(x, weight).float() # Upcast for pure PyTorch stability check
    loss_std = F.cross_entropy(logits_std, labels)
    loss_std.backward()
    
    dx_std = x.grad.clone()
    dw_std = weight.grad.clone()
    
    x.grad.zero_()
    weight.grad.zero_()
    
    # Custom Triton-Chunked method
    loss_chunked = chunked_linear_cross_entropy(x, weight, labels, chunk_size=2048)
    loss_chunked.backward()
    
    dx_chunked = x.grad.clone()
    dw_chunked = weight.grad.clone()
    
    print(f"Standard Loss: {loss_std.item():.6f}")
    print(f"Chunked Loss:  {loss_chunked.item():.6f}")
    print(f"Max diff in dx: {torch.max(torch.abs(dx_std - dx_chunked)).item()}")
    print(f"Max diff in dw: {torch.max(torch.abs(dw_std - dw_chunked)).item()}")