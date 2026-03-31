import torch
import torch.nn.functional as F

class ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, labels, chunk_size=1024, ignore_index=-100):
        """
        Args:
            x: Input hidden states, shape (N, D)
            weight: Output projection weights, shape (V, D)
            labels: Target token IDs, shape (N,)
            chunk_size: Number of tokens to process at once to save memory.
            ignore_index: Token ID to ignore in loss calculation.
        """
        # Save tensors needed for the backward pass
        ctx.save_for_backward(x, weight, labels)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index

        N = x.size(0)
        total_loss = 0.0
        
        # Count valid tokens to compute the correct mean loss across all chunks
        valid_tokens = (labels != ignore_index).sum().item()
        ctx.valid_tokens = valid_tokens

        # Edge case: if the entire batch is ignored tokens
        if valid_tokens == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Forward pass: process in chunks
        for i in range(0, N, chunk_size):
            x_chunk = x[i:i+chunk_size]
            labels_chunk = labels[i:i+chunk_size]
            
            with torch.no_grad():
                # Project to logits (upcast to float32 for numerical stability in CE)
                logits_chunk = F.linear(x_chunk, weight).float()
                
                # Compute sum of loss for this chunk
                loss_chunk = F.cross_entropy(
                    logits_chunk, 
                    labels_chunk, 
                    ignore_index=ignore_index, 
                    reduction='sum'
                )
                total_loss += loss_chunk.item()

        # Return the mean loss
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
        
        # Initialize gradient tensors
        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight)

        # Backward pass: process in chunks again
        for i in range(0, N, chunk_size):
            x_chunk = x[i:i+chunk_size]
            labels_chunk = labels[i:i+chunk_size]

            with torch.no_grad():
                # Recompute logits for this chunk in float32
                logits_chunk = F.linear(x_chunk, weight).float()
                
                # Compute Softmax probabilities
                probs = F.softmax(logits_chunk, dim=-1)
                
                # Mathematical gradient of Cross Entropy: (probs - true_labels)
                dlogits = probs.clone()
                
                valid_mask = labels_chunk != ignore_index
                valid_labels = labels_chunk[valid_mask]
                
                # Subtract 1.0 from the true label probabilities
                dlogits[valid_mask, valid_labels] -= 1.0
                
                # Zero out gradients for ignored indices
                dlogits[~valid_mask] = 0.0
                
                # Scale by grad_output and total valid tokens (since loss was mean reduced)
                dlogits.mul_(grad_output.item() / valid_tokens)
                
                # Cast back to the original dtype (e.g., float16 or bfloat16) for GEMM
                dlogits = dlogits.to(x.dtype)

                # Compute gradients for x (dx) and weight (dw) using standard matmul
                dx[i:i+chunk_size] = torch.matmul(dlogits, weight)
                dw.addmm_(dlogits.t(), x_chunk)

        # Return gradients in the exact order of the forward pass arguments
        return dx, dw, None, None, None

# =========================================================================
# The API wrapper that matches the expected function call in model_quack.py
# =========================================================================
def chunked_linear_cross_entropy(x, weight, labels, chunk_size=1024, ignore_index=-100):
    return ChunkedLinearCrossEntropyFunction.apply(x, weight, labels, chunk_size, ignore_index)

# --- Quick Verification Test ---
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 # Change to torch.float16/bfloat16 if testing on GPU
    
    # Mock data
    batch_seq_len = 10000
    hidden_dim = 512
    vocab_size = 32000
    
    x = torch.randn(batch_seq_len, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_seq_len,), device=device)
    
    # 1. Test standard PyTorch method
    logits_std = F.linear(x, weight)
    loss_std = F.cross_entropy(logits_std, labels)
    loss_std.backward()
    
    dx_std = x.grad.clone()
    dw_std = weight.grad.clone()
    
    # Zero gradients for the next test
    x.grad.zero_()
    weight.grad.zero_()
    
    # 2. Test our Custom Chunked method
    loss_chunked = chunked_linear_cross_entropy(x, weight, labels, chunk_size=2048)
    loss_chunked.backward()
    
    dx_chunked = x.grad.clone()
    dw_chunked = weight.grad.clone()
    
    print(f"Standard Loss: {loss_std.item():.6f}")
    print(f"Chunked Loss:  {loss_chunked.item():.6f}")
    print(f"Max diff in dx: {torch.max(torch.abs(dx_std - dx_chunked)).item()}")
    print(f"Max diff in dw: {torch.max(torch.abs(dw_std - dw_chunked)).item()}")
