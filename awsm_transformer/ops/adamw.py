import torch
import triton
import triton.language as tl

@triton.jit
def apply_adamw_kernel(
    # Pointers to inputs/outputs
    p_ptr,      # Parameters (Master weights)
    g_ptr,      # Gradients
    m_ptr,      # Optimizer Mean (First Moment)
    v_ptr,      # Optimizer Variance (Second Moment)
    
    # Hyperparameters
    n_elements,      # Total number of elements to process
    lr,              # Learning rate
    beta1,           # Beta 1
    beta2,           # Beta 2
    eps,             # Epsilon
    weight_decay,    # Weight decay
    
    # Pre-calculated Bias Correction terms
    bias_correction1, 
    bias_correction2,
    
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 1. Load and Cast to FP32
    p = tl.load(p_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=mask).to(tl.float32)
    m = tl.load(m_ptr + offsets, mask=mask).to(tl.float32)
    v = tl.load(v_ptr + offsets, mask=mask).to(tl.float32)

    # 2. AdamW Logic
    
    # A. Weight Decay
    # Apply decay to the parameter directly
    p = p - (lr * weight_decay * p)

    # B. Update Moments
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * (g * g)

    # C. Bias Correction (Using pre-calculated scalars)
    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    # D. Update Parameters
    denom = tl.sqrt(v_hat) + eps
    p = p - lr * (m_hat / denom)

    # 3. Store
    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(m_ptr + offsets, m, mask=mask)
    tl.store(v_ptr + offsets, v, mask=mask)

def awsm_adamw_step(params, grads, exp_avgs, exp_avg_sqs, 
               lr=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, 
               weight_decay=0.001, step=1, check_error=False):
    
    if not params.is_cuda:
        raise ValueError("Parameters must be on CUDA")

    n_elements = params.numel()
    
    # Pre-calculate bias corrections on CPU to save GPU cycles
    # step must be non-zero to avoid division by zero
    step = max(1, step) 
    bias_correction1 = 1.0 - (beta1 ** step)
    bias_correction2 = 1.0 - (beta2 ** step)

    BLOCK_SIZE = 1024 
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    ## check if any grads are inf or nan
    if check_error and (grads.isnan().any() or grads.isinf().any()):
        print("Grad is inf or nan")
        return -1

    apply_adamw_kernel[grid](
        p_ptr=params,
        g_ptr=grads,
        m_ptr=exp_avgs,
        v_ptr=exp_avg_sqs,
        n_elements=n_elements,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        # Pass computed corrections
        bias_correction1=bias_correction1,
        bias_correction2=bias_correction2,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return 0