import torch
import triton
import triton.language as tl

# -----------------
# Forward Kernel (1D Flat)
# -----------------

@triton.jit
def swiglu_fwd_kernel(
    X1_PTR,    # pointer to x1
    X3_PTR,    # pointer to x3
    OUT_PTR,   # pointer to output
    N_ELEMENTS, # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Vectorized offset calculation
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Pointers
    x1_ptrs = X1_PTR + offsets
    x3_ptrs = X3_PTR + offsets
    out_ptrs = OUT_PTR + offsets

    # Load
    x1 = tl.load(x1_ptrs, mask=mask)
    x3 = tl.load(x3_ptrs, mask=mask)

    # Compute
    # x1 * sigmoid(x1) * x3
    x1_f32 = x1.to(tl.float32)
    s = tl.sigmoid(x1_f32)
    out = (x1 * s) * x3

    # Store
    tl.store(out_ptrs, out.to(OUT_PTR.dtype.element_ty), mask=mask)


# -----------------
# Backward Kernel (1D Flat)
# -----------------

@triton.jit
def swiglu_bwd_kernel(
    X1_PTR,    # pointer to x1
    X3_PTR,    # pointer to x3
    DOUT_PTR,  # pointer to upstream grad
    DX1_PTR,   # pointer to dx1
    DX3_PTR,   # pointer to dx3
    ACT_PTR,   # pointer to activation storage (optional)
    N_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
    STORE_ACTIVATIONS: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Pointers
    x1_ptrs = X1_PTR + offsets
    x3_ptrs = X3_PTR + offsets
    dout_ptrs = DOUT_PTR + offsets
    dx1_ptrs = DX1_PTR + offsets
    dx3_ptrs = DX3_PTR + offsets

    # Load
    x1 = tl.load(x1_ptrs, mask=mask)
    x3 = tl.load(x3_ptrs, mask=mask)
    dout = tl.load(dout_ptrs, mask=mask)

    # Compute
    x1_f32 = x1.to(tl.float32)
    s = tl.sigmoid(x1_f32)
    a = x1 * s # silu(x1)

    # Optional: Recompute and store activation
    if STORE_ACTIVATIONS:
        act = a * x3
        act_ptrs = ACT_PTR + offsets
        tl.store(act_ptrs, act.to(ACT_PTR.dtype.element_ty), mask=mask)

    # Gradients
    # dx3 = dout * silu(x1)
    dx3 = dout * a

    # dx1 = dout * x3 * d/dx1(silu(x1))
    # d/dx1(silu) = s + x1 * s * (1 - s) = s * (1 + x1 * (1 - s))
    dsilu_dx1 = s * (1.0 + x1_f32 * (1.0 - s))
    dx1 = dout * x3 * dsilu_dx1

    # Store
    tl.store(dx1_ptrs, dx1.to(DX1_PTR.dtype.element_ty), mask=mask)
    tl.store(dx3_ptrs, dx3.to(DX3_PTR.dtype.element_ty), mask=mask)


# -----------------
# Wrappers
# -----------------

def awsm_swiglu_fwd(x1: torch.Tensor, x3: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    # Validation
    assert x1.shape == x3.shape
    assert x1.is_cuda and x3.is_cuda
    
    # Optimization: Ensure contiguous memory for 1D flattening.
    # If inputs are sliced/transposed, this forces a copy, which is 
    # necessary for the 1D kernel optimization anyway.
    if not x1.is_contiguous(): x1 = x1.contiguous()
    if not x3.is_contiguous(): x3 = x3.contiguous()

    if out is None:
        out = torch.empty_like(x1)
    
    n_elements = x1.numel()
    
    # Config
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    swiglu_fwd_kernel[grid](
        x1,
        x3,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8 
    )

    return out


def awsm_swiglu_bwd(
    x1: torch.Tensor, 
    x3: torch.Tensor, 
    dout: torch.Tensor, 
    dx1: torch.Tensor = None, 
    dx3: torch.Tensor = None,
    store_activations: bool = False,
    activations: torch.Tensor = None
) -> tuple:
    
    assert x1.shape == x3.shape == dout.shape
    assert x1.is_cuda and x3.is_cuda and dout.is_cuda

    # Ensure contiguous memory for efficient 1D access
    if not x1.is_contiguous(): x1 = x1.contiguous()
    if not x3.is_contiguous(): x3 = x3.contiguous()
    if not dout.is_contiguous(): dout = dout.contiguous()

    if dx1 is None: dx1 = torch.empty_like(x1)
    if dx3 is None: dx3 = torch.empty_like(x3)

    act_ptr = None
    if store_activations:
        if activations is None:
            activations = torch.empty_like(x1)
        act_ptr = activations

    n_elements = x1.numel()
    
    # Config
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    swiglu_bwd_kernel[grid](
        x1,
        x3,
        dout,
        dx1,
        dx3,
        act_ptr, # Passing None is safe if STORE_ACTIVATIONS is False
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        STORE_ACTIVATIONS=store_activations,
        num_warps=8
    )

    if store_activations:
        return dx1, dx3, activations
    return dx1, dx3