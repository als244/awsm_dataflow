import torch

def awsm_muon_step(param, grad, momentum, lr=1e-4, beta=0.95, eps=1e-8, 
                   a=3.4445, b=-4.775, c=2.0315, ns_iters=5, weight_decay=0, 
                   nesterov=True, check_error=False):
    """
    AWSM Muon step preserving input datatype.
    """

    ## check if any grads are inf or nan
    if check_error and (grad.isnan().any() or grad.isinf().any()):
        print("Grad is inf or nan")
        return -1
    
    grad_bf16 = grad.bfloat16()
    # 1. Update Momentum (Standard SGD variant used in Muon)
    # Reference: buf.mul_(momentum).add_(g)
    # This means: v_{t+1} = mu * v_{t} + g_{t}
    momentum.mul_(beta).add_(grad_bf16)

    # 2. Apply Nesterov
    # Reference: g = g.add(buf, alpha=momentum)
    # This implies we orthogonalize (g + mu * v_{t+1})
    if nesterov:
        g = grad_bf16.add(momentum, alpha=beta)
    else:
        g = momentum.clone()

    # 3. Handle Convolutional Layers
    if g.ndim == 4:
        g = g.view(len(g), -1)
    
    # 4. Setup Newton-Schulz Buffer
    X = g 

    # Transpose if the matrix is "tall" (rows > cols) to optimize NS loop
    # After this, X is shaped (Small, Large)
    row_count, col_count = X.shape[-2], X.shape[-1]
    transposed = row_count > col_count
    
    if transposed:
        X = X.mT
        # Swap dims for scaling calculation later
        row_count, col_count = col_count, row_count
    
    # Spectral Norm Scaling
    norm = X.norm(dim=(-2, -1), keepdim=True)
    X.div_(norm.add_(eps))

    # Quintic Newton-Schulz Loop
    # Optimization: If X is (N, M) where N < M:
    # Calculating (bA + cA^2) @ X is cheaper than b(A@X) + c(A@(A@X))
    for _ in range(ns_iters):
        # A = X @ X.T  (Shape: N x N)
        A = torch.mm(X, X.mT)
        
        # B = b * A + c * A @ A   (Shape: N x N)
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        
        # X = a * X + B @ X
        X = torch.addmm(X, B, X, beta=a, alpha=1.0)

    # 5. Untranspose if needed
    if transposed:
        X = X.mT

    # 6. Apply Weight Decay and Update Param
    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)

    # 7. Apply Scale Correction
    # Moonshot scale factor
    scale_factor = .2 * max(row_count, col_count) ** 0.5


    param.add_(X.view_as(param).to(dtype=param.dtype), alpha=-lr * scale_factor)

    return 0