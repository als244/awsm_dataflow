import torch
import triton
import triton.language as tl
# Import for benchmarking
from triton.testing import do_bench

@triton.jit
def top_p_sampler_kernel(
    # Pointers to Tensors
    sorted_probs_ptr,       # (T, K) matrix of sorted probabilities
    sorted_indices_ptr,     # (T, K) matrix of original indices
    top_p_ptr,       # (T,) vector of top-p values
    output_indices_ptr,     # (T,) vector for output indices
    output_probs_ptr,       # (T,) vector for (optional) output probabilities
    # Dimensions
    T,                      # Batch dimension
    K,                      # Top-K dimension
    # RNG seeds
    seed,
    offset,
    # Strides
    stride_probs_t, stride_probs_k,
    stride_indices_t, stride_indices_k,
    stride_top_p,
    stride_out_idx,
    stride_out_prob,
    # Constexpr
    BLOCK_S_K: tl.constexpr, # Block size for K, must be power of 2 >= K
    STORE_PROBS: tl.constexpr, # Compile-time flag to store probs
):
    """
    Triton kernel for Top-P sampling from pre-sorted probability distributions.
    """
    # --- 1. Kernel Setup ---
    
    # This program handles one row of the batch
    pid = tl.program_id(axis=0) # pid is the row index t, from 0 to T-1
    
    # Boundary check: don't run for pids >= T
    if pid >= T:
        return

    # --- 2. Load Data for this Row ---
    
    # Load the top-p value for this row
    top_p = tl.load(top_p_ptr + pid * stride_top_p)
    
    # Create offsets for the K dimension
    k_offsets = tl.arange(0, BLOCK_S_K)
    k_mask = k_offsets < K
    
    # Load the row of probabilities
    # (row_start_probs) + (k_offsets)
    row_probs_ptr = sorted_probs_ptr + pid * stride_probs_t
    probs = tl.load(row_probs_ptr + k_offsets, mask=k_mask, other=0.0)
    
    # Load the row of indices
    # (row_start_indices) + (k_offsets)
    row_indices_ptr = sorted_indices_ptr + pid * stride_indices_t
    indices = tl.load(row_indices_ptr + k_offsets, mask=k_mask, other=0)

    # --- 3. Pass 1: Find the Top-P Cutoff ---
    
    # We need to find the set of probabilities to sample from.
    # This is the smallest set of probs[0...q] such that cumsum(probs[0...q]) > top_p.
    # A more efficient way to mask this is:
    # We include index `j` if cumsum(probs[0...j-1]) <= top_p.
    
    # Calculate exclusive cumulative sum: [0, p0, p0+p1, p0+p1+p2, ...]
    # We get this by right-shifting the inclusive cumsum and prepending 0.
    excl_cum_probs = tl.cumsum(probs) - probs
    
    # Create a mask for elements to include in sampling
    # sampling_mask[j] is True if excl_cum_probs[j] <= top_p
    sampling_mask = excl_cum_probs <= top_p
    
    # Get the probabilities we are allowed to sample from
    # probs = [0.5, 0.3, 0.1, ...], top_p = 0.75
    # excl_cum_probs = [0.0, 0.5, 0.8, ...]
    # sampling_mask = [T, T, F, ...]
    # valid_probs = [0.5, 0.3, 0.0, ...]
    valid_probs = tl.where(sampling_mask, probs, 0.0)
    
    # Calculate the total probability mass we are sampling from
    # This is our normalization constant
    total_prob_for_sampling = tl.sum(valid_probs, axis=0)

    # --- 4. Pass 2: Sample from the Valid Probs ---
    
    # Generate a random number [0, 1)
    # We use pid and a host-provided offset to ensure different results 
    # across batches and rows.
    rand_offset = offset + pid
    r = tl.rand(seed, rand_offset) #
    
    # Scale the random number by the total probability
    # r_scaled is in [0, total_prob_for_sampling)
    r_scaled = r * total_prob_for_sampling
    
    # Find the first index `j` such that cumsum(valid_probs[0...j]) > r_scaled
    sampling_cum_probs = tl.cumsum(valid_probs)
    
    # Create a mask where r_scaled is less than the cumulative probability
    # [0.5, 0.8, 0.8, ...], r_scaled = 0.7
    # sampling_idx_mask = [F, T, T, ...]
    sampling_idx_mask = r_scaled < sampling_cum_probs
    
    # Find the index of the *first* True value
    # We do this by replacing False with a large value and finding the minimum.
    large_index = BLOCK_S_K
    sampled_k_idx = tl.min(
        tl.where(sampling_idx_mask, k_offsets, large_index), 
        axis=0
    )
    
    # --- FIX for edge case ---
    # If total_prob_for_sampling was 0, r_scaled is 0, 
    # sampling_cum_probs is all 0, sampling_idx_mask is all False,
    # and sampled_k_idx will be large_index.
    # We fix this by defaulting to index 0 if total_prob_for_sampling is 0.
    sampled_k_idx = tl.where(total_prob_for_sampling == 0.0, 0, sampled_k_idx)

    # --- 5. Gather Sampled Index and Probability ---
    
    # `sampled_k_idx` is a scalar (tensor) holding the index (0 <= idx < K)
    # We need to use it to look up the original index and probability.
    # This is a dynamic index (a "gather") within the block.
    
    # Create a mask to select the `sampled_k_idx`
    # e.g., if sampled_k_idx = 1, select_mask = [F, T, F, F, ...]
    select_mask = (k_offsets == sampled_k_idx)
    
    # Use the mask to get the corresponding original index
    # tl.sum works because only one element is True
    selected_original_index = tl.sum(
        tl.where(select_mask, indices, 0), 
        axis=0
    )
    
    # --- 6. Write Output ---
    
    # Store the final sampled original index
    tl.store(output_indices_ptr + pid * stride_out_idx, selected_original_index)
    
    # If requested, also store the probability of that sample
    if STORE_PROBS:
        selected_prob = tl.sum(
            tl.where(select_mask, probs, 0.0), 
            axis=0
        )
        tl.store(output_probs_ptr + pid * stride_out_prob, selected_prob)


def awsm_sample_top_p(
    sorted_probs: torch.Tensor,
    sorted_indices: torch.Tensor,
    top_p: torch.Tensor | float = 1.0,
    chosen_token_indices: torch.Tensor = None,
    to_store_probs: bool = False,
    output_probs: torch.Tensor = None,
    seed: int = 42,
    offset: int = 0,
):
    """
    Python wrapper for the Top-P sampling kernel.

    Args:
        sorted_probs (torch.Tensor): (T, K) tensor of probabilities, 
                                     sorted descending.
        sorted_indices (torch.Tensor): (T, K) tensor of original indices.
        seed (int): RNG seed.
        top_p (torch.Tensor | float, optional): (T,) tensor of p-values 
                                     for each row, or a single float
                                     to be broadcasted to all rows. 
                                     Defaults to 1.0.
        offset (int, optional): RNG offset, change this for different 
                                sampling results. Defaults to 0.
        output_probs (torch.Tensor, optional): Pre-allocated (T,) tensor 
                                 to store output probabilities. 
                                 If None, probabilities are not stored.

    Returns:
        torch.Tensor: (T,) tensor of sampled original indices.
        torch.Tensor (optional): (T,) tensor of sampled probabilities, 
                                 if output_probs was provided.
    """
    # --- 1. Input Validation ---
    assert sorted_probs.is_cuda and sorted_indices.is_cuda, \
           "sorted_probs and sorted_indices must be on CUDA."
           
    assert sorted_probs.is_contiguous() and \
           sorted_indices.is_contiguous(), \
           "sorted_probs and sorted_indices must be contiguous."
           
    assert sorted_probs.shape == sorted_indices.shape, \
           "Probs and indices must have the same shape."
           
    T, K = sorted_probs.shape

    # Handle top_p being a float or a tensor
    if isinstance(top_p, float):
        # Broadcast float to a tensor
        top_p_tensor = torch.full(
            (T,), 
            top_p, 
            dtype=torch.float32, 
            device=sorted_probs.device
        )
    elif isinstance(top_p, torch.Tensor):
        # Use existing tensor
        top_p_tensor = top_p
        assert top_p_tensor.is_cuda, "top_p tensor must be on CUDA."
        assert top_p_tensor.is_contiguous(), "top_p tensor must be contiguous."
        assert top_p_tensor.shape == (T,), "top_p must have shape (T,)."
    else:
        raise TypeError("top_p must be a float or a torch.Tensor.")

    assert sorted_indices.dtype == torch.int64, \
           "sorted_indices must be of dtype torch.int64."

    # --- 2. Output Allocation ---
    # Allocate output tensor for indices

    if chosen_token_indices is not None:
        assert chosen_token_indices.shape == (T,), "chosen_token_indices must have shape (T,)."
        assert chosen_token_indices.device == sorted_probs.device, \
               "chosen_token_indices must be on the same device."
        assert chosen_token_indices.is_contiguous()
        assert chosen_token_indices.dtype == torch.int64
    else:
        chosen_token_indices = torch.empty(
            T, dtype=torch.int64, device=sorted_probs.device
        )
    
    # Handle optional probability output
    STORE_PROBS = to_store_probs
    if STORE_PROBS:
        if output_probs is None:
            output_probs = torch.empty(
                T, dtype=torch.float32, device=sorted_probs.device
            )
        assert output_probs.shape == (T,), "output_probs must have shape (T,)."
        assert output_probs.device == sorted_probs.device, \
               "output_probs must be on the same device."
        assert output_probs.is_contiguous()
        stride_out_prob = output_probs.stride(0)
    else:
        # Pass a dummy tensor. It won't be written to.
        output_probs = output_indices 
        stride_out_prob = 0 # This stride won't be used


    # --- 3. Kernel Launch ---
    
    # Kernel grid
    grid = (T,)
    
    # Helper function to find the smallest power of 2 >= K
    # This replaces the dependency on triton.utils
    def _next_power_of_2(n):
        if n == 0:
            return 1
        # If n is already a power of 2, return it
        if (n & (n - 1) == 0) and n > 0:
            return n
        # Otherwise, find the next power of 2
        return 1 << (n - 1).bit_length()

    # Find the smallest power of 2 >= K
    BLOCK_S_K = _next_power_of_2(K)

    # Launch the kernel
    top_p_sampler_kernel[grid](
        # Pointers
        sorted_probs,
        sorted_indices,
        top_p_tensor, # Use the processed tensor
        chosen_token_indices,
        output_probs,
        # Dims
        T, K,
        # RNG
        seed, offset,
        # Strides
        sorted_probs.stride(0), sorted_probs.stride(1),
        sorted_indices.stride(0), sorted_indices.stride(1),
        top_p_tensor.stride(0), # Use the processed tensor
        chosen_token_indices.stride(0),
        stride_out_prob,
        # Constexpr
        BLOCK_S_K=BLOCK_S_K,
        STORE_PROBS=STORE_PROBS
    )
    
    # --- 4. Return ---
    if STORE_PROBS:
        return chosen_token_indices, output_probs
    else:
        return chosen_token_indices, None