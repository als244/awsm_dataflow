import torch

def get_torch_dtype(dtype_str):
    if dtype_str == "bfloat16" or dtype_str == "bf16" or dtype_str == "BF16":
        return torch.bfloat16
    elif dtype_str == "float16" or dtype_str == "fp16" or dtype_str == "FP16":
        return torch.float16
    elif dtype_str == "float32" or dtype_str == "fp32" or dtype_str == "FP32":
        return torch.float32
    else:
        raise ValueError("Invalid dtype: " + dtype_str)

def get_embedding_size_bytes(model_dims):

    embedding_dtype = get_torch_dtype(model_dims["datatypes"]["embed"])
    
    embedding_element_size_bytes = embedding_dtype.itemsize

    d_model = model_dims["d_model"]
    vocab_size = model_dims["vocab_size"]

    return embedding_element_size_bytes * d_model * vocab_size

def get_head_size_bytes(model_dims):

    head_proj_dtype = get_torch_dtype(model_dims["datatypes"]["head_proj"])
    norm_dtype = get_torch_dtype(model_dims["datatypes"]["norm"])
    
    head_proj_element_size_bytes = head_proj_dtype.itemsize
    norm_element_size_bytes = norm_dtype.itemsize

    d_model = model_dims["d_model"]
    vocab_size = model_dims["vocab_size"]
    
    ## include norm weights
    return head_proj_element_size_bytes * d_model * vocab_size + norm_element_size_bytes * d_model

def get_context_size_bytes(model_dims, context_window_size):
    
    attn_proj_dtype = get_torch_dtype(model_dims["datatypes"]["attn_proj"])
    
    attn_proj_element_size_bytes = attn_proj_dtype.itemsize
    
    n_kv_heads = model_dims["n_kv_heads"]
    head_dim = model_dims["head_dim"]
    
    ctx_dim = n_kv_heads * head_dim
    
    ## 1 for K and 1 for V
    return 2 * context_window_size * ctx_dim * attn_proj_element_size_bytes

def get_backbone_layer_size_bytes(model_dims):

    attn_proj_dtype = get_torch_dtype(model_dims["datatypes"]["attn_proj"])
    expert_proj_dtype = get_torch_dtype(model_dims["datatypes"]["expert_proj"])
    router_dtype = get_torch_dtype(model_dims["datatypes"]["router"])
    norm_dtype = get_torch_dtype(model_dims["datatypes"]["norm"])
    
    d_model = model_dims["d_model"]
    n_heads = model_dims["n_heads"]
    head_dim = model_dims["head_dim"]
    n_kv_heads = model_dims["n_kv_heads"]

    attn_dim = n_heads * head_dim
    ctx_dim = n_kv_heads * head_dim

    attn_proj_element_size_bytes = attn_proj_dtype.itemsize
    expert_proj_element_size_bytes = expert_proj_dtype.itemsize
    router_element_size_bytes = router_dtype.itemsize
    norm_element_size_bytes = norm_dtype.itemsize

    backbone_layer_size_bytes = 0

    ## NORMS

    ## attn and ffn norm
    backbone_layer_size_bytes += 2 * norm_element_size_bytes * d_model

    ## ATTENTION

    ## attn projects

    ## Q and O projections
    backbone_layer_size_bytes += 2 *attn_proj_element_size_bytes * d_model * attn_dim
    ## K and V projections
    backbone_layer_size_bytes += 2 * attn_proj_element_size_bytes * d_model * ctx_dim


    
    ## MLP

    expert_dim = model_dims["expert_dim"]

    ## shared experts
    num_shared_experts = model_dims["num_shared_experts"]
    backbone_layer_size_bytes += num_shared_experts * expert_proj_element_size_bytes * 3 *d_model * expert_dim
    
    ## routed experts
    num_routed_experts = model_dims["num_routed_experts"]
    if num_routed_experts > 0:
        ## router
        backbone_layer_size_bytes += router_element_size_bytes * d_model * num_routed_experts
        ## routed expert projects
        backbone_layer_size_bytes += num_routed_experts * expert_proj_element_size_bytes * 3 * d_model * expert_dim
    
    ## router
    backbone_layer_size_bytes += router_element_size_bytes * d_model * num_routed_experts

    return backbone_layer_size_bytes


def get_full_act_slot_size_bytes(model_dims, chunk_size):

        d_model = model_dims["d_model"]
        n_heads = model_dims["n_heads"]
        n_kv_heads = model_dims["n_kv_heads"]
        head_dim = model_dims["head_dim"]
        expert_dim = model_dims["expert_dim"]
        num_shared_experts = model_dims["num_shared_experts"]
        num_routed_experts = model_dims["num_routed_experts"]
        top_k = model_dims["top_k"]

        act_dtype = get_torch_dtype(model_dims["datatypes"]["residual"])
        router_dtype = get_torch_dtype(model_dims["datatypes"]["router"])

        attn_norm_rstd_size = chunk_size * torch.float32.itemsize
        ffn_norm_rstd_size = chunk_size * torch.float32.itemsize
        x_inp_size = chunk_size * d_model * act_dtype.itemsize
        xk_size = chunk_size * n_kv_heads * head_dim * act_dtype.itemsize
        xv_size = chunk_size * n_kv_heads * head_dim * act_dtype.itemsize
        x_router_size = chunk_size * num_routed_experts * router_dtype.itemsize
        expert_counts_size = num_routed_experts * torch.int32.itemsize
        router_weights_size = chunk_size * top_k * router_dtype.itemsize
        chosen_experts_size = chunk_size * top_k * torch.int32.itemsize
        scattered_router_weights_size = chunk_size * top_k * router_dtype.itemsize
        attn_result_size = chunk_size * n_heads * head_dim * act_dtype.itemsize
        softmax_lse_size = n_heads * chunk_size * torch.float32.itemsize
        xq_size = chunk_size * n_heads * head_dim * act_dtype.itemsize
        xo_size = chunk_size * d_model * act_dtype.itemsize
        x_up_size = chunk_size * top_k * 2 * expert_dim * act_dtype.itemsize
        x_up_shared_size = chunk_size * num_shared_experts * 2 * expert_dim * act_dtype.itemsize

        return attn_norm_rstd_size + ffn_norm_rstd_size + x_inp_size + xk_size + xv_size + x_router_size + expert_counts_size + router_weights_size + chosen_experts_size + scattered_router_weights_size + attn_result_size + softmax_lse_size + xq_size + xo_size + x_up_size + x_up_shared_size

def get_min_act_slot_size_bytes(model_dims, chunk_size):

        d_model = model_dims["d_model"]
        n_heads = model_dims["n_heads"]
        n_kv_heads = model_dims["n_kv_heads"]
        head_dim = model_dims["head_dim"]
        expert_dim = model_dims["expert_dim"]
        num_shared_experts = model_dims["num_shared_experts"]
        num_routed_experts = model_dims["num_routed_experts"]
        top_k = model_dims["top_k"]

        act_dtype = get_torch_dtype(model_dims["datatypes"]["residual"])
        router_dtype = get_torch_dtype(model_dims["datatypes"]["router"])

        attn_norm_rstd_size = chunk_size * torch.float32.itemsize
        ffn_norm_rstd_size = chunk_size * torch.float32.itemsize
        x_inp_size = chunk_size * d_model * act_dtype.itemsize
        xk_size = chunk_size * n_kv_heads * head_dim * act_dtype.itemsize
        xv_size = chunk_size * n_kv_heads * head_dim * act_dtype.itemsize
        x_router_size = chunk_size * num_routed_experts * router_dtype.itemsize
        expert_counts_size = num_routed_experts * torch.int32.itemsize
        router_weights_size = chunk_size * top_k * router_dtype.itemsize
        chosen_experts_size = chunk_size * top_k * torch.int32.itemsize
        scattered_router_weights_size = chunk_size * top_k * router_dtype.itemsize

        return attn_norm_rstd_size + ffn_norm_rstd_size + x_inp_size + xk_size + xv_size + x_router_size + expert_counts_size + router_weights_size + chosen_experts_size + scattered_router_weights_size


### Only for forward pass
def get_layer_matmul_flops_per_token(model_dims):

    model_dim = model_dims["d_model"]
    n_heads = model_dims["n_heads"]
    head_dim = model_dims["head_dim"]
    n_kv_heads = model_dims["n_kv_heads"]
    expert_dim = model_dims["expert_dim"]
    vocab_size = model_dims["vocab_size"]
    top_k = model_dims["top_k"]
    num_shared_experts = model_dims["num_shared_experts"]
    is_causal = model_dims["is_causal"]

    ctx_dim = n_kv_heads * head_dim
    attn_dim = n_heads * head_dim

    active_params_per_layer = 2 * model_dim * attn_dim + 2 * model_dim * ctx_dim + 3 * (num_shared_experts + top_k) *model_dim * expert_dim

    return 2 * active_params_per_layer
    

    

def get_model_flops_per_sequence(seq_len, model_dims):

    n_layers = model_dims["n_layers"]
    model_dim = model_dims["d_model"]
    n_heads = model_dims["n_heads"]
    head_dim = model_dims["head_dim"]
    n_kv_heads = model_dims["n_kv_heads"]
    expert_dim = model_dims["expert_dim"]
    vocab_size = model_dims["vocab_size"]
    top_k = model_dims["top_k"]
    num_shared_experts = model_dims["num_shared_experts"]
    is_causal = model_dims["is_causal"]

    ctx_dim = n_kv_heads * head_dim
    attn_dim = n_heads * head_dim

    active_params_per_layer = 2 * model_dim * attn_dim + 2 * model_dim * ctx_dim + 3 * (num_shared_experts + top_k) *model_dim * expert_dim 
    
    matmul_flops_per_layer = 6 * seq_len * active_params_per_layer

    attn_flop_factor = 1.0
    if is_causal:
        attn_flop_factor = 0.5

    attn_flops_per_layer = 12 * attn_flop_factor * seq_len * seq_len * attn_dim

    backbone_flops = n_layers * (matmul_flops_per_layer + attn_flops_per_layer)

    head_active_params = model_dim * vocab_size
    head_flops = 6 * seq_len * head_active_params

    total_flops = backbone_flops + head_flops

    return total_flops

def round_to_nearest(x, base):
    # Divide the number by the base, round to the nearest integer, then multiply by the base
    rounded_x = base * round(x / base)
    # Cast to int if the result is a float
    return int(rounded_x)

def round_to_nearest_divisor(value, divisor_of, direction=None):
    """
    Round 'value' to the nearest divisor of 'divisor_of'.
    
    Args:
        value: The number to round
        divisor_of: Find divisors of this number
        direction: None (nearest), 'up' (ceiling), or 'down' (floor)
    
    Returns:
        The divisor of 'divisor_of' closest to 'value' in the specified direction
    """
    if divisor_of == 0:
        raise ValueError("divisor_of cannot be zero")
    
    divisor_of = abs(divisor_of)
    
    # Find all divisors
    divisors = []
    for i in range(1, int(divisor_of**0.5) + 1):
        if divisor_of % i == 0:
            divisors.append(i)
            if i != divisor_of // i:
                divisors.append(divisor_of // i)
    
    divisors.sort()
    
    if direction is None:
        # Nearest divisor
        return min(divisors, key=lambda d: abs(d - value))
    
    elif direction == 'down':
        # Largest divisor <= value
        candidates = [d for d in divisors if d <= value]
        if not candidates:
            return 1
        return max(candidates)
    
    elif direction == 'up':
        # Smallest divisor >= value
        candidates = [d for d in divisors if d >= value]
        if not candidates:
            return divisor_of
        return min(candidates)
    
    else:
        raise ValueError("direction must be None, 'up', or 'down'")


def get_divisors(x):
    divisors = []
    for i in range(1, x + 1):
        if x % i == 0:
            divisors.append(i)
    return divisors


import bisect

# HCNs below 55440, then numbers with >= 100 divisors up to 1081080
GOOD_BATCH_SIZES = [4096, 5040, 7560, 8192, 10080, 15120, 16384, 20160, 25200, 27720, 32768, 45360, 50400,
    55440, 60480, 65520, 65536, 70560, 75600, 80640, 83160, 85680, 90720, 95760,
    100800, 105840, 110880, 115920, 120960, 126000, 131040, 131072, 136080, 141120,
    151200, 155520, 160380, 161280, 166320, 171360, 176400, 181440, 191520,
    196560, 201600, 211680, 216720, 221760, 226800, 231840, 241920, 246960,
    252000, 262080, 262144, 272160, 277200, 282240, 287280, 302400, 317520, 322560,
    327600, 332640, 342720, 352800, 357840, 362880, 378000, 383040, 393120,
    403200, 408240, 415800, 423360, 428400, 432432, 443520, 453600, 468720,
    472500, 478800, 483840, 498960, 504000, 524288, 514080, 524160, 529200, 544320,
    554400, 564480, 574560, 584640, 589680, 604800, 612360, 622440, 635040,
    645120, 655200, 665280, 680400, 685440, 695520, 705600, 720720, 725760,
    730800, 740880, 756000, 766080, 776160, 786240, 800800, 806400, 816480,
    831600, 846720, 856800, 864864, 871200, 876960, 887040, 907200, 917280,
    937440, 942480, 957600, 972720, 982800, 997920, 1009008, 1029600, 1048320, 1048576,
    1058400, 1081080
]

def next_high_div(n):
    """Returns the smallest high-divisor number >= n"""
    idx = bisect.bisect_left(GOOD_BATCH_SIZES, n)
    return GOOD_BATCH_SIZES[idx] if idx < len(GOOD_BATCH_SIZES) else None

def prev_high_div(n):
    """Returns the largest high-divisor number <= n"""
    idx = bisect.bisect_right(GOOD_BATCH_SIZES, n)
    return GOOD_BATCH_SIZES[idx - 1] if idx > 0 else GOOD_BATCH_SIZES[0]

def nearest_high_div(n):
    """Returns the high-divisor number closest to n"""
    idx = bisect.bisect_left(GOOD_BATCH_SIZES, n)
    if idx == 0:
        return GOOD_BATCH_SIZES[0]
    if idx == len(GOOD_BATCH_SIZES):
        return GOOD_BATCH_SIZES[-1]
    before, after = GOOD_BATCH_SIZES[idx - 1], GOOD_BATCH_SIZES[idx]
    return before if (n - before) <= (after - n) else after


def muon_flops(M, N, ns_iters=5):
    return ns_iters * (4 * min(M, N) * min(M, N) * max(M, N) + 2 * min(M, N) * min(M, N) * min(M, N))

def get_opt_flops(model_dims, ns_iters=5, is_muon=False):

    if not is_muon:
        return 0
    
    n_layers = model_dims["n_layers"]

    model_dim = model_dims["d_model"]
    n_heads = model_dims["n_heads"]
    head_dim = model_dims["head_dim"]
    n_kv_heads = model_dims["n_kv_heads"]
    expert_dim = model_dims["expert_dim"]
    num_experts = model_dims["num_shared_experts"] + model_dims["num_routed_experts"]

    attn_dim = n_heads * head_dim
    ctx_dim = n_kv_heads * head_dim
    
    qo_flops = 2 * muon_flops(model_dim, attn_dim)
    kv_flops = 2 * muon_flops(model_dim, ctx_dim)

    exp_flops = 3 * muon_flops(model_dim, expert_dim)
    layer_flops = qo_flops + kv_flops + num_experts * exp_flops

    total_opt_flops = layer_flops * n_layers
    return total_opt_flops

def get_lr(step_num, max_lr=3e-4, warmup_pct=0.1, cooldown_pct=0.2, final_lr=1e-5, est_total_steps=1000):

    #return 2e-4

    warmup_steps = int(est_total_steps * warmup_pct)
    cooldown_steps = int(est_total_steps * cooldown_pct)
    decay_start = est_total_steps - cooldown_steps

    if step_num < warmup_steps:
        return max_lr * (step_num / warmup_steps)
    elif step_num > decay_start:
        return final_lr + max(0, (max_lr - final_lr) * (1 - (step_num - decay_start) / cooldown_steps))
    else:
        return max_lr
