from awsm_transformer import prev_high_div
from awsm_transformer import get_hardware_env
from awsm_transformer import get_torch_dtype
from awsm_transformer.utils import *
from awsm_transformer.saved_activations_policy import get_transformer_saved_act_sizes
import copy
import math

### this is a factor that impacts chunk size estimation
### where > 1 will lean towards bigger matmuls than would be "necessary"
### and creates larger chunk
ARITH_BOUND_FACTOR = 1.25

### BYtes for all layers in host memory, head/grad + 1 full (master + grad + opt) in GPU memory
def get_baseline_model_memory_requirements(model_dims, num_local_layers, training_config=None, has_embed=True, has_head=True):

    required_gpu_bytes = 0
    required_host_bytes = 0

    ### Case of training

    grad_dims = None
    opt_dims = None
    opt_mult = 0

    if training_config is not None:
        master_dims = copy.deepcopy(model_dims)
        for key in master_dims["datatypes"]:
            master_dims["datatypes"][key] = training_config["master_weight_dtype"]

        grad_dims = copy.deepcopy(model_dims)
        for key in grad_dims["datatypes"]:
            grad_dims["datatypes"][key] = training_config["grad_dtype"]

        ## either AdamW or Muon
        opt_choice = training_config["opt_choice"]

        if opt_choice == "AdamW":
            opt_mult = 2
        elif opt_choice == "Muon":
            opt_mult = 1
        else:
            raise ValueError("Invalid opt_choice: Must be AdamW or Muon")


        opt_dims = copy.deepcopy(model_dims)
        for key in opt_dims["datatypes"]:
            opt_dims["datatypes"][key] = training_config["opt_dtype"]

    ### Require embed/head training state in GPU memory

    if has_embed and grad_dims is not None:
        embed_master_bytes = get_embedding_size_bytes(master_dims)
        embed_grad_bytes = get_embedding_size_bytes(grad_dims)
        embed_opt_bytes = opt_mult * get_embedding_size_bytes(opt_dims)

        required_gpu_bytes += embed_master_bytes + embed_grad_bytes + embed_opt_bytes

        ## for simplicity require copy in host memory
        required_host_bytes += embed_master_bytes + embed_grad_bytes + embed_opt_bytes

    if has_head and grad_dims is not None:
        head_master_bytes = get_head_size_bytes(master_dims)
        head_grad_bytes = get_head_size_bytes(grad_dims)
        head_opt_bytes = opt_mult * get_head_size_bytes(opt_dims)

        required_gpu_bytes += head_master_bytes + head_grad_bytes + head_opt_bytes

        ## for simplicity require copy in host memory
        required_host_bytes += head_master_bytes + head_grad_bytes + head_opt_bytes

    backbone_sizes = None
    ### Require backbone training state in host memory
    if training_config is not None and num_local_layers > 0:

        backbone_master_bytes = get_backbone_layer_size_bytes(master_dims)
        backbone_weight_bytes = get_backbone_layer_size_bytes(model_dims)
        backbone_grad_bytes = get_backbone_layer_size_bytes(grad_dims)
        backbone_opt_bytes = opt_mult * get_backbone_layer_size_bytes(opt_dims)

        backbone_sizes = {"master_bytes": backbone_master_bytes, "weight_bytes": backbone_weight_bytes, "grad_bytes": backbone_grad_bytes, "opt_bytes": backbone_opt_bytes}

        required_host_bytes += num_local_layers * (backbone_master_bytes + backbone_grad_bytes + backbone_opt_bytes)
        
        ## require at least 1 layer in GPU memory of total training state
        required_gpu_bytes += (backbone_master_bytes + backbone_grad_bytes + backbone_opt_bytes)
    ### Require at least 1 backbone layer in GPU memory
    elif num_local_layers > 0:
        backbone_weight_bytes = get_backbone_layer_size_bytes(model_dims)

        ## require all layers to be in host memory
        required_host_bytes += num_local_layers * backbone_weight_bytes
        
        ## require at least 1 layer in GPU memory of total weight bytes
        required_gpu_bytes += backbone_weight_bytes

    
        
    return required_gpu_bytes, required_host_bytes, backbone_sizes

### this is during computation, so we arent using master weights/opt
### this doesnt account for transition table or context windows

### purpose is to determine how many full layers we can fit in GPU memory
def get_full_compute_layer_size_bytes(model_dims, num_tokens, backbone_sizes):

    weights_bytes = backbone_sizes["weight_bytes"]
    grad_bytes = backbone_sizes["grad_bytes"]

    training_state_size_bytes = weights_bytes + grad_bytes

    ## now need to account for activations
    act_bytes = get_full_act_slot_size_bytes(model_dims, num_tokens)

    total_layer_bytes = training_state_size_bytes + act_bytes
    
    return total_layer_bytes

def get_model_compute_size_bytes(model_dims, backbone_sizes):

    weights_bytes = backbone_sizes["weight_bytes"]
    grad_bytes = backbone_sizes["grad_bytes"]

    training_state_size_bytes = weights_bytes + grad_bytes
    
    return training_state_size_bytes

def get_context_window_size_bytes(model_dims, max_seq_len, max_chunk_size, is_training=True):
    
    required_gpu_bytes = 0
    
    context_window_size = max(max_chunk_size, max_seq_len)
    min_ctx_bytes = get_context_size_bytes(model_dims, context_window_size)
    required_gpu_bytes += min_ctx_bytes
    ## backwards context window for during
    if is_training:
        required_gpu_bytes += min_ctx_bytes

    return required_gpu_bytes

def get_transition_table_size_bytes(model_dims, num_tokens):
    residual_dtype = get_torch_dtype(model_dims["datatypes"]["residual"])
    return num_tokens * model_dims["d_model"] * residual_dtype.itemsize
    

def get_baseline_gpu_activation_memory_requirements(model_dims, max_seq_len, chunk_size, num_chunks, training_config=None):

    required_gpu_bytes = 0
    d_model = model_dims["d_model"]

    ## Tranisition Table
    tokens_per_round = num_chunks * chunk_size
    residual_dtype = get_torch_dtype(model_dims["datatypes"]["residual"])
    required_gpu_bytes += tokens_per_round * d_model * residual_dtype.itemsize

    ## Context Window
    context_window_size = max(chunk_size, max_seq_len)
    min_ctx_bytes = get_context_size_bytes(model_dims, context_window_size)
    required_gpu_bytes += min_ctx_bytes
    ## backwards context window for during
    if training_config is not None:
        required_gpu_bytes += min_ctx_bytes

    

    ## Working space during execution

    ## for moe models we need scatter space

    ## really should be 2, but having problems here...
    moe_workspace = 2 * (chunk_size * model_dims["top_k"] * model_dims["d_model"]) * residual_dtype.itemsize
    dense_workspace = model_dims["num_shared_experts"] * chunk_size * model_dims["expert_dim"] * residual_dtype.itemsize
    gpu_working_space_bytes = moe_workspace + dense_workspace
    required_gpu_bytes += gpu_working_space_bytes

    return required_gpu_bytes


def determine_working_set_config(model_dims, max_seq_len, max_global_batch_tokens, training_config=None, has_embed=True, has_head=True, num_local_layers=None, chunk_size = None, max_gpu_mem_bytes=None, max_host_mem_bytes=None, leeway_gpu_mem_bytes=3e9, leeway_host_mem_bytes=10e9, verbose=False, device_id=0, min_tokens_per_round=4096, fixed_seq_len=False, min_chunk_size=None, max_chunk_size=None):

    if num_local_layers is None:
        num_local_layers = model_dims["n_layers"]
    
    ### Get baseline Hardware Environment with Chunk Size not Set (if not specified)

    if verbose:
        print("[Working Set Log] Obtaining Baseline Hardware Environment...")

    baseline_hardware_env = get_hardware_env(chunk_size, model_dims, device_id=device_id)

    

    available_gpu_memory_capacity_bytes = baseline_hardware_env["available_gpu_memory_capacity"]
    available_host_memory_capacity_bytes = baseline_hardware_env["available_host_memory_capacity"]

    if verbose:
        print(f"[Working Set Log] Observed Available GPU Memory Capacity of {available_gpu_memory_capacity_bytes / 1e9:.2f}GB and Host Memory Capacity of {available_host_memory_capacity_bytes / 1e9:.2f}GB")

    if max_gpu_mem_bytes is None:
        max_gpu_mem_bytes = available_gpu_memory_capacity_bytes - leeway_gpu_mem_bytes
    else:
        if max_gpu_mem_bytes > available_gpu_memory_capacity_bytes:
            raise ValueError("max_gpu_mem_bytes is greater than available_gpu_memory_capacity_bytes")

    if max_host_mem_bytes is None:
        max_host_mem_bytes = available_host_memory_capacity_bytes - leeway_host_mem_bytes
    else:
        if max_host_mem_bytes > available_host_memory_capacity_bytes:
            raise ValueError("max_host_mem_bytes is greater than available_host_memory_capacity_bytes")

    
    baseline_gpu_bytes, baseline_host_bytes, backbone_sizes = get_baseline_model_memory_requirements(model_dims, num_local_layers, training_config=training_config, has_embed=has_embed, has_head=has_head)
    
    if max_gpu_mem_bytes < baseline_gpu_bytes:
        raise ValueError(f"max_gpu_mem_bytes ({max_gpu_mem_bytes / 1e9:,.3f}GB) is less than required minimum baseline_gpu_bytes ({baseline_gpu_bytes / 1e9:,.2f}GB)")
    if max_host_mem_bytes < baseline_host_bytes:
        raise ValueError(f"max_host_mem_bytes ({max_host_mem_bytes / 1e9:,.3f}GB) is less than required minimum baseline_host_bytes ({baseline_host_bytes / 1e9:,.2f}GB)")

    remaining_gpu_mem_bytes = max_gpu_mem_bytes - baseline_gpu_bytes
    remaining_host_mem_bytes = max_host_mem_bytes - baseline_host_bytes

    if verbose:
        print(f"[Working Set Log] After Baseline Model Memory Requirements, Determined: Remaining GPU Memory of {remaining_gpu_mem_bytes / 1e9:,.2f}GB and Remaining Host Memory of {remaining_host_mem_bytes / 1e9:,.2f}GB")
    

    ### Now we can fit at least 1 full layer in GPU memory (+ embed/head full training state)
    ### We can fit all training state in host memory

    ### We need to determine how many tokens to process each round, which
    ### will then determine how many "full layers" (weights + grads + opt state)  we can fit in GPU memory
    ### lastly we determine chunk size to satisfy 

    ### First get rough upper bound on max tokens per round and ensure we have enough memory to support max_seq_len

    remaining_total_mem = remaining_gpu_mem_bytes + remaining_host_mem_bytes

    ## need to store transitions
    d_model = model_dims["d_model"]
    ctx_dim = model_dims["head_dim"] * model_dims["n_kv_heads"]
    residual_dtype = get_torch_dtype(model_dims["datatypes"]["residual"])

    ### <= 100% intra-layer recomputation and no kv recomputation constraints
    ### here we use aggregate memory because activations can be saved to host
    recomp_lim_max_tokens_per_round = remaining_total_mem / ((d_model + 2 * ctx_dim) * num_local_layers * residual_dtype.itemsize)

    ## accounting for device context windows (fwd + bwd) and transition table
    ## assume ctx is same datatype as residual
    gpu_lim_max_tokens_per_round = (remaining_gpu_mem_bytes - max_seq_len * 4 * ctx_dim * residual_dtype.itemsize) / (d_model * residual_dtype.itemsize)

    ## a decent heuristic for max potential tokens per round, though we want to find
    ## the smallest limit that still gives good performance
    max_tokens_per_round = int(min(recomp_lim_max_tokens_per_round, gpu_lim_max_tokens_per_round))

    max_tokens_per_round = min(max_tokens_per_round, max_global_batch_tokens)

    if max_tokens_per_round < max_seq_len:
        raise ValueError(f"Could not find a valid configuration for seq len {max_seq_len}; estimating max tokens per round to be {max_tokens_per_round}")
    
    if verbose:
        print(f"[Working Set Log] Determined Max Tokens Per Round of {max_tokens_per_round} based on aggregate available memory of {remaining_total_mem / 1e9:.2f}GB, and GPU memory of {remaining_gpu_mem_bytes / 1e9:.2f}GB")




    ### set target upper bound for tokens per round based on transfer duration
    ### (if long seqs in dataset then we can surpass this target)

    ### Simple rule to satisfy is fwd computation time per layer >= layer transfer time + grad transfer time

    ### Retrieve worse-case transfer latency of weights
    layer_transfer_duration_sec = baseline_hardware_env["transfer_report"]["layer_concurrent_transfer_duration_sec"]
    
    ## here gb means GB
    transfer_bandwidth_gb_per_sec = baseline_hardware_env["transfer_report"]["overall_unidirectional_concurrent_bandwidth_gb_per_sec"]

    grad_layer_size = 0
    grad_transfer_duration_sec = 0
    ## during training
    if "grad_bytes" in backbone_sizes:
        grad_layer_size = backbone_sizes["grad_bytes"]
        grad_transfer_duration_sec = grad_layer_size / (transfer_bandwidth_gb_per_sec * 1e9)

    min_layer_computation_time = layer_transfer_duration_sec + grad_transfer_duration_sec

    est_tflops = baseline_hardware_env["basic_peak_tflops_est"]
    est_mem_bw_gb_per_sec = baseline_hardware_env["basic_peak_mem_bandwidth_gb_per_sec"]

    if verbose:
        print(f"[Working Set Log] Observed Layer Transfer Duration of {layer_transfer_duration_sec * 1e3:.2f} ms, Estimated Peak (N=8192 matmul) TFLOPS: {est_tflops:.2f}, Estimated Memory Bandwidth: {est_mem_bw_gb_per_sec:.2f} GB/s")

    ### now we need to determine number of tokens to at least take this long
    matmul_flops_per_token = get_layer_matmul_flops_per_token(model_dims)

    ### as we might not know seqlen ahead of time we can conservatively ignore attention flops
    ### (means more tokens per round than if we accounted for it)
    attn_flops_min_est = 0
    
    ## if fixed seq len we know seq len exactly and can use it to better get better estimate
    ## for layer time (knowing we need at least 1 sequence per round)
    ## if we have multiple seqs per round this is still an underestimate but ok
    if fixed_seq_len:
        attn_factor = 1
        if model_dims["is_causal"]:
            attn_factor = 0.5
        attn_flops_min_est = attn_factor * max_seq_len * max_seq_len * model_dims["head_dim"] * model_dims["n_heads"]   

    flops_per_token_est = matmul_flops_per_token + attn_flops_min_est
    ### matmul computation time should be linearly proportional to tokens per round (if reached arithmetic intensity)
    ### this is likely an overestimate, and we would be ok with less tokens per round

    target_layer_flops = min_layer_computation_time * est_tflops * 1e12

    target_tokens_per_round = round_to_nearest(math.ceil(target_layer_flops / flops_per_token_est), 256)
    
    ### cannot exceed max tokens per round determined by memory constraints
    target_tokens_per_round = min(max_tokens_per_round, target_tokens_per_round)

    ### in case we are testing and want to set a minimum threshold
    if min_tokens_per_round is not None:
        target_tokens_per_round = max(min_tokens_per_round, target_tokens_per_round)

    if fixed_seq_len:
        target_tokens_per_round = max(max_seq_len, round_to_nearest(target_tokens_per_round, max_seq_len))
        if target_tokens_per_round > max_tokens_per_round:
            target_tokens_per_round -= max_seq_len
            if target_tokens_per_round > max_tokens_per_round or target_tokens_per_round == 0:
                raise ValueError(f"Error: Could not find a valid configuration for fixed seq len {fixed_seq_len}; estimated max tokens per round to be {target_tokens_per_round}")
    else:
        target_tokens_per_round = prev_high_div(target_tokens_per_round)

    if min_chunk_size is not None:
        target_tokens_per_round = max(min_chunk_size, target_tokens_per_round)

    target_tokens_per_round = min(max_global_batch_tokens, target_tokens_per_round)

    if verbose:
         print(f"[Working Set Log] Determined Initial Target Tokens Per Round Est: {target_tokens_per_round}")


    ### get estimate for minimum chunk size based on MLP (important for MoE)
    hardware_arith_bound = (est_tflops * 1e12) / (est_mem_bw_gb_per_sec * 1e9)

    H = hardware_arith_bound
    K = model_dims["expert_dim"]
    N = model_dims["d_model"]

    if model_dims["num_routed_experts"] > 0:
        target_min_tokens_per_exp = ARITH_BOUND_FACTOR * H * K * N / (K * N - H * (K + N))
        inv_sparsity_factor = model_dims["num_routed_experts"] / model_dims["top_k"]
        init_target_min_chunk_size = inv_sparsity_factor * target_min_tokens_per_exp
    else:
        init_target_min_chunk_size = ARITH_BOUND_FACTOR * H * K * N / (K * N - H * (K + N))
    
    if verbose:
        print(f"[Working Set Log] Determined Initial Target Min Chunk Size Est (based on Arithmetic Intensity) of: {init_target_min_chunk_size}")

        
    target_chunk_size = round_to_nearest_divisor(init_target_min_chunk_size, target_tokens_per_round, direction="up")

    if verbose:
        print(f"[Working Set Log] Determined Target Chunk Size Est: {target_chunk_size}")

    target_num_chunks = math.ceil(target_tokens_per_round / target_chunk_size)

    if verbose:
        print(f"[Working Set Log] Determined Target Num Chunks Est: {target_num_chunks}")

    ### this includes transition table, context window, and activation workspace
    baseline_act_gpu_memory = get_baseline_gpu_activation_memory_requirements(model_dims, max_seq_len, target_chunk_size, target_num_chunks, training_config=training_config)

    remaining_gpu_mem_bytes -= baseline_act_gpu_memory

    ### first try to fill up the 1st layer worth of act slots
    full_act_slot_size_bytes = get_full_act_slot_size_bytes(model_dims, target_chunk_size)

    first_layer_act_slots = min(target_num_chunks, remaining_gpu_mem_bytes // full_act_slot_size_bytes)

    if first_layer_act_slots < 1:
        raise ValueError("Error: Not enough GPU memory to hold single act slot")

    gpu_act_workspace_size_bytes = first_layer_act_slots * full_act_slot_size_bytes

    remaining_gpu_mem_bytes -= gpu_act_workspace_size_bytes
        
    ### TODO: indicate we need smaller chunk size
    #if cur_remaining_gpu_mem_bytes < 0:
    #    raise ValueError(f"Error: Could not find a valid configuration for target tokens per round {target_tokens_per_round}; estimated remaining GPU memory to be {cur_remaining_gpu_mem_bytes}")

    ### now determine how many complete model layers we should have
    ### At this point we can equally divide remaining GPU memory to know how many complete
    ### layers (weights + grad + activations) we can store, however we will need to account
    ### for chunk size which may increase context window size + be a factor of addition memory workspace
    additional_full_compute_layer_size_bytes = get_full_compute_layer_size_bytes(model_dims, target_tokens_per_round, backbone_sizes)

    ### this is on top of the 1 full layer we have as part of baseline
    additional_complete_layers_est = min(num_local_layers - 1, remaining_gpu_mem_bytes // additional_full_compute_layer_size_bytes)

    if verbose:
        print(f"[Working Set Log] Determined # Additional Complete Layers (weights + grad + act slots): {additional_complete_layers_est}")
    
    n_gpu_layers = 1 + additional_complete_layers_est
    n_gpu_grad_layers = 1 + additional_complete_layers_est

    complete_layers_size_est = additional_complete_layers_est * additional_full_compute_layer_size_bytes
    
    leftover_post_complete_layers_bytes = remaining_gpu_mem_bytes - complete_layers_size_est

    ### baseline for act workspace
    gpu_act_workspace_size_bytes += additional_complete_layers_est * get_full_act_slot_size_bytes(model_dims, target_tokens_per_round)
    
    if gpu_act_workspace_size_bytes < backbone_sizes["opt_bytes"]:
        gpu_act_workspace_size_bytes += leftover_post_complete_layers_bytes
        if gpu_act_workspace_size_bytes < backbone_sizes["opt_bytes"]:
            raise ValueError("Error: Not enough GPU memory to have act buffer > 1 layer of opt state")
    else:
        ### if we can fit addtional model layer give priority to that, then grad layer then act workspace
        if leftover_post_complete_layers_bytes >= backbone_sizes["weight_bytes"]:
            n_gpu_layers += 1
            leftover_post_complete_layers_bytes -= backbone_sizes["weight_bytes"]
        if leftover_post_complete_layers_bytes >= backbone_sizes["grad_bytes"]:
            n_gpu_grad_layers += 1
            leftover_post_complete_layers_bytes -= backbone_sizes["grad_bytes"]
        gpu_act_workspace_size_bytes += leftover_post_complete_layers_bytes

    total_act_slots = target_num_chunks * num_local_layers

    gpu_act_slots = min(total_act_slots, gpu_act_workspace_size_bytes // full_act_slot_size_bytes)
    
    gpu_act_buffer_size = gpu_act_slots * full_act_slot_size
    
    ## we reuse gpu act buffer during opt step
    assert gpu_act_buffer_size >= backbone_sizes["opt_bytes"]

    n_gpu_opt_layers = gpu_act_buffer_size // backbone_sizes["opt_bytes"]
    
    est_total_gpu_bytes = baseline_act_gpu_memory + gpu_act_workspace_size_bytes + backbone_sizes["weight_bytes"] * n_gpu_layers + backbone_sizes["grad_bytes"] * n_gpu_grad_layers 

    assert est_total_gpu_bytes <= max_gpu_mem_bytes

    ## Now ensure we have enough host memory for minimal amount of activations

    host_act_slots = total_act_slots - gpu_act_slots
    
    ### Will not need more than this amount of host memory
    max_host_act_buffer_size = host_act_slots * full_act_slot_size

    host_act_buffer_size = min(max_host_act_buffer_size, remaining_host_mem_bytes)

    est_total_host_bytes = host_act_buffer_size + baseline_host_bytes

    saved_act_sizes = get_transformer_saved_act_sizes(model_dims, max_chunk_size)
    min_act_slot_size_bytes = saved_act_sizes[0]

    min_host_act_buffer_size = host_act_slots * min_act_slot_size_bytes

    assert host_act_buffer_size >= min_host_act_buffer_size
    
    assert est_total_host_bytes <= max_host_mem_bytes
    
    if verbose:
        print(f"[Working Set Log] Determined Target Max Chunk Size of {target_chunk_size}, Target Tokens Per Round of {target_tokens_per_round}\n\t# GPU Full Act Slots: {gpu_act_slots}\n\t# Host Act Slots: {host_act_slots}\n\t# GPU Act Buffer Size: {gpu_act_buffer_size / 1e9:.2f}GB\n\t# Host Act Buffer Size: {host_act_buffer_size / 1e9:.2f}GB")
        print(f"[Working Set Log] Expected GPU Memory Usage: {est_total_gpu_bytes / 1e9:.2f}GB, Expected Host Memory Usage: {est_total_host_bytes / 1e9:.2f}GB")

    working_set_config = {
        "n_gpu_layers": n_gpu_layers,
        "n_gpu_grads": n_gpu_grad_layers,
        "n_gpu_opt_layers": n_gpu_opt_layers,
        "max_chunk_size": target_chunk_size,
        "max_seq_len": max_seq_len,
        "target_round_tokens": target_tokens_per_round,
        "max_total_round_tokens": max_tokens_per_round,
        "host_act_buffer_size": int(host_act_buffer_size),
        "gpu_act_buffer_size": int(gpu_act_buffer_size),
        "max_host_mem_gb": max_host_mem_bytes / 1e9,
        "max_gpu_mem_gb": max_gpu_mem_bytes / 1e9,
    }

    if verbose:
        print("[Working Set Log] Running Hardware Environment Check to Return Estimated Hardware Environment...")

    chosen_hardware_env = get_hardware_env(max_chunk_size, model_dims, device_id=device_id)

    return working_set_config, chosen_hardware_env

