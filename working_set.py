from awsm_transformer import get_hardware_env
from awsm_transformer import get_torch_dtype
from awsm_transformer.utils import *
from awsm_transformer.saved_activations_policy import get_transformer_saved_act_sizes
import copy
import math

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

    ### Require backbone training state in host memory
    if training_config is not None and num_local_layers > 0:

        backbone_master_bytes = get_backbone_layer_size_bytes(master_dims)
        backbone_grad_bytes = get_backbone_layer_size_bytes(grad_dims)
        backbone_opt_bytes = opt_mult * get_backbone_layer_size_bytes(opt_dims)

        required_host_bytes += num_local_layers * (backbone_master_bytes + backbone_grad_bytes + backbone_opt_bytes)
        
        ## require at least 2 layers in GPU memory of total training state for simplicity
        required_gpu_bytes += min(num_local_layers, 2) * (backbone_master_bytes + backbone_grad_bytes + backbone_opt_bytes)
    ### Require at least 1 backbone layer in GPU memory
    elif num_local_layers > 0:
        backbone_weight_bytes = get_backbone_layer_size_bytes(model_dims)

        ## require all layers to be in host memory
        required_host_bytes += num_local_layers * backbone_weight_bytes
        
        ## require at least 1 layer in GPU memory of total weight bytes
        required_gpu_bytes += backbone_weight_bytes
        
    return required_gpu_bytes, required_host_bytes, backbone_opt_bytes

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

    static_gpu_bytes = required_gpu_bytes

    ## Require at least 2 full activation slot in GPU memory
    full_act_slot_size = get_full_act_slot_size_bytes(model_dims, chunk_size)
    required_gpu_bytes += 2 *full_act_slot_size

    return required_gpu_bytes, static_gpu_bytes


def determine_working_set_config(model_dims, max_seq_len, max_global_batch_tokens, training_config=None, has_embed=True, has_head=True, num_local_layers=None, chunk_size = None, max_gpu_mem_bytes=None, max_host_mem_bytes=None, leeway_gpu_mem_bytes=3e9, leeway_host_mem_bytes=10e9, verbose=False, device_id=0, min_tokens_per_round=4096, fixed_seq_len=None):

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

    
    baseline_gpu_bytes, baseline_host_bytes, backbone_opt_bytes = get_baseline_model_memory_requirements(model_dims, num_local_layers, training_config=training_config, has_embed=has_embed, has_head=has_head)
    
    if max_gpu_mem_bytes < baseline_gpu_bytes:
        raise ValueError(f"max_gpu_mem_bytes ({max_gpu_mem_bytes / 1e9:,.3f}GB) is less than required minimum baseline_gpu_bytes ({baseline_gpu_bytes / 1e9:,.2f}GB)")
    if max_host_mem_bytes < baseline_host_bytes:
        raise ValueError(f"max_host_mem_bytes ({max_host_mem_bytes / 1e9:,.3f}GB) is less than required minimum baseline_host_bytes ({baseline_host_bytes / 1e9:,.2f}GB)")

    remaining_gpu_mem_bytes = max_gpu_mem_bytes - baseline_gpu_bytes
    remaining_host_mem_bytes = max_host_mem_bytes - baseline_host_bytes

    if verbose:
        print(f"[Working Set Log] After Baseline Model Memory Requirements, Determined: Remaining GPU Memory of {remaining_gpu_mem_bytes / 1e9:,.2f}GB and Remaining Host Memory of {remaining_host_mem_bytes / 1e9:,.2f}GB")
    

    ### Retrieve worse-case transfer latency of weights
    layer_transfer_duration_sec = baseline_hardware_env["transfer_report"]["layer_concurrent_transfer_duration_sec"]

    est_tflops = baseline_hardware_env["basic_peak_tflops_est"]
    est_mem_bw_gb_per_sec = baseline_hardware_env["basic_peak_mem_bandwidth_gb_per_sec"]

    if verbose:
        print(f"[Working Set Log] Observed Layer Transfer Duration of {layer_transfer_duration_sec * 1e3:.2f} ms, Estimated Peak TFLOPS: {est_tflops:.2f}, Estimated Memory Bandwidth: {est_mem_bw_gb_per_sec:.2f} GB/s")

    matmul_flops_per_token = get_layer_matmul_flops_per_token(model_dims)

    remaining_total_mem = remaining_gpu_mem_bytes + remaining_host_mem_bytes

    ## need to store transitions
    d_model = model_dims["d_model"]
    ctx_dim = model_dims["head_dim"] * model_dims["n_kv_heads"]
    residual_dtype = get_torch_dtype(model_dims["datatypes"]["residual"])

    ### <= 100% recomputation and no kv recomputaiton constraints
    recomp_lim_max_tokens_per_round = remaining_total_mem / ((d_model + 2 * ctx_dim) * num_local_layers * residual_dtype.itemsize)

    ## accounting for device context windows (fwd + bwd) and transition table
    gpu_lim_max_tokens_per_round = (remaining_gpu_mem_bytes - max_seq_len * 4 * ctx_dim * residual_dtype.itemsize) / (d_model * residual_dtype.itemsize)

    ## a decent heuristic for max potential tokens per round, though we want to find
    ## the smallest limit that still gives good performance
    max_tokens_per_round = int(min(recomp_lim_max_tokens_per_round, gpu_lim_max_tokens_per_round))

    if max_tokens_per_round < max_seq_len:
        raise ValueError(f"Could not find a valid configuration for seq len {max_seq_len}; estimating max tokens per round to be {max_tokens_per_round}")
    
    if verbose:
        print(f"[Working Set Log] Determined Max Tokens Per Round of {max_tokens_per_round} based on aggregate available memory of {remaining_total_mem / 1e9:.2f}GB")

    ## for usual seq lens ignore attention flops and try to have enough tokens to hide weight/gradient transfer latency
    ## this sets a good upper bound for number of tokens per round, then we can lessen this to fit within memory constraints
    ## might want a factor of 2 for the transfer duration to ensure no stalls during bwd, but this is normally good (also uss peak tflops instead of realistic/with recompute)
    target_upper_bound_tokens_per_round_est = min(max_tokens_per_round, math.ceil((layer_transfer_duration_sec * est_tflops * 1e12) / matmul_flops_per_token))

    target_upper_bound_tokens_per_round = max(min_tokens_per_round, prev_high_div(target_upper_bound_tokens_per_round_est))

    if verbose:
        print(f"[Working Set Log] Determined Layer Transfer Time of {layer_transfer_duration_sec * 1e3:.2f} ms, Orig Target Tokens Per Round Est: {target_upper_bound_tokens_per_round}")

    cur_tokens_per_round = min(max_global_batch_tokens, target_upper_bound_tokens_per_round)

    if fixed_seq_len is not None:
        cur_tokens_per_round = round_to_nearest(cur_tokens_per_round, fixed_seq_len)
        if cur_tokens_per_round > max_tokens_per_round:
            cur_tokens_per_round -= fixed_seq_len
            if cur_tokens_per_round > max_tokens_per_round or cur_tokens_per_round == 0:
                raise ValueError(f"Error: Could not find a valid configuration for fixed seq len {fixed_seq_len}; estimated max tokens per round to be {cur_tokens_per_round}")
            
    
    ## we reuse gpu act buffer for optimizer state so require enough for 2 layers of this
    min_opt_state_bytes = 2 * backbone_opt_bytes

    if verbose:
        print(f"[Working Set Log] Min Opt State/GPU Act Buffer size of: {min_opt_state_bytes}")

    if min_opt_state_bytes > remaining_gpu_mem_bytes:
        raise ValueError("Error: Not enough GPU memory to hold 2 layers of weights, grads, and optimizer state")

    ## We should already be good for overall host memory constraints with valid max_tokens_per_round
    ## but we need to determine valid chunk size; too large a chunk size will incur excess temporary memory usage
    ## (particularly for MoE where we have staging buffers for scatter/gather)
    satisfied=False
    while not satisfied:
        divisors = get_divisors(cur_tokens_per_round)
        divisors.sort(reverse=True)
        for potential_chunk_size in divisors:

                num_chunks = cur_tokens_per_round // potential_chunk_size

                act_required_gpu_bytes, static_gpu_bytes = get_baseline_gpu_activation_memory_requirements(model_dims, max_seq_len, potential_chunk_size, num_chunks, training_config=training_config)

                if act_required_gpu_bytes < remaining_gpu_mem_bytes:
                    ### THESE ARE THE KEY VALUES WE ARE AFTER!
                    max_chunk_size = potential_chunk_size
                    target_tokens_per_round = cur_tokens_per_round
                    satisfied=True
                    break
        
        if not satisfied:
            ### arbitrary; we choose target tokens per round from list of high divisors in utils.py anyways
            ### try for different combination, though only the chunk size should matter here...
            if fixed_seq_len is not None:
                cur_tokens_per_round -= fixed_seq_len
            else:
                cur_tokens_per_round -= 1024

            if cur_tokens_per_round < min_tokens_per_round:
                raise ValueError(f"Error: Not enough memory to run with min tokens per round of: {min_tokens_per_round}")

    est_num_chunks = math.ceil(target_tokens_per_round / max_chunk_size)
    full_act_slot_size = get_full_act_slot_size_bytes(model_dims, max_chunk_size)
    
    ## can use remaining bytes for act space
    gpu_act_bytes = remaining_gpu_mem_bytes - static_gpu_bytes
    
    ## might not need to use all memory
    gpu_act_slots = int(max(1, min(est_num_chunks * num_local_layers, gpu_act_bytes // full_act_slot_size)))

    gpu_act_buffer_size = gpu_act_slots * full_act_slot_size
    
    ## we reuse gpu act buffer during opt step
    assert gpu_act_buffer_size >= min_opt_state_bytes

    est_total_gpu_bytes = static_gpu_bytes + gpu_act_buffer_size + baseline_gpu_bytes

    assert est_total_gpu_bytes <= max_gpu_mem_bytes

    ## Now ensure we have enough host memory for minimal amount of activations

    host_act_slots = int(est_num_chunks * num_local_layers - gpu_act_slots)
    
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
        print(f"[Working Set Log] Determined Max Chunk Size of {max_chunk_size}, Target Tokens Per Round of {target_tokens_per_round}\n\t# GPU Full Act Slots: {gpu_act_slots}\n\t# Host Act Slots: {host_act_slots}\n\t# GPU Act Buffer Size: {gpu_act_buffer_size / 1e9:.2f}GB\n\t# Host Act Buffer Size: {host_act_buffer_size / 1e9:.2f}GB")
        print(f"[Working Set Log] Expected GPU Memory Usage: {est_total_gpu_bytes / 1e9:.2f}GB, Expected Host Memory Usage: {est_total_host_bytes / 1e9:.2f}GB")

    working_set_config = {
        "n_gpu_layers": min(num_local_layers, 2),
        "n_gpu_grads": min(num_local_layers, 2),
        "n_gpu_opt_layers": min(num_local_layers, 2),
        "max_chunk_size": max_chunk_size,
        "max_seq_len": max_seq_len,
        "target_round_tokens": target_tokens_per_round,
        "max_total_round_tokens": round_to_nearest(max_tokens_per_round, 256),
        "host_act_buffer_size": int(host_act_buffer_size),
        "gpu_act_buffer_size": int(gpu_act_buffer_size),
        "max_host_mem_gb": max_host_mem_bytes / 1e9,
        "max_gpu_mem_gb": max_gpu_mem_bytes / 1e9,
    }

    if verbose:
        print("[Working Set Log] Running Hardware Environment Check to Return Estimated Hardware Environment...")

    chosen_hardware_env = get_hardware_env(max_chunk_size, model_dims, device_id=device_id)

    return working_set_config, chosen_hardware_env

