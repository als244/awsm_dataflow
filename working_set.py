from awsm_transformer import prev_high_div
from awsm_transformer import get_hardware_env
from awsm_transformer import get_torch_dtype
from awsm_transformer.utils import *
from awsm_transformer.saved_activations_policy import get_transformer_saved_act_sizes
import copy
import math

### this is critical a factor that impacts chunk size estimation
### where > 1 will lean towards bigger matmuls than would be "necessary"
### and creates larger chunk

### during training we want very compute bound to hide latency of other ops
### so we use a large arith bound factor
ARITH_BOUND_FACTOR = 2

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

    endpoint_sizes = {"embed_bytes": 0, "head_bytes": 0}

    if has_embed and grad_dims is not None:
        embed_master_bytes = get_embedding_size_bytes(master_dims)
        embed_grad_bytes = get_embedding_size_bytes(grad_dims)

        ### endpoints use AdamW
        embed_opt_mult = 2

        embed_opt_bytes = embed_opt_mult * get_embedding_size_bytes(opt_dims)

        required_gpu_bytes += embed_master_bytes + embed_grad_bytes + embed_opt_bytes

        endpoint_sizes["embed_bytes"] = embed_master_bytes + embed_grad_bytes + embed_opt_bytes

        ## for simplicity require copy in host memory
        required_host_bytes += embed_master_bytes + embed_grad_bytes + embed_opt_bytes

    if has_head and grad_dims is not None:
        head_master_bytes = get_head_size_bytes(master_dims)
        head_grad_bytes = get_head_size_bytes(grad_dims)

        ### endpoints use AdamW
        head_opt_mult = 2

        head_opt_bytes = head_opt_mult * get_head_size_bytes(opt_dims)

        required_gpu_bytes += head_master_bytes + head_grad_bytes + head_opt_bytes

        endpoint_sizes["head_bytes"] = head_master_bytes + head_grad_bytes + head_opt_bytes

        ## for simplicity require copy in host memory
        required_host_bytes += head_master_bytes + head_grad_bytes + head_opt_bytes

    backbone_sizes = None
    ### Require backbone training state in host memory
    if training_config is not None and num_local_layers > 0:

        backbone_master_bytes = get_backbone_layer_size_bytes(master_dims)
        backbone_weight_bytes = get_backbone_layer_size_bytes(model_dims)
        backbone_grad_bytes = get_backbone_layer_size_bytes(grad_dims)
        backbone_opt_bytes = opt_mult * get_backbone_layer_size_bytes(opt_dims)

        ### Need to account for fact that router and norms use adamW
        if opt_choice == "Muon":
            backbone_opt_bytes += get_torch_dtype(training_config["opt_dtype"]).itemsize * (2 * model_dims["d_model"] + model_dims["num_routed_experts"] * model_dims["d_model"])

        backbone_sizes = {"master_bytes": backbone_master_bytes, "weight_bytes": backbone_weight_bytes, "grad_bytes": backbone_grad_bytes, "opt_bytes": backbone_opt_bytes}

        required_host_bytes += num_local_layers * (backbone_master_bytes + backbone_grad_bytes + backbone_opt_bytes)
        
        ## require at least 1 layer in GPU memory of weights and gradients, will handle opt separately
        ## as part of activation buffer
        required_gpu_bytes += (backbone_master_bytes + backbone_grad_bytes)
    ### Require at least 1 backbone layer in GPU memory
    elif num_local_layers > 0:
        backbone_weight_bytes = get_backbone_layer_size_bytes(model_dims)

        ## require all layers to be in host memory
        required_host_bytes += num_local_layers * backbone_weight_bytes
        
        ## require at least 1 layer in GPU memory of total weight bytes
        required_gpu_bytes += backbone_weight_bytes

    
        
    return required_gpu_bytes, required_host_bytes, endpoint_sizes, backbone_sizes

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

    ## during backwards to get dX_attn_up and dQ and local dK,dV
    ## flash workspace requires copies of dQ, dK, dV for accumulation
    attn_workspace = chunk_size * (4 * model_dims["n_heads"] * model_dims["head_dim"] + 4 * model_dims["n_kv_heads"] * model_dims["head_dim"]) * residual_dtype.itemsize
    mlp_workspace = 0
    if model_dims["num_routed_experts"] > 0:
        ### for attn norm output, scattered X and scattered upstream
        mlp_workspace = chunk_size * (model_dims["d_model"] + 2 * model_dims["top_k"] * model_dims["d_model"]) * residual_dtype.itemsize
        ### for temporary workspace to do intra-expert backprop
        ### the chunk size * top_k / routed experts is suppose to maximum instead of average, but this shoudld be reasonable good estimate
        ### and should be minimal compared to other workspace requirements for fine-grained moe
        mlp_workspace += 2 * int(chunk_size * model_dims["top_k"] / model_dims["num_routed_experts"]) * 4 * model_dims["expert_dim"] * residual_dtype.itemsize
    else:
        ### during backwards when we compute activation upstream and recomptue forward activations (d act upstream, fwd act, dx1_up, dx3_up)
        mlp_workspace = chunk_size * 4 * model_dims["expert_dim"] * residual_dtype.itemsize

    resid_workspace = chunk_size * model_dims["d_model"] * residual_dtype.itemsize


    gpu_working_space_bytes = resid_workspace + max(attn_workspace, mlp_workspace)
    required_gpu_bytes += gpu_working_space_bytes


    return required_gpu_bytes


def determine_working_set_config(model_dims, max_seq_len, max_global_batch_tokens, training_config=None, has_embed=True, has_head=True, num_local_layers=None, chunk_size = None, max_gpu_mem_bytes=None, max_host_mem_bytes=None, leeway_gpu_mem_bytes=2 * (1 << 30), leeway_host_mem_bytes=10 * (1 << 30), verbose=False, device_id=0, min_tokens_per_round_limit=None, max_tokens_per_round_limit=None, fixed_seq_len=False, min_chunk_size=None, max_chunk_size=None):

    if num_local_layers is None:
        num_local_layers = model_dims["n_layers"]
    
    ### Get baseline Hardware Environment with Chunk Size not Set (if not specified)

    if verbose:
        print("[Working Set Log] Obtaining Baseline Hardware Environment...", flush=True)

    baseline_hardware_env = get_hardware_env(chunk_size, model_dims, device_id=device_id)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    available_gpu_memory_capacity_bytes = baseline_hardware_env["available_gpu_memory_capacity"]
    available_host_memory_capacity_bytes = baseline_hardware_env["available_host_memory_capacity"]

    if verbose:
        print(f"[Working Set Log] Raw Observed Available GPU Memory Capacity of {available_gpu_memory_capacity_bytes / (1 << 30):.2f}GiB and Host Memory Capacity of {available_host_memory_capacity_bytes / (1 << 30):.2f}GiB", flush=True)
        if max_gpu_mem_bytes is not None:
            print(f"[Working Set Log] Inputted Max GPU Memory of {max_gpu_mem_bytes / (1 << 30):.2f}GiB", flush=True)
        if max_host_mem_bytes is not None:
            print(f"[Working Set Log] Inputted Max Host Memory of {max_host_mem_bytes / (1 << 30):.2f}GiB", flush=True)
        print(f"[Working Set Log] Using Leeway of {leeway_gpu_mem_bytes / (1 << 30):.2f}GiB for GPU Memory and {leeway_host_mem_bytes / (1 << 30):.2f}GiB for Host Memory", flush=True)

    if max_gpu_mem_bytes is None:
        max_gpu_mem_bytes = available_gpu_memory_capacity_bytes
    else:
        if max_gpu_mem_bytes > available_gpu_memory_capacity_bytes:
            print(f"Inputted max_gpu_mem_bytes ({max_gpu_mem_bytes}) is greater than available_gpu_memory_capacity_bytes ({available_gpu_memory_capacity_bytes}), setting max gpu bytes to {available_gpu_memory_capacity_bytes}", flush=True)
            max_gpu_mem_bytes = available_gpu_memory_capacity_bytes

    if max_host_mem_bytes is None:
        max_host_mem_bytes = available_host_memory_capacity_bytes   
    else:
        if max_host_mem_bytes > available_host_memory_capacity_bytes:
            print(f"Inputted max_host_mem_bytes ({max_host_mem_bytes}) is greater than available_host_memory_capacity_bytes ({available_host_memory_capacity_bytes}, setting max host bytes to {available_host_memory_capacity_bytes}", flush=True)
            max_host_mem_bytes = available_host_memory_capacity_bytes

    max_gpu_mem_bytes -= leeway_gpu_mem_bytes
    if max_gpu_mem_bytes < 0:
        raise ValueError("max_gpu_mem_bytes is less than 0 after accounting for leeway")
    max_host_mem_bytes -= leeway_host_mem_bytes
    if max_host_mem_bytes < 0:
        raise ValueError("max_host_mem_bytes is less than 0 after accounting for leeway")

    
    baseline_gpu_bytes, baseline_host_bytes, endpoint_sizes, backbone_sizes = get_baseline_model_memory_requirements(model_dims, num_local_layers, training_config=training_config, has_embed=has_embed, has_head=has_head)
    
    if max_gpu_mem_bytes < baseline_gpu_bytes:
        raise ValueError(f"max_gpu_mem_bytes ({max_gpu_mem_bytes / (1 << 30):,.3f}GiB) is less than required minimum baseline_gpu_bytes ({baseline_gpu_bytes / (1 << 30):,.2f}GiB)", flush=True)
    if max_host_mem_bytes < baseline_host_bytes:
        raise ValueError(f"max_host_mem_bytes ({max_host_mem_bytes / (1 << 30):,.3f}GiB) is less than required minimum baseline_host_bytes ({baseline_host_bytes / (1 << 30):,.2f}GiB)", flush=True)

    remaining_gpu_mem_bytes = max_gpu_mem_bytes - baseline_gpu_bytes
    remaining_host_mem_bytes = max_host_mem_bytes - baseline_host_bytes

    if verbose:
        print(f"[Working Set Log] After Baseline Model Memory Requirements and Accounting for Set Memory Bounds, Determined: Remaining GPU Memory of {remaining_gpu_mem_bytes / (1 << 30):,.2f}GiB and Remaining Host Memory of {remaining_host_mem_bytes / (1 << 30):,.2f}GiB", flush=True)
    

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

    if verbose:
        print(f"[Working Set Log] Orig max tok per round: {max_tokens_per_round}\n\tMax global batch tokens: {max_global_batch_tokens}", flush=True)

    max_tokens_per_round = min(max_tokens_per_round, max_global_batch_tokens)

    if max_tokens_per_round < max_seq_len:
        raise ValueError(f"Could not find a valid configuration for seq len {max_seq_len}; estimating max tokens per round to be {max_tokens_per_round}", flush=True)
    
    if verbose:
        print(f"[Working Set Log] Determined Max Tokens Per Round of {max_tokens_per_round} based on aggregate available memory of {remaining_total_mem / (1 << 30):.2f}GiB, and GPU memory of {remaining_gpu_mem_bytes / (1 << 30):.2f}GiB", flush=True)




    ### set target upper bound for tokens per round based on transfer duration

    ### Simple rule to satisfy is fwd computation time per layer >= layer transfer time + grad transfer time
    ### However if low on host memory too many tokens per round induces excessive recomputation
    ### so it is a tricky balance...
    ### Using just layer transfer is a good rule of thumb

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

    #min_layer_computation_time = (2 * layer_transfer_duration_sec + grad_transfer_duration_sec) / 2
    #min_layer_computation_time = layer_transfer_duration_sec
    min_layer_computation_time = layer_transfer_duration_sec + grad_transfer_duration_sec

    ## In some cases less tokens per round => larger window size => less recomputation => higher throughput
    ## this applies to very constrained GPU memory, high ratio of processing speed:interconnect bw, or host memory constrained
    ## regimes, though difference should be ~ 5-10% different and the above formula is more theoretically grounded
    #min_layer_computation_time = layer_transfer_duration_sec

    est_tflops = baseline_hardware_env["basic_peak_tflops_est"]
    est_mem_bw_gb_per_sec = baseline_hardware_env["basic_peak_mem_bandwidth_gb_per_sec"]

    if verbose:
        print(f"[Working Set Log] Observed Layer Transfer Duration of {layer_transfer_duration_sec * 1e3:.2f} ms, Estimated Peak (N=8192 matmul) TFLOPS: {est_tflops:.2f}, Estimated Memory Bandwidth: {est_mem_bw_gb_per_sec:.2f} GB/s", flush=True)

    ### now we need to determine number of tokens to at least take this long
    matmul_flops_per_token = get_layer_matmul_flops_per_token(model_dims)



    ### matmul computation time should be linearly proportional to tokens per round (if reached arithmetic intensity)
    ### this is likely an overestimate, and we would be ok with less tokens per round

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
        attn_flops_min_est = attn_factor * 4 * max_seq_len * max_seq_len * model_dims["head_dim"] * model_dims["n_heads"] 

    target_layer_flops = min_layer_computation_time * est_tflops * 1e12

    target_tokens_per_round = math.ceil((target_layer_flops - attn_flops_min_est) / matmul_flops_per_token)
        
    if verbose:
        print(f"[Working Set Log] Baseline Target Tokens Per Round for Sufficient Computation Time: {target_tokens_per_round}", flush=True)
    
    compute_lim_tokens_per_round = target_tokens_per_round

    if fixed_seq_len:
        target_tokens_per_round = max(max_seq_len, target_tokens_per_round)
    
    full_agg_act_bytes_per_token = num_local_layers * get_full_act_slot_size_bytes(model_dims, 1)
    min_act_bytes_per_token = num_local_layers * get_min_act_slot_size_bytes(model_dims, 1)
    full_save_tokens_per_round = remaining_total_mem // full_agg_act_bytes_per_token
    min_save_tokens_per_round = remaining_total_mem // min_act_bytes_per_token

    if verbose:
        print(f"[Working Set Log] Based on aggregate available memory to save all activations must use <= {full_save_tokens_per_round} tokens per round and to save minimum activations must use <= {min_save_tokens_per_round} tokens per round", flush=True)
        print(f"[Working Set Log] Comparing prior tokens per round: {target_tokens_per_round} with max seq len: {max_seq_len}, full save tokens per round: {full_save_tokens_per_round}, min save tokens per round: {min_save_tokens_per_round} and max tokens per round: {max_tokens_per_round}", flush=True)


    #target_tokens_per_round = max(max_seq_len, min(target_tokens_per_round, full_save_tokens_per_round))

    ## ensure we have enough memory to minimally save activations
    target_tokens_per_round = min(min_save_tokens_per_round, target_tokens_per_round)

    if target_tokens_per_round < max_seq_len:
        raise ValueError(f"Error: Could not find a valid configuration for seq len {max_seq_len}; estimated max tokens with min activations to be {min_save_tokens_per_round}")

    ### cannot exceed max tokens per round determined by memory constraints
    target_tokens_per_round = min(max_tokens_per_round, target_tokens_per_round)

    ### in case we are testing and want to set a minimum threshold
    if min_tokens_per_round_limit is not None:
        target_tokens_per_round = max(min_tokens_per_round_limit, target_tokens_per_round)

    if max_tokens_per_round_limit is not None:
        target_tokens_per_round = min(max_tokens_per_round_limit, target_tokens_per_round)

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

    if verbose:
        print(f"[Working Set Log] Comparing prior tokens per round: {target_tokens_per_round} with min chunk size: {min_chunk_size} and max global batch tokens: {max_global_batch_tokens}", flush=True)

    target_tokens_per_round = min(max_global_batch_tokens, target_tokens_per_round)

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
        print(f"[Working Set Log] Determined Initial Target Min Chunk Size Est (based on Arithmetic Intensity bound x factor of {ARITH_BOUND_FACTOR}) of: {init_target_min_chunk_size}", flush=True)

    
    if min_chunk_size is not None:
        init_target_min_chunk_size = max(min_chunk_size, init_target_min_chunk_size)

    init_chunk_size_options = sorted(get_divisors(target_tokens_per_round), reverse=True)

    if fixed_seq_len:
        max_seqs_per_round = target_tokens_per_round // max_seq_len
        chunk_size_options = [max_seq_len * i for i in range(max_seqs_per_round, 0, -1)]
        seq_len_divs = sorted(get_divisors(max_seq_len), reverse=True)
        for seq_len_div in seq_len_divs:
            chunk_size_options.append(seq_len_div)
    else:
        chunk_size_options = init_chunk_size_options

    chunk_size_options = [d for d in chunk_size_options if d >= init_target_min_chunk_size]
    
    cur_remaining_gpu_mem_bytes = remaining_gpu_mem_bytes

    best_option = None
    valid_options = []

    if verbose:
        print(f"[Working Set Log] Chunk Size Options: {chunk_size_options}", flush=True)
        print(f"[Working Set Log] Before deciding chunk size, observe remaining gpu mem bytes as: {remaining_gpu_mem_bytes}, target tokens per round: {target_tokens_per_round}", flush=True)

    for chunk_size in chunk_size_options:

        ## restart
        cur_remaining_gpu_mem_bytes = remaining_gpu_mem_bytes
        
        if max_chunk_size is not None and chunk_size > max_chunk_size:
            #print(f"[Working Set Log] Chunk size {chunk_size} exceeds max chunk size {max_chunk_size}, skipping")
            continue

        ### at this point we break and will choose first valid chunk size
        if (min_chunk_size is not None and chunk_size < min_chunk_size):
            break

        target_num_chunks = target_tokens_per_round // chunk_size

        temp_target_tokens_per_round = target_num_chunks * chunk_size
        final_round_tokens = max_global_batch_tokens % temp_target_tokens_per_round


        ### if the last round will be too small and cause extra overhead, choose different chunk size
        if final_round_tokens > 0 and final_round_tokens < 0.4 * compute_lim_tokens_per_round:
            continue

        ### this includes transition table, context window, and activation workspace
        baseline_act_gpu_memory = get_baseline_gpu_activation_memory_requirements(model_dims, max_seq_len, chunk_size, target_num_chunks, training_config=training_config)

        n_gpu_layers = 1
        n_gpu_grad_layers = 1

        cur_remaining_gpu_mem_bytes -= baseline_act_gpu_memory

    
        ### first try to fill up the 1st layer worth of act slots
        full_act_slot_size_bytes = get_full_act_slot_size_bytes(model_dims, chunk_size)

        ### need at least 1 act slot
        if cur_remaining_gpu_mem_bytes < full_act_slot_size_bytes:
            continue 

        gpu_act_workspace_size_bytes = full_act_slot_size_bytes

        cur_remaining_gpu_mem_bytes -= gpu_act_workspace_size_bytes

        ### need to have at least 1 layer of optimizer as part of act workspace
        if gpu_act_workspace_size_bytes < backbone_sizes["opt_bytes"]:
            extra_opt_bytes = backbone_sizes["opt_bytes"] - gpu_act_workspace_size_bytes
            ### not enough memory for 1 layer of optimizer
            if cur_remaining_gpu_mem_bytes < extra_opt_bytes:
                continue
            gpu_act_workspace_size_bytes += extra_opt_bytes
            cur_remaining_gpu_mem_bytes -= extra_opt_bytes

        
        ### now first prioritize getting 2 act slots, then 2 layers/gradients, then complete layers...

        temp_act_slots = gpu_act_workspace_size_bytes // full_act_slot_size_bytes
        if temp_act_slots < 2:
            ## use space for 2nd act slot
            if cur_remaining_gpu_mem_bytes >= full_act_slot_size_bytes:
                gpu_act_workspace_size_bytes += full_act_slot_size_bytes
                cur_remaining_gpu_mem_bytes -= full_act_slot_size_bytes
        
        ## now prioritize second layer weights
        if cur_remaining_gpu_mem_bytes >= backbone_sizes["weight_bytes"]:
            n_gpu_layers = 2
            cur_remaining_gpu_mem_bytes -= backbone_sizes["weight_bytes"]

        ## now prioritize second layer gradients
        if cur_remaining_gpu_mem_bytes >= backbone_sizes["grad_bytes"]:
            n_gpu_grad_layers = 2
            cur_remaining_gpu_mem_bytes -= backbone_sizes["grad_bytes"]


        ### now determine how many complete model layers we should have
        ### first fill up first layer of act slots, then fill up second layer, then apply remaining memory to complete layers

        ### fill first layer of act slots
        temp_act_slots = gpu_act_workspace_size_bytes // full_act_slot_size_bytes
        if temp_act_slots < target_num_chunks:

            remain_first_layer_slots = target_num_chunks - temp_act_slots
            first_layer_additional_act_bytes = remain_first_layer_slots * full_act_slot_size_bytes
            ### can only assign partial act slots to first layer and we are done with assignments
            if cur_remaining_gpu_mem_bytes < first_layer_additional_act_bytes:
                remain_slots = cur_remaining_gpu_mem_bytes // full_act_slot_size_bytes
                gpu_act_workspace_size_bytes += remain_slots * full_act_slot_size_bytes
                cur_remaining_gpu_mem_bytes -= remain_slots * full_act_slot_size_bytes
            else:
                ### can assign first full layer
                gpu_act_workspace_size_bytes += first_layer_additional_act_bytes
                cur_remaining_gpu_mem_bytes -= first_layer_additional_act_bytes

        temp_act_slots = gpu_act_workspace_size_bytes // full_act_slot_size_bytes
        ### fill second layer of act slots
        if temp_act_slots < 2 * target_num_chunks:
            remain_second_layer_slots = 2 * target_num_chunks - temp_act_slots
            second_layer_additional_act_bytes = remain_second_layer_slots * full_act_slot_size_bytes
            ### can only assign partial act slots to second layer and we are done with assignments
            if cur_remaining_gpu_mem_bytes < second_layer_additional_act_bytes:
                remain_slots = cur_remaining_gpu_mem_bytes // full_act_slot_size_bytes
                gpu_act_workspace_size_bytes += remain_slots * full_act_slot_size_bytes
                cur_remaining_gpu_mem_bytes -= remain_slots * full_act_slot_size_bytes
            else:
                ### can assign second full layer
                gpu_act_workspace_size_bytes += second_layer_additional_act_bytes
                cur_remaining_gpu_mem_bytes -= second_layer_additional_act_bytes
    
        ### At this point we can equally divide remaining GPU memory to know how many complete
        ### layers (weights + grad + activations) we can store...
        ### chunk size may increase context window size + be a factor of addition memory workspace
        ### activation size scales linearly with total number of tokens regardless of chunking (chunking impacts baseline act workspace)
        additional_full_compute_layer_size_bytes = get_full_compute_layer_size_bytes(model_dims, chunk_size * target_num_chunks, backbone_sizes)

        ### this is on top of the 1 full layer we have as part of baseline
        additional_complete_layers_est = int(min(num_local_layers - 1, cur_remaining_gpu_mem_bytes // additional_full_compute_layer_size_bytes))
    
        n_gpu_layers += additional_complete_layers_est
        n_gpu_grad_layers += additional_complete_layers_est

        complete_layers_size_est = additional_complete_layers_est * additional_full_compute_layer_size_bytes
    
        leftover_post_complete_layers_bytes = cur_remaining_gpu_mem_bytes - complete_layers_size_est

        ### baseline for act workspace
        gpu_act_workspace_size_bytes += additional_complete_layers_est * get_full_act_slot_size_bytes(model_dims, chunk_size * target_num_chunks)
    
        
        ### if we have less than 2 layers/gradients give priority to that, otherwise prioritize act workspace
        if n_gpu_layers < 2 and n_gpu_layers < num_local_layers and leftover_post_complete_layers_bytes >= backbone_sizes["weight_bytes"]:
            n_gpu_layers += 1
            leftover_post_complete_layers_bytes -= backbone_sizes["weight_bytes"]
        if n_gpu_grad_layers < 2 and n_gpu_grad_layers < num_local_layers and leftover_post_complete_layers_bytes >= backbone_sizes["grad_bytes"]:
            n_gpu_grad_layers += 1
            leftover_post_complete_layers_bytes -= backbone_sizes["grad_bytes"]
        gpu_act_workspace_size_bytes += leftover_post_complete_layers_bytes

        n_gpu_opt_layers = int(min(num_local_layers, gpu_act_workspace_size_bytes // backbone_sizes["opt_bytes"]))

        total_act_slots = int(target_num_chunks * num_local_layers)

        gpu_act_slots = int(min(total_act_slots, gpu_act_workspace_size_bytes // full_act_slot_size_bytes))

        saved_act_sizes = get_transformer_saved_act_sizes(model_dims, chunk_size)
        min_act_slot_size_bytes = saved_act_sizes[0]

        if remaining_host_mem_bytes < min_act_slot_size_bytes * (total_act_slots - gpu_act_slots):
            continue

        # if verbose:
        #     print(f"[Working Set Log] Determined Target Chunk Size: {chunk_size}, Target Num Chunks: {target_num_chunks}")
        #     print(f"[Working Set Log] Determined Complete Compute Layers (weights + grad + act slots): {additional_complete_layers_est + 1}")

        option = {"target_chunk_size": chunk_size, "target_num_chunks": target_num_chunks, "n_gpu_layers": n_gpu_layers, "n_gpu_grad_layers": n_gpu_grad_layers, "n_gpu_opt_layers": n_gpu_opt_layers, "gpu_act_workspace_size_bytes": gpu_act_workspace_size_bytes, "gpu_act_slots": gpu_act_slots, "total_act_slots": total_act_slots, "act_slot_size_bytes": full_act_slot_size_bytes}
        valid_options.append(option)

        ### this means a total of 2 complete layers
        if additional_complete_layers_est >= 1:
            best_option = option
            break

        if best_option is None:
            best_option = option

        if best_option["gpu_act_slots"] == 1 and option["gpu_act_slots"] > 1:
            best_option = option

        if best_option["gpu_act_slots"] < best_option["target_num_chunks"] and option["gpu_act_slots"] >= option["target_num_chunks"]:
            best_option = option
        
        if best_option["n_gpu_layers"] == 1 and option["n_gpu_layers"] > 1:
            best_option = option

        if best_option["n_gpu_grad_layers"] == 1 and option["n_gpu_grad_layers"] > 1:
            best_option = option

        if best_option["n_gpu_opt_layers"] == 1 and option["n_gpu_opt_layers"] > 1:
            best_option = option


    if best_option is None:
        raise ValueError("Error: Not enough GPU memory to fit any valid chunk size large enough to fit at least 1 additional complete layer")
    
    if verbose:
        print(f"[Working Set Log] Selected Best Option: {best_option}", flush=True)


    target_chunk_size = best_option["target_chunk_size"]
    target_num_chunks = best_option["target_num_chunks"]
    n_gpu_layers = best_option["n_gpu_layers"]
    n_gpu_grad_layers = best_option["n_gpu_grad_layers"]
    n_gpu_opt_layers = best_option["n_gpu_opt_layers"]
    gpu_act_workspace_size_bytes = best_option["gpu_act_workspace_size_bytes"]
    total_act_slots = best_option["total_act_slots"]
    gpu_act_slots = best_option["gpu_act_slots"]

    full_act_slot_size_bytes = get_full_act_slot_size_bytes(model_dims, target_chunk_size)

    gpu_act_buffer_size_bytes = gpu_act_workspace_size_bytes
    
    endpoint_bytes = endpoint_sizes["embed_bytes"] + endpoint_sizes["head_bytes"]

    ### recompute with chosen values

    baseline_act_gpu_memory = get_baseline_gpu_activation_memory_requirements(model_dims, max_seq_len, target_chunk_size, target_num_chunks, training_config=training_config)

    est_total_gpu_bytes = baseline_act_gpu_memory + gpu_act_workspace_size_bytes + backbone_sizes["weight_bytes"] * n_gpu_layers + backbone_sizes["grad_bytes"] * n_gpu_grad_layers + endpoint_bytes

    assert est_total_gpu_bytes <= max_gpu_mem_bytes

    ## Now ensure we have enough host memory for minimal amount of activations

    host_act_slots = total_act_slots - gpu_act_slots
    
    ### Will not need more than this amount of host memory
    max_host_act_buffer_size_bytes = host_act_slots * full_act_slot_size_bytes

    host_act_buffer_size_bytes = min(max_host_act_buffer_size_bytes, remaining_host_mem_bytes)

    est_total_host_bytes = host_act_buffer_size_bytes + baseline_host_bytes

    saved_act_sizes = get_transformer_saved_act_sizes(model_dims, target_chunk_size)
    min_act_slot_size_bytes = saved_act_sizes[0]

    
    if verbose:
        print(f"[Working Set Log] Determined Target Max Chunk Size of {target_chunk_size}, Target Tokens Per Round of {target_tokens_per_round}\n\tAct Slot Size: {full_act_slot_size_bytes / (1 << 20):.2f}MiB\n\t# GPU Full Act Slots: {gpu_act_slots}\n\t# Host Act Slots: {host_act_slots}\n\t# GPU Act Buffer Size: {gpu_act_buffer_size_bytes / (1 << 30):.2f}GiB\n\t# Host Act Buffer Size: {host_act_buffer_size_bytes / (1 << 30):.2f}GiB", flush=True)
        print(f"[Working Set Log] Expected GPU Memory Usage: {est_total_gpu_bytes / (1 << 30):.2f}GiB, Expected Host Memory Usage: {est_total_host_bytes / (1 << 30):.2f}GiB", flush=True)

    
    min_host_act_buffer_size_bytes = host_act_slots * min_act_slot_size_bytes
    
    assert host_act_buffer_size_bytes >= min_host_act_buffer_size_bytes
    
    assert est_total_host_bytes <= max_host_mem_bytes

    ### based on selected chunk size and num chunks
    target_tokens_per_round = target_chunk_size * target_num_chunks

    working_set_config = {
        "available_gpu_memory_bytes": available_gpu_memory_capacity_bytes,
        "available_host_memory_bytes": available_host_memory_capacity_bytes,
        "leeway_gpu_memory_bytes": leeway_gpu_mem_bytes,
        "leeway_host_memory_bytes": leeway_host_mem_bytes,
        "n_gpu_layers": min(n_gpu_layers, num_local_layers),
        "n_gpu_grads": min(n_gpu_grad_layers, num_local_layers),
        "n_gpu_opt_layers": min(n_gpu_opt_layers, num_local_layers),
        "max_training_chunks": target_num_chunks,
        "max_chunk_size": target_chunk_size,
        "max_seq_len": max_seq_len,
        "target_round_tokens": target_chunk_size * target_num_chunks,
        "target_num_rounds": math.ceil(max_global_batch_tokens /(target_chunk_size * target_num_chunks)),
        "max_total_round_tokens": max_tokens_per_round,
        "host_act_buffer_size": int(host_act_buffer_size_bytes),
        "gpu_act_buffer_size": int(gpu_act_buffer_size_bytes),
        "max_host_mem_gb": max_host_mem_bytes / (1 << 30),
        "max_gpu_mem_gb": max_gpu_mem_bytes / (1 << 30),
    }

    if verbose:
        print("[Working Set Log] Running Hardware Environment Check to Return Estimated Hardware Environment...", flush=True)

    chosen_hardware_env = get_hardware_env(target_chunk_size, model_dims, device_id=device_id)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return working_set_config, chosen_hardware_env

