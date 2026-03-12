import torch

from .bench_matmul import bench_matmul
from .bench_transfer import bench_transfer
from .utils import get_backbone_layer_size_bytes, get_torch_dtype
from .query_memory import get_available_gpu_memory, get_available_host_memory

def get_transformer_transfer_report(chunk_size, model_dims, device_id=0, to_pin=True, n_reps=100):

    transfer_report = {
        "chunk_size": chunk_size,
        "model_dims": model_dims,
        "device_id": device_id,
        "to_pin": to_pin,
        "layer_num_bytes": 0,
        "layer_inbound_transfer_duration_sec": 0,
        "layer_outbound_transfer_duration_sec": 0,
        "layer_concurrent_transfer_duration_sec": 0,
        "estimate_cpu_to_gpu_bandwidth_gb_per_sec": 0,
        "estimate_gpu_to_cpu_bandwidth_gb_per_sec": 0,
        "overall_unidirectional_concurrent_bandwidth_gb_per_sec": 0,
    }
    

    layer_num_bytes = get_backbone_layer_size_bytes(model_dims)

    transfer_report["layer_num_bytes"] = layer_num_bytes

    inbound_avg_duration_sec, inbound_throughput_bytes_per_sec = bench_transfer(layer_num_bytes, src="cpu", dst=f"cuda:{device_id}", to_pin=True, n_reps=n_reps)

    transfer_report["layer_inbound_transfer_duration_sec"] = inbound_avg_duration_sec
    transfer_report["estimate_cpu_to_gpu_bandwidth_gb_per_sec"] = inbound_throughput_bytes_per_sec / 1e9

    outbound_avg_duration_sec, outbound_throughput_bytes_per_sec = bench_transfer(layer_num_bytes, src=f"cuda:{device_id}", dst="cpu", to_pin=True, n_reps=n_reps)

    transfer_report["layer_outbound_transfer_duration_sec"] = outbound_avg_duration_sec
    transfer_report["estimate_gpu_to_cpu_bandwidth_gb_per_sec"] = outbound_throughput_bytes_per_sec / 1e9

    concurrent_avg_duration_sec, concurrent_throughput_bytes_per_sec = bench_transfer(layer_num_bytes, src="cpu", dst=f"cuda:{device_id}", to_pin=True, n_reps=n_reps, concurrent=True)

    transfer_report["layer_concurrent_transfer_duration_sec"] = concurrent_avg_duration_sec
    transfer_report["overall_unidirectional_concurrent_bandwidth_gb_per_sec"] = concurrent_throughput_bytes_per_sec / 1e9

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    return transfer_report


def get_transformer_matmul_report(chunk_size, model_dims, device_id=0, n_reps=100):

    attn_proj_dtype = get_torch_dtype(model_dims["datatypes"]["attn_proj"])
    expert_proj_dtype = get_torch_dtype(model_dims["datatypes"]["expert_proj"])
    router_dtype = get_torch_dtype(model_dims["datatypes"]["router"])
    
    ### FORMAT OF MATMUL REPORT
    matmul_report = {
        "chunk_size": chunk_size,
        "model_dims": model_dims,
        "attn_proj_dtype": attn_proj_dtype,
        "expert_proj_dtype": expert_proj_dtype,
        "router_dtype": router_dtype,
        "device_id": device_id,
        "attn": {
            "qkv_proj": {
                "duration_sec": 0,
                "throughput_tflops_per_sec": 0
            },
            "o_proj": {
                "duration_sec": 0,
                "throughput_tflops_per_sec": 0
            }
        },
        "mlp": {
            "shared_expert": {
                "up": {
                    "duration_sec": 0,
                    "throughput_tflops_per_sec": 0
                },
                "down": {
                    "duration_sec": 0,
                    "throughput_tflops_per_sec": 0
                }
            },
            "router": {
                "duration_sec": 0,
                "throughput_tflops_per_sec": 0
            },
            "avg_routed_expert": {
                "up": {
                    "duration_sec": 0,
                    "throughput_tflops_per_sec": 0
                },
                "down": {
                    "duration_sec": 0,
                    "throughput_tflops_per_sec": 0
                }
            }
        },
        "overall_layer_matmul_duration": 0,
        "overall_layer_matmul_num_flops": 0,
        "overall_layer_matmul_throughput_tflops_per_sec": 0
    }

    device = torch.device(f"cuda:{device_id}")

    torch.cuda.set_device(device)

    ## MODEL DIMS

    d_model = model_dims["d_model"]
    n_heads = model_dims["n_heads"]
    head_dim = model_dims["head_dim"]
    n_kv_heads = model_dims["n_kv_heads"]

    attn_dim = n_heads * head_dim
    ctx_dim = n_kv_heads * head_dim

    expert_dim = model_dims["expert_dim"]
    num_shared_experts = model_dims["num_shared_experts"]
    num_routed_experts = model_dims["num_routed_experts"]
    top_k = model_dims["top_k"]

    overall_layer_matmul_duration_sec = 0
    overall_layer_matmul_num_flops = 0

    ### Q,K,V proj

    

    X = torch.randn(chunk_size, d_model, dtype=attn_proj_dtype, device=device)
    W_qkv = torch.randn(d_model, attn_dim + 2 * ctx_dim, dtype=attn_proj_dtype, device=device)


    qkv_matmul_duration_sec, qkv_throughput_flops_per_sec = bench_matmul(X, W_qkv, n_reps=n_reps)

    matmul_report["attn"]["qkv_proj"]["duration_sec"] = qkv_matmul_duration_sec
    matmul_report["attn"]["qkv_proj"]["throughput_tflops_per_sec"] = (qkv_throughput_flops_per_sec) / 1e12

    overall_layer_matmul_duration_sec += qkv_matmul_duration_sec
    overall_layer_matmul_num_flops += 2 * chunk_size * d_model * (attn_dim + 2 * ctx_dim)

    ### o PROJ

    X_attn = torch.randn(chunk_size, attn_dim, dtype=attn_proj_dtype, device=device)

    W_o = torch.randn(attn_dim, d_model, dtype=attn_proj_dtype, device=device)

    resid = torch.randn(chunk_size, d_model, dtype=attn_proj_dtype, device=device)

    o_matmul_duration_sec, o_throughput_flops_per_sec = bench_matmul(X_attn, W_o, C=resid, D=resid, alpha=1.0, beta=1.0, n_reps=n_reps)

    matmul_report["attn"]["o_proj"]["duration_sec"] = o_matmul_duration_sec
    matmul_report["attn"]["o_proj"]["throughput_tflops_per_sec"] = (o_throughput_flops_per_sec) / 1e12

    overall_layer_matmul_duration_sec += o_matmul_duration_sec
    overall_layer_matmul_num_flops += 2 * chunk_size * attn_dim * d_model

    ## expert proj

    expert_dim = model_dims["expert_dim"]

    W_up = torch.randn(d_model, 2 * expert_dim, dtype=expert_proj_dtype, device=device)
    W_down = torch.randn(expert_dim, d_model, dtype=expert_proj_dtype, device=device)

    if num_shared_experts > 0:

        shared_expert_up_matmul_duration_sec, shared_expert_up_throughput_flops_per_sec = bench_matmul(X, W_up, n_reps=n_reps)

        matmul_report["mlp"]["shared_expert"]["up"]["duration_sec"] = shared_expert_up_matmul_duration_sec
        matmul_report["mlp"]["shared_expert"]["up"]["throughput_tflops_per_sec"] = (shared_expert_up_throughput_flops_per_sec) / 1e12

        X_act = torch.randn(chunk_size, expert_dim, dtype=expert_proj_dtype, device=device)

        shared_expert_down_matmul_duration_sec, shared_expert_down_throughput_flops_per_sec = bench_matmul(X_act, W_down, C=resid, D=resid, alpha=1.0, beta=1.0, n_reps=n_reps)

        matmul_report["mlp"]["shared_expert"]["down"]["duration_sec"] = shared_expert_down_matmul_duration_sec
        matmul_report["mlp"]["shared_expert"]["down"]["throughput_tflops_per_sec"] = (shared_expert_down_throughput_flops_per_sec) / 1e12

        overall_layer_matmul_duration_sec += shared_expert_up_matmul_duration_sec + shared_expert_down_matmul_duration_sec
        overall_layer_matmul_num_flops += 2 * chunk_size * 3 * expert_dim * d_model

    ## rotued_expert_proj

    if num_routed_experts > 0:

        ### router

        W_router = torch.randn(d_model, num_routed_experts, dtype=router_dtype, device=device)

        router_matmul_duration_sec, router_throughput_flops_per_sec = bench_matmul(X, W_router, n_reps=n_reps)

        matmul_report["mlp"]["router"]["duration_sec"] = router_matmul_duration_sec
        matmul_report["mlp"]["router"]["throughput_tflops_per_sec"] = (router_throughput_flops_per_sec) / 1e12

        overall_layer_matmul_duration_sec += router_matmul_duration_sec
        overall_layer_matmul_num_flops += 2 *chunk_size * d_model * num_routed_experts

        ### avg routed expert

        avg_tokens_per_expert = int((chunk_size * top_k) / num_routed_experts)

        X_avg_exp_input = torch.randn(avg_tokens_per_expert, d_model, dtype=expert_proj_dtype, device=device)

        X_avg_exp_up_matmul_duration_sec, X_avg_exp_up_throughput_flops_per_sec = bench_matmul(X_avg_exp_input, W_up, n_reps=n_reps)

        matmul_report["mlp"]["avg_routed_expert"]["up"]["duration_sec"] = X_avg_exp_up_matmul_duration_sec
        matmul_report["mlp"]["avg_routed_expert"]["up"]["throughput_tflops_per_sec"] = (X_avg_exp_up_throughput_flops_per_sec) / 1e12

        X_avg_exp_act = torch.randn(avg_tokens_per_expert, expert_dim, dtype=expert_proj_dtype, device=device)

        X_avg_exp_down_matmul_duration_sec, X_avg_exp_down_throughput_flops_per_sec = bench_matmul(X_avg_exp_act, W_down, n_reps=n_reps)

        matmul_report["mlp"]["avg_routed_expert"]["down"]["duration_sec"] = X_avg_exp_down_matmul_duration_sec
        matmul_report["mlp"]["avg_routed_expert"]["down"]["throughput_tflops_per_sec"] = (X_avg_exp_down_throughput_flops_per_sec) / 1e12

        overall_layer_matmul_duration_sec += num_routed_experts * (X_avg_exp_down_matmul_duration_sec + X_avg_exp_up_matmul_duration_sec)
        overall_layer_matmul_num_flops += 2 * chunk_size * top_k * 3 * expert_dim * d_model
    
    matmul_report["overall_layer_matmul_duration"] = overall_layer_matmul_duration_sec
    matmul_report["overall_layer_matmul_num_flops"] = overall_layer_matmul_num_flops
    matmul_report["overall_layer_matmul_throughput_tflops_per_sec"] = (overall_layer_matmul_num_flops) / (1e12 * overall_layer_matmul_duration_sec)

    torch.cuda.synchronize()

    del X, W_qkv, X_attn, W_o, resid, W_up, W_down

    if num_shared_experts > 0:
        del X_act

    if num_routed_experts > 0:
        del W_router, X_avg_exp_input, X_avg_exp_act

    torch.cuda.empty_cache()

    return matmul_report

def get_basic_peak_flops_est(n=8192, dtype=torch.bfloat16, device="cuda:0"):

    X = torch.randn(n, n, dtype=dtype, device=device)
    W = torch.randn(n, n, dtype=dtype, device=device)

    matmul_duration_sec, matmul_throughput_flops_per_sec = bench_matmul(X, W, n_reps=1000)

    torch.cuda.synchronize()

    del X, W

    torch.cuda.empty_cache()

    return matmul_throughput_flops_per_sec

def get_basic_peak_mem_bandwidth_gb_per_sec(n=16384, dtype=torch.bfloat16, device="cuda:0"):
    
    X = torch.randn(1, n, dtype=dtype, device=device)
    W = torch.randn(n, n, dtype=dtype, device=device)

    matmul_duration_sec, matmul_throughput_flops_per_sec = bench_matmul(X, W, n_reps=1000)

    torch.cuda.synchronize()

    del X, W

    bytes_touched = dtype.itemsize * (n * n + 2 * n)

    est_peak_mem_bandwidth_gb_per_sec = bytes_touched / matmul_duration_sec / 1e9 

    torch.cuda.empty_cache()

    return est_peak_mem_bandwidth_gb_per_sec
    


def get_hardware_env(chunk_size, model_dims, device_id=0, to_pin=True, n_transfer_reps=10, n_matmul_reps=100):

    transfer_report = get_transformer_transfer_report(chunk_size, model_dims, device_id=device_id, to_pin=to_pin, n_reps=n_transfer_reps)

    if chunk_size is None:
        matmul_report = None
    else:
        matmul_report = get_transformer_matmul_report(chunk_size, model_dims, device_id=device_id, n_reps=n_matmul_reps)
    
    return {
        "available_gpu_memory_capacity": get_available_gpu_memory(device_id),
        "available_host_memory_capacity": get_available_host_memory(),
        "transfer_report": transfer_report,
        "matmul_report": matmul_report,
        "basic_peak_tflops_est": get_basic_peak_flops_est() / 1e12,
        "basic_peak_mem_bandwidth_gb_per_sec": get_basic_peak_mem_bandwidth_gb_per_sec()
    }
        
