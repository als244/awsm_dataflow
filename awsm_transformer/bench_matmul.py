
import torch
import time

from .matmul_dispatchers import dispatcher


def bench_matmul(A, B, C=None, D=None, alpha=1.0, beta=0.0, n_warmup=5, n_reps=100):

    if len(A.shape) != 2 or len(B.shape) != 2:
        raise ValueError("A and B must be 2D tensors")
    if A.shape[1] != B.shape[0]:
        raise ValueError("A and B must have matching inner dimensions")

    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]

    created_D = False
    if D is None:
        created_D = True
        D = torch.empty(M, N, dtype=A.dtype, device=A.device)

    stream_obj = torch.cuda.current_stream()
    stream_ptr = stream_obj.cuda_stream

    for i in range(n_warmup):
        dispatcher.matmul(stream_ptr, A=A, B=B, C=C, D=D, alpha=alpha, beta=beta)

    torch.cuda.synchronize()

    start_time = time.perf_counter_ns()

    ### Assumes matrices large enough so cache impact doesnt mattter,
    ### submitting many in a row to simulate reality of dispatching back-to-back that avoids submission latency
    for i in range(n_reps):
        dispatcher.matmul(stream_ptr, A=A, B=B, C=C, D=D, alpha=alpha, beta=beta)

    torch.cuda.synchronize()

    end_time = time.perf_counter_ns()

    total_duration_sec = (end_time - start_time) / 1e9
    per_matmul_duration_sec = total_duration_sec / n_reps

    throughput_flops_per_sec = (2 * M * K * N) / per_matmul_duration_sec

    if created_D:
        del D

    return per_matmul_duration_sec, throughput_flops_per_sec
