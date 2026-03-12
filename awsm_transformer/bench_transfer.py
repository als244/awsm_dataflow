import time
import torch

def bench_transfer(num_bytes, src="cpu", dst="cuda:0", to_pin=True, n_reps=100, concurrent=False):


    if src == "cpu" and to_pin:
        to_pin_src = True
    else:
        to_pin_src = False

    src_tensor = torch.randint(256, (num_bytes,), dtype=torch.uint8, device=src, pin_memory=to_pin_src)

    if dst == "cpu" and to_pin:
        to_pin_dst = True
    else:
        to_pin_dst = False

    dst_tensor = torch.randint(256, (num_bytes,), dtype=torch.uint8, device=dst, pin_memory=to_pin_dst)


    primary_stream = torch.cuda.Stream()

    if concurrent:    
        secondary_stream = torch.cuda.Stream()
        secondary_src = dst
        secondary_src_to_pin = to_pin_dst
        secondary_dst = src
        secondary_dst_to_pin = to_pin_src
        secondary_src_tensor = torch.randint(256, (num_bytes,), dtype=torch.uint8, device=secondary_src, pin_memory=secondary_src_to_pin)
        secondary_dst_tensor = torch.randint(256, (num_bytes,), dtype=torch.uint8, device=secondary_dst, pin_memory=secondary_dst_to_pin)
    
    
    torch.cuda.synchronize()

    durations_sec = []

    for i in range(n_reps):

        start_time = time.perf_counter_ns()

        with primary_stream:
            dst_tensor.copy_(src_tensor, non_blocking=True)

        if concurrent:

            with secondary_stream:
                secondary_dst_tensor.copy_(secondary_src_tensor, non_blocking=True)

            primary_stream.wait_stream(secondary_stream)
        
        primary_stream.synchronize()

        end_time = time.perf_counter_ns()

        durations_sec.append((end_time - start_time) / 1e9)

    avg_duration_sec = sum(durations_sec) / len(durations_sec)
    
    throughput_bytes_per_sec = num_bytes / avg_duration_sec

    torch.cuda.synchronize()

    del src_tensor
    del dst_tensor

    if concurrent:
        del secondary_src_tensor
        del secondary_dst_tensor

    return avg_duration_sec, throughput_bytes_per_sec

        

    