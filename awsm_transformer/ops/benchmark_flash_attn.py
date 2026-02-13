#!/usr/bin/env python3
"""
Benchmark script for Flash Attention varlen forward and backward passes.

Usage:
    python benchmark_flash_attn.py /path/to/example_attn_data --repeats 100
"""

import argparse
import os
import time
from pathlib import Path

import torch

# Import flash attention varlen function
from flash_attn import flash_attn_varlen_func


def calculate_flops(cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, 
                    num_heads: int, head_dim: int, causal: bool) -> dict:
    """Calculate FLOPs for forward and backward passes.
    
    Forward: sum(4 * causal_factor * s_q_i * s_k_i * n_q_heads * head_dim)
    Backward: sum(8 * causal_factor * s_q_i * s_k_i * n_q_heads * head_dim)
    
    For self-attention (cu_seqlens_q == cu_seqlens_k), s_q_i == s_k_i == s_i
    """
    causal_factor = 0.5 if causal else 1.0
    
    # Calculate sequence lengths from cumulative offsets
    seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).cpu()
    seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).cpu()
    
    # Sum of s_q_i * s_k_i across all sequences
    seq_products_sum = (seqlens_q.float() * seqlens_k.float()).sum().item()
    
    # FLOPs calculation
    fwd_flops = 4 * causal_factor * seq_products_sum * num_heads * head_dim
    bwd_flops = 8 * causal_factor * seq_products_sum * num_heads * head_dim
    fwd_bwd_flops = fwd_flops + bwd_flops
    
    return {
        'forward': fwd_flops,
        'backward': bwd_flops,
        'fwd_bwd': fwd_bwd_flops,
    }


def flops_to_tflops(flops: float, time_ms: float) -> float:
    """Convert FLOPs and time to TFLOPS."""
    time_s = time_ms / 1000.0
    return (flops / time_s) / 1e12


def load_inputs(directory: str) -> dict:
    """Load all input tensors from the specified directory."""
    dir_path = Path(directory)
    
    inputs = {}
    required_files = ['q.pt', 'k.pt', 'v.pt', 'q_seq_offsets.pt', 'k_seq_offsets.pt']
    optional_files = ['dout.pt', 'q_seq_lens.pt', 'k_seq_lens.pt']
    
    # Load required files
    for fname in required_files:
        fpath = dir_path / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Required file not found: {fpath}")
        inputs[fname.replace('.pt', '')] = torch.load(fpath, weights_only=True)
    
    # Load optional files
    for fname in optional_files:
        fpath = dir_path / fname
        if fpath.exists():
            inputs[fname.replace('.pt', '')] = torch.load(fpath, weights_only=True)
    
    return inputs


def derive_max_seqlens(inputs: dict) -> tuple[int, int]:
    """Derive max_seqlen_q and max_seqlen_k from sequence lengths or offsets."""
    
    # Try using seq_lens arrays first
    if 'q_seq_lens' in inputs:
        max_seqlen_q = inputs['q_seq_lens'].max().item()
    else:
        # Derive from offsets: seqlens[i] = offsets[i+1] - offsets[i]
        q_offsets = inputs['q_seq_offsets']
        q_seqlens = q_offsets[1:] - q_offsets[:-1]
        max_seqlen_q = q_seqlens.max().item()
    
    if 'k_seq_lens' in inputs:
        max_seqlen_k = inputs['k_seq_lens'].max().item()
    else:
        k_offsets = inputs['k_seq_offsets']
        k_seqlens = k_offsets[1:] - k_offsets[:-1]
        max_seqlen_k = k_seqlens.max().item()
    
    return int(max_seqlen_q), int(max_seqlen_k)


def benchmark_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
                      repeats: int, warmup: int = 10, causal: bool = True,
                      deterministic: bool = True) -> dict:
    """Benchmark forward pass."""
    
    # Warmup
    for _ in range(warmup):
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=0.0,
            causal=causal,
            deterministic=deterministic,
        )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    
    for i in range(repeats):
        start_events[i].record()
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=0.0,
            causal=causal,
            deterministic=deterministic,
        )
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'times': times,
    }


def benchmark_backward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       dout, repeats: int, warmup: int = 10, causal: bool = True,
                       deterministic: bool = True) -> dict:
    """Benchmark backward pass."""
    
    # Warmup
    for _ in range(warmup):
        q_grad = q.clone().requires_grad_(True)
        k_grad = k.clone().requires_grad_(True)
        v_grad = v.clone().requires_grad_(True)
        
        out = flash_attn_varlen_func(
            q_grad, k_grad, v_grad,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=0.0,
            causal=causal,
            deterministic=deterministic,
        )
        out.backward(dout)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    
    for i in range(repeats):
        q_grad = q.clone().requires_grad_(True)
        k_grad = k.clone().requires_grad_(True)
        v_grad = v.clone().requires_grad_(True)
        
        out = flash_attn_varlen_func(
            q_grad, k_grad, v_grad,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=0.0,
            causal=causal,
            deterministic=deterministic,
        )
        
        torch.cuda.synchronize()
        start_events[i].record()
        out.backward(dout)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'times': times,
    }


def benchmark_fwd_bwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                      dout, repeats: int, warmup: int = 10, causal: bool = True,
                      deterministic: bool = True) -> dict:
    """Benchmark combined forward + backward pass."""
    
    # Warmup
    for _ in range(warmup):
        q_grad = q.clone().requires_grad_(True)
        k_grad = k.clone().requires_grad_(True)
        v_grad = v.clone().requires_grad_(True)
        
        out = flash_attn_varlen_func(
            q_grad, k_grad, v_grad,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=0.0,
            causal=causal,
            deterministic=deterministic,
        )
        out.backward(dout)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    
    for i in range(repeats):
        q_grad = q.clone().requires_grad_(True)
        k_grad = k.clone().requires_grad_(True)
        v_grad = v.clone().requires_grad_(True)
        
        start_events[i].record()
        out = flash_attn_varlen_func(
            q_grad, k_grad, v_grad,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=0.0,
            causal=causal,
            deterministic=deterministic,
        )
        out.backward(dout)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'times': times,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark Flash Attention varlen forward and backward passes')
    parser.add_argument('directory', type=str, help='Directory containing input tensors')
    parser.add_argument('--repeats', type=int, default=100, help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--causal', action='store_true', default=True, help='Use causal attention (default: True)')
    parser.add_argument('--no-causal', action='store_false', dest='causal', help='Disable causal attention')
    parser.add_argument('--deterministic', action='store_true', default=True, help='Use deterministic backward (default: True)')
    parser.add_argument('--no-deterministic', action='store_false', dest='deterministic', help='Disable deterministic backward')
    args = parser.parse_args()
    
    print(f"Loading inputs from: {args.directory}")
    inputs = load_inputs(args.directory)
    
    # Extract tensors
    q = inputs['q'].cuda()
    k = inputs['k'].cuda()
    v = inputs['v'].cuda()
    cu_seqlens_q = inputs['q_seq_offsets'].cuda().to(torch.int32)
    cu_seqlens_k = inputs['k_seq_offsets'].cuda().to(torch.int32)
    
    # Derive max sequence lengths
    max_seqlen_q, max_seqlen_k = derive_max_seqlens(inputs)
    
    # Get or create dout for backward pass
    if 'dout' in inputs:
        dout = inputs['dout'].cuda()
    else:
        # Create a dummy dout with same shape as expected output
        with torch.no_grad():
            out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=0.0,
                causal=args.causal,
                deterministic=args.deterministic,
            )
        dout = torch.randn_like(out)
    
    # Print input info
    batch_size = cu_seqlens_q.shape[0] - 1
    total_q = q.shape[0]
    total_k = k.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    
    print(f"\n{'='*60}")
    print("Input Information:")
    print(f"  Batch size: {batch_size}")
    print(f"  Total Q tokens: {total_q}")
    print(f"  Total K tokens: {total_k}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Max seqlen Q: {max_seqlen_q}")
    print(f"  Max seqlen K: {max_seqlen_k}")
    print(f"  Q dtype: {q.dtype}")
    print(f"  Causal: {args.causal}")
    print(f"  Deterministic: {args.deterministic}")
    print(f"{'='*60}\n")
    
    # Calculate FLOPs
    flops = calculate_flops(cu_seqlens_q, cu_seqlens_k, num_heads, head_dim, args.causal)
    
    print(f"Running benchmarks with {args.repeats} repeats and {args.warmup} warmup iterations...")
    print()
    
    # Benchmark forward pass
    print("Benchmarking Forward Pass...")
    fwd_results = benchmark_forward(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        repeats=args.repeats, warmup=args.warmup, causal=args.causal,
        deterministic=args.deterministic
    )
    fwd_tflops = flops_to_tflops(flops['forward'], fwd_results['mean_ms'])
    print(f"  Mean: {fwd_results['mean_ms']:.3f} ms | {fwd_tflops:.2f} TFLOPS")
    print(f"  Min:  {fwd_results['min_ms']:.3f} ms | {flops_to_tflops(flops['forward'], fwd_results['min_ms']):.2f} TFLOPS")
    print(f"  Max:  {fwd_results['max_ms']:.3f} ms | {flops_to_tflops(flops['forward'], fwd_results['max_ms']):.2f} TFLOPS")
    print()
    
    # Benchmark backward pass
    print("Benchmarking Backward Pass...")
    bwd_results = benchmark_backward(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dout, repeats=args.repeats, warmup=args.warmup, causal=args.causal,
        deterministic=args.deterministic
    )
    bwd_tflops = flops_to_tflops(flops['backward'], bwd_results['mean_ms'])
    print(f"  Mean: {bwd_results['mean_ms']:.3f} ms | {bwd_tflops:.2f} TFLOPS")
    print(f"  Min:  {bwd_results['min_ms']:.3f} ms | {flops_to_tflops(flops['backward'], bwd_results['min_ms']):.2f} TFLOPS")
    print(f"  Max:  {bwd_results['max_ms']:.3f} ms | {flops_to_tflops(flops['backward'], bwd_results['max_ms']):.2f} TFLOPS")
    print()
    
    # Benchmark combined forward + backward
    print("Benchmarking Forward + Backward (combined)...")
    fwd_bwd_results = benchmark_fwd_bwd(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dout, repeats=args.repeats, warmup=args.warmup, causal=args.causal,
        deterministic=args.deterministic
    )
    fwd_bwd_tflops = flops_to_tflops(flops['fwd_bwd'], fwd_bwd_results['mean_ms'])
    print(f"  Mean: {fwd_bwd_results['mean_ms']:.3f} ms | {fwd_bwd_tflops:.2f} TFLOPS")
    print(f"  Min:  {fwd_bwd_results['min_ms']:.3f} ms | {flops_to_tflops(flops['fwd_bwd'], fwd_bwd_results['min_ms']):.2f} TFLOPS")
    print(f"  Max:  {fwd_bwd_results['max_ms']:.3f} ms | {flops_to_tflops(flops['fwd_bwd'], fwd_bwd_results['max_ms']):.2f} TFLOPS")
    print()
    
    # Summary
    print(f"{'='*60}")
    print("Summary:")
    print(f"  Forward:          {fwd_results['mean_ms']:.3f} ms | {fwd_tflops:.2f} TFLOPS")
    print(f"  Backward:         {bwd_results['mean_ms']:.3f} ms | {bwd_tflops:.2f} TFLOPS")
    print(f"  Forward+Backward: {fwd_bwd_results['mean_ms']:.3f} ms | {fwd_bwd_tflops:.2f} TFLOPS")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
