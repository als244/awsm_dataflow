"""
Test harness: compare FlashAttentionHelper (C library wrapper) against the
original awsm_attention Python implementation (flash_attn_2_cuda / flash_attn_3_cuda).

Usage:
    python test_compare.py
    python test_compare.py --dtype fp16
    python test_compare.py --no-backward
    python test_compare.py --atol 1e-2 --rtol 1e-2
"""

import argparse
import sys
import torch

from awsm_attention import FlashAttentionHelper

# Import the reference implementation
from orig_attention import awsm_attention_fwd, awsm_attention_bwd


def make_test_data(
    seq_lens: list[int],
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Build q, k, v, offsets, lens tensors for a batch of variable-length sequences."""
    total_tokens = sum(seq_lens)
    num_seqs = len(seq_lens)
    max_seqlen = max(seq_lens)

    # Cumulative offsets with leading 0
    offsets = [0]
    for s in seq_lens:
        offsets.append(offsets[-1] + s)

    q_seq_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    k_seq_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    q_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    k_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    # Use same random seed for reproducibility
    torch.manual_seed(42)
    q = torch.randn(total_tokens, n_q_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)

    return q, k, v, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen


def compare_tensors(name: str, ref: torch.Tensor, test: torch.Tensor, atol: float, rtol: float) -> bool:
    """Compare two tensors, print stats, return True if close enough."""
    if ref.shape != test.shape:
        print(f"  FAIL {name}: shape mismatch ref={ref.shape} vs test={test.shape}")
        return False

    abs_diff = (ref.float() - test.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Use torch.allclose logic
    close = torch.allclose(ref.float(), test.float(), atol=atol, rtol=rtol)

    status = "PASS" if close else "FAIL"
    print(f"  {status} {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")

    if not close:
        # Show where the biggest differences are
        rel_diff = abs_diff / (ref.float().abs() + 1e-8)
        print(f"         max_rel_diff={rel_diff.max().item():.6e}, "
              f"frac_exceeding_atol={((abs_diff > atol).float().mean().item()):.4%}")

    return close


def run_test(
    seq_lens: list[int],
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    causal: bool,
    test_backward: bool,
    atol: float,
    rtol: float,
    helper: FlashAttentionHelper,
) -> bool:
    """Run a single test case. Returns True if all checks pass."""
    total_tokens = sum(seq_lens)
    dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16"}[dtype]

    print(f"\n{'='*70}")
    print(f"Test: seqs={seq_lens}, heads={n_q_heads}/{n_kv_heads}, "
          f"dim={head_dim}, {dtype_str}, causal={causal}")
    print(f"{'='*70}")

    q, k, v, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen = \
        make_test_data(seq_lens, n_q_heads, n_kv_heads, head_dim, dtype, device)

    all_passed = True

    # ---- Forward: Reference (awsm_flash_attn) ----
    # NOTE: flash2_gpu.varlen_fwd returns softmax_lse in padded layout
    # (n_q_heads, max_seqlen), while the C wrapper uses unpadded layout
    # (total_tokens, n_q_heads). We let the reference use its own LSE
    # allocation and only compare the attention output.
    ref_out = torch.empty(total_tokens, n_q_heads, head_dim, dtype=dtype, device=device)
    # flash2's mha_varlen_fwd returns softmax_lse as (num_heads, total_q)
    # with unpadded_lse=true. The C wrapper uses (total_q, num_heads).
    ref_lse = torch.empty(n_q_heads, total_tokens, dtype=torch.float32, device=device)

    try:
        awsm_attention_fwd(
            q, k, v, ref_out, ref_lse,
            q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens,
            max_seqlen, max_seqlen,
            causal=causal,
        )
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(f"Reference (awsm_flash_attn) forward failed: {e}") from e

    # ---- Forward: C Library Wrapper (FlashAttentionHelper) ----
    test_out = torch.empty(total_tokens, n_q_heads, head_dim, dtype=dtype, device=device)
    # C wrapper also writes LSE as (num_heads, total_q) — same as flash2 varlen
    test_lse = torch.empty(n_q_heads, total_tokens, dtype=torch.float32, device=device)

    try:
        helper.forward(
            q, k, v, test_out, test_lse,
            q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens,
            max_seqlen, max_seqlen,
            causal=causal,
        )
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(f"C wrapper (FlashAttentionHelper) forward failed: {e}") from e

    print("Forward:")
    all_passed &= compare_tensors("out", ref_out, test_out, atol, rtol)
    all_passed &= compare_tensors("softmax_lse", ref_lse, test_lse, atol, rtol)

    if not test_backward:
        return all_passed

    # ---- Backward: Reference (awsm_flash_attn) ----
    # Use ref_out and ref_lse (in reference layout) for reference backward
    dout = torch.randn_like(ref_out)

    ref_dq = torch.zeros_like(q)
    ref_dk = torch.zeros_like(k)
    ref_dv = torch.zeros_like(v)

    try:
        awsm_attention_bwd(
            dout, q, k, v, ref_out, ref_lse,
            ref_dq, ref_dk, ref_dv,
            q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens,
            max_seqlen, max_seqlen,
            causal=causal,
            deterministic=True,
        )
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(f"Reference (awsm_flash_attn) backward failed: {e}") from e

    # ---- Backward: C Library Wrapper (FlashAttentionHelper) ----
    # Use test_out and test_lse (in C wrapper layout) for C wrapper backward
    test_dq = torch.zeros_like(q)
    test_dk = torch.zeros_like(k)
    test_dv = torch.zeros_like(v)

    try:
        helper.backward(
            dout, q, k, v, test_out, test_lse,
            test_dq, test_dk, test_dv,
            q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens,
            max_seqlen, max_seqlen,
            causal=causal,
        )
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(f"C wrapper (FlashAttentionHelper) backward failed: {e}") from e

    print("Backward:")
    all_passed &= compare_tensors("dq", ref_dq, test_dq, atol, rtol)
    all_passed &= compare_tensors("dk", ref_dk, test_dk, atol, rtol)
    all_passed &= compare_tensors("dv", ref_dv, test_dv, atol, rtol)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Compare FlashAttentionHelper vs reference implementation")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="Data type (default: bf16)")
    parser.add_argument("--no-backward", action="store_true", help="Skip backward pass tests")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance (default: 1e-2)")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance (default: 1e-2)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    helper = FlashAttentionHelper(device=device)
    print(f"GPU: {torch.cuda.get_device_name(device)}, arch=SM{helper.arch}, sm_count={helper.sm_count}")
    print(f"dtype={args.dtype}, atol={args.atol}, rtol={args.rtol}, backward={not args.no_backward}")

    # Define test cases: (seq_lens, n_q_heads, n_kv_heads, head_dim, causal)
    test_cases = [
        # Basic: single sequence
        ([128],         32, 32, 128, True),
        ([256],         32, 32, 128, False),

        # GQA: num_q_heads != num_kv_heads
        ([128],         32, 8,  128, True),
        ([256],         32, 8,  128, True),

        # Multiple sequences (variable length)
        ([128, 256],    32, 32, 128, True),
        ([128, 256],    32, 8,  128, True),

        # Longer sequences
        ([512, 1024],   32, 8,  128, True),

        # Many short sequences
        ([64, 64, 64, 64], 32, 8, 128, True),

        # Different head dims
        ([128, 256],    32, 8,  64,  True),

        # Non-causal
        ([128, 256],    32, 8,  128, False),

        # Single token sequences (edge case)
        ([1, 1, 1],     32, 8,  128, True),

        # Mixed short and long
        ([16, 512],     32, 8,  128, True),
    ]

    num_passed = 0
    num_failed = 0
    failed_tests = []

    for seq_lens, nqh, nkvh, hdim, causal in test_cases:
        try:
            passed = run_test(
                seq_lens=seq_lens,
                n_q_heads=nqh,
                n_kv_heads=nkvh,
                head_dim=hdim,
                dtype=dtype,
                device=device,
                causal=causal,
                test_backward=not args.no_backward,
                atol=args.atol,
                rtol=args.rtol,
                helper=helper,
            )
            if passed:
                num_passed += 1
            else:
                num_failed += 1
                failed_tests.append((seq_lens, nqh, nkvh, hdim, causal))
        except Exception as e:
            num_failed += 1
            failed_tests.append((seq_lens, nqh, nkvh, hdim, causal))
            import traceback
            print(f"  EXCEPTION ({type(e).__name__}): {e}")
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {num_passed} passed, {num_failed} failed out of {num_passed + num_failed} tests")
    print(f"{'='*70}")

    if failed_tests:
        print("\nFailed tests:")
        for seq_lens, nqh, nkvh, hdim, causal in failed_tests:
            print(f"  seqs={seq_lens}, heads={nqh}/{nkvh}, dim={hdim}, causal={causal}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()