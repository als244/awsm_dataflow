"""
Three-way comparison: naive attention vs flash_attn vs
C wrapper (FlashAttentionHelper).

The naive implementation uses the same dtype (bf16/fp16) as the flash
implementations for matmuls, with fp32 softmax, matching how flash attention
handles precision internally. This provides a ground-truth to judge which
flash implementation is correct when they disagree.

Usage:
    python test_three_way.py
    python test_three_way.py --dtype fp16
    python test_three_way.py --no-backward
    python test_three_way.py --atol 1e-2 --rtol 1e-2
"""

import argparse
import sys
import torch
import torch.nn.functional as F

from awsm_attention import FlashAttentionHelper

# Import flash_attn implementation
from orig_attention import awsm_attention_fwd, awsm_attention_bwd


# ---------------------------------------------------------------------------
# Naive (ground-truth) attention — runs in fp32 for accuracy
# ---------------------------------------------------------------------------

def naive_attention_fwd(
    q: torch.Tensor,             # (total_q, n_q_heads, head_dim)
    k: torch.Tensor,             # (total_k, n_kv_heads, head_dim)
    v: torch.Tensor,             # (total_k, n_kv_heads, head_dim)
    q_seq_offsets: torch.Tensor,  # (num_seqs + 1,) int32
    k_seq_offsets: torch.Tensor,  # (num_seqs + 1,) int32
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Straightforward QKV attention per-sequence.

    Uses the same dtype as inputs (bf16/fp16) for matmuls with fp32 softmax,
    matching how flash attention handles precision internally.

    Returns (out, softmax_lse) where:
      - out: (total_q, n_q_heads, head_dim) in the original dtype
      - softmax_lse: (n_q_heads, total_q) in fp32
    """
    orig_dtype = q.dtype
    device = q.device
    n_q_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    total_q = q.shape[0]
    num_seqs = q_seq_offsets.shape[0] - 1
    gqa_ratio = n_q_heads // n_kv_heads

    scale = head_dim ** -0.5

    # Work in the original dtype to match flash implementations
    out = torch.zeros(total_q, n_q_heads, head_dim, dtype=orig_dtype, device=device)
    # LSE in (n_q_heads, total_q) layout to match flash2
    softmax_lse = torch.zeros(n_q_heads, total_q, dtype=torch.float32, device=device)

    offsets_q = q_seq_offsets.cpu().tolist()
    offsets_k = k_seq_offsets.cpu().tolist()

    for i in range(num_seqs):
        q_start, q_end = offsets_q[i], offsets_q[i + 1]
        k_start, k_end = offsets_k[i], offsets_k[i + 1]
        seq_len_q = q_end - q_start
        seq_len_k = k_end - k_start

        # q_seq: (seq_len_q, n_q_heads, head_dim)
        q_seq = q[q_start:q_end]
        k_seq = k[k_start:k_end]
        v_seq = v[k_start:k_end]

        for h in range(n_q_heads):
            kv_h = h // gqa_ratio

            # q_h: (seq_len_q, head_dim), k_h: (seq_len_k, head_dim)
            q_h = q_seq[:, h, :]
            k_h = k_seq[:, kv_h, :]
            v_h = v_seq[:, kv_h, :]

            # scores: (seq_len_q, seq_len_k)
            scores = (torch.matmul(q_h, k_h.transpose(0, 1)) * scale).float()

            if causal:
                # For varlen: position within each sequence is what matters
                row_idx = torch.arange(seq_len_q, device=device).unsqueeze(1)
                col_idx = torch.arange(seq_len_k, device=device).unsqueeze(0)
                causal_mask = col_idx > row_idx
                scores.masked_fill_(causal_mask, float("-inf"))

            # logsumexp for each query position (fp32)
            lse = torch.logsumexp(scores, dim=-1)  # (seq_len_q,)
            softmax_lse[h, q_start:q_end] = lse

            # softmax in fp32, then cast back for matmul with v
            attn = torch.softmax(scores, dim=-1).to(orig_dtype)

            # out_h: (seq_len_q, head_dim)
            out_h = torch.matmul(attn, v_h)
            out[q_start:q_end, h, :] = out_h

    return out, softmax_lse


def naive_attention_bwd(
    dout: torch.Tensor,           # (total_q, n_q_heads, head_dim)
    q: torch.Tensor,              # (total_q, n_q_heads, head_dim)
    k: torch.Tensor,              # (total_k, n_kv_heads, head_dim)
    v: torch.Tensor,              # (total_k, n_kv_heads, head_dim)
    q_seq_offsets: torch.Tensor,
    k_seq_offsets: torch.Tensor,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Naive backward via autograd. Uses same dtype as inputs (bf16/fp16) with
    fp32 softmax to match flash attention's numerical behavior.
    Returns (dq, dk, dv) in original dtype.
    """
    orig_dtype = q.dtype
    device = q.device
    n_q_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    total_q = q.shape[0]
    total_k = k.shape[0]
    num_seqs = q_seq_offsets.shape[0] - 1
    gqa_ratio = n_q_heads // n_kv_heads
    scale = head_dim ** -0.5

    # Use autograd in the original dtype (bf16/fp16) to match flash behavior
    q_ag = q.detach().clone().requires_grad_(True)
    k_ag = k.detach().clone().requires_grad_(True)
    v_ag = v.detach().clone().requires_grad_(True)
    dout_t = dout.detach().clone()

    offsets_q = q_seq_offsets.cpu().tolist()
    offsets_k = k_seq_offsets.cpu().tolist()

    out = torch.zeros(total_q, n_q_heads, head_dim, dtype=orig_dtype, device=device)

    for i in range(num_seqs):
        q_start, q_end = offsets_q[i], offsets_q[i + 1]
        k_start, k_end = offsets_k[i], offsets_k[i + 1]
        seq_len_q = q_end - q_start
        seq_len_k = k_end - k_start

        q_seq = q_ag[q_start:q_end]
        k_seq = k_ag[k_start:k_end]
        v_seq = v_ag[k_start:k_end]

        for h in range(n_q_heads):
            kv_h = h // gqa_ratio

            q_h = q_seq[:, h, :]
            k_h = k_seq[:, kv_h, :]
            v_h = v_seq[:, kv_h, :]

            # QK^T in original dtype, softmax in fp32, AV in original dtype
            scores = (torch.matmul(q_h, k_h.transpose(0, 1)) * scale).float()

            if causal:
                row_idx = torch.arange(seq_len_q, device=device).unsqueeze(1)
                col_idx = torch.arange(seq_len_k, device=device).unsqueeze(0)
                causal_mask = col_idx > row_idx
                scores = scores.masked_fill(causal_mask, float("-inf"))

            attn = torch.softmax(scores, dim=-1).to(orig_dtype)
            out_h = torch.matmul(attn, v_h)
            out[q_start:q_end, h, :] = out_h

    out.backward(dout_t)

    dq = q_ag.grad
    dk = k_ag.grad
    dv = v_ag.grad

    # GQA: dk/dv are accumulated across q-head groups via autograd on
    # shared k_f/v_f slices, so shapes already match (total_k, n_kv_heads, head_dim).
    return dq, dk, dv


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

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
    offsets = [0]
    for s in seq_lens:
        offsets.append(offsets[-1] + s)

    q_seq_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    k_seq_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    q_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    k_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    torch.manual_seed(42)
    q = torch.randn(total_tokens, n_q_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)

    max_seqlen = max(seq_lens)
    return q, k, v, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen


def compare_tensors(
    name: str,
    ref: torch.Tensor,
    test: torch.Tensor,
    atol: float,
    rtol: float,
    label_ref: str = "ref",
    label_test: str = "test",
) -> bool:
    """Compare two tensors, print stats, return True if close enough."""
    if ref.shape != test.shape:
        print(f"    FAIL {name}: shape mismatch {label_ref}={ref.shape} vs {label_test}={test.shape}")
        return False

    abs_diff = (ref.float() - test.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    close = torch.allclose(ref.float(), test.float(), atol=atol, rtol=rtol)

    status = "PASS" if close else "FAIL"
    print(f"    {status} {name} ({label_ref} vs {label_test}): "
          f"max={max_diff:.6e}, mean={mean_diff:.6e}")

    if not close:
        rel_diff = abs_diff / (ref.float().abs() + 1e-8)
        print(f"           max_rel={rel_diff.max().item():.6e}, "
              f"frac>{atol}={((abs_diff > atol).float().mean().item()):.4%}")

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
    """Run a single test case comparing all three implementations."""
    total_tokens = sum(seq_lens)
    dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16"}[dtype]

    print(f"\n{'='*70}")
    print(f"Test: seqs={seq_lens}, heads={n_q_heads}/{n_kv_heads}, "
          f"dim={head_dim}, {dtype_str}, causal={causal}")
    print(f"{'='*70}")

    q, k, v, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen = \
        make_test_data(seq_lens, n_q_heads, n_kv_heads, head_dim, dtype, device)

    all_passed = True

    # ==================== FORWARD ====================

    # 1. Naive (ground truth, fp32 math)
    print("  Computing naive attention (fp32 ground truth)...")
    naive_out, naive_lse = naive_attention_fwd(
        q, k, v, q_seq_offsets, k_seq_offsets, causal=causal,
    )
    torch.cuda.synchronize()

    # 2. flash_attn (awsm_flash_attn)
    ref_out = torch.empty(total_tokens, n_q_heads, head_dim, dtype=dtype, device=device)
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
        raise RuntimeError(f"flash_attn forward failed: {e}") from e

    # 3. C wrapper (FlashAttentionHelper)
    cwrap_out = torch.empty(total_tokens, n_q_heads, head_dim, dtype=dtype, device=device)
    cwrap_lse = torch.empty(n_q_heads, total_tokens, dtype=torch.float32, device=device)
    try:
        helper.forward(
            q, k, v, cwrap_out, cwrap_lse,
            q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens,
            max_seqlen, max_seqlen,
            causal=causal,
        )
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(f"C wrapper (FlashAttentionHelper) forward failed: {e}") from e

    # Compare forward outputs
    print("  Forward (out):")
    all_passed &= compare_tensors("out", naive_out, ref_out, atol, rtol,
                                  "naive", "flash_attn")
    all_passed &= compare_tensors("out", naive_out, cwrap_out, atol, rtol,
                                  "naive", "c_wrapper")
    all_passed &= compare_tensors("out", ref_out, cwrap_out, atol, rtol,
                                  "flash_attn", "c_wrapper")

    print("  Forward (softmax_lse):")
    all_passed &= compare_tensors("lse", naive_lse, ref_lse, atol, rtol,
                                  "naive", "flash_attn")
    all_passed &= compare_tensors("lse", naive_lse, cwrap_lse, atol, rtol,
                                  "naive", "c_wrapper")
    all_passed &= compare_tensors("lse", ref_lse, cwrap_lse, atol, rtol,
                                  "flash_attn", "c_wrapper")

    if not test_backward:
        return all_passed

    # ==================== BACKWARD ====================

    dout = torch.randn_like(ref_out)

    # 1. Naive backward (autograd, fp32)
    print("  Computing naive backward (autograd fp32)...")
    naive_dq, naive_dk, naive_dv = naive_attention_bwd(
        dout, q, k, v, q_seq_offsets, k_seq_offsets, causal=causal,
    )
    torch.cuda.synchronize()

    # 2. flash_attn backward
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
        raise RuntimeError(f"flash_attn backward failed: {e}") from e

    # 3. C wrapper backward
    cwrap_dq = torch.zeros_like(q)
    cwrap_dk = torch.zeros_like(k)
    cwrap_dv = torch.zeros_like(v)
    try:
        helper.backward(
            dout, q, k, v, cwrap_out, cwrap_lse,
            cwrap_dq, cwrap_dk, cwrap_dv,
            q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens,
            max_seqlen, max_seqlen,
            causal=causal,
        )
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(f"C wrapper backward failed: {e}") from e

    for grad_name, n_grad, r_grad, c_grad in [
        ("dq", naive_dq, ref_dq, cwrap_dq),
        ("dk", naive_dk, ref_dk, cwrap_dk),
        ("dv", naive_dv, ref_dv, cwrap_dv),
    ]:
        print(f"  Backward ({grad_name}):")
        all_passed &= compare_tensors(grad_name, n_grad, r_grad, atol, rtol,
                                      "naive", "flash_attn")
        all_passed &= compare_tensors(grad_name, n_grad, c_grad, atol, rtol,
                                      "naive", "c_wrapper")
        all_passed &= compare_tensors(grad_name, r_grad, c_grad, atol, rtol,
                                      "flash_attn", "c_wrapper")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Three-way comparison: naive vs flash_attn vs C wrapper"
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument("--atol", type=float, default=0.08,
                        help="Absolute tolerance (default: 0.08, covers bf16 rounding)")
    parser.add_argument("--rtol", type=float, default=0.05,
                        help="Relative tolerance (default: 0.05)")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    helper = FlashAttentionHelper(device=device)
    print(f"GPU: {torch.cuda.get_device_name(device)}, "
          f"arch=SM{helper.arch}, sm_count={helper.sm_count}")
    print(f"dtype={args.dtype}, atol={args.atol}, rtol={args.rtol}, "
          f"backward={not args.no_backward}")

    # Test cases: (seq_lens, n_q_heads, n_kv_heads, head_dim, causal)
    # Kept small-ish because naive attention is O(n^2) and runs per-head
    test_cases = [
        # Basic MHA
        ([64],          32, 32, 128, True),
        ([64],          32, 32, 128, False),

        # GQA
        ([64],          32, 8,  128, True),
        ([64],          32, 8,  128, False),

        # Multi-sequence
        ([32, 64],      32, 32, 128, True),
        ([32, 64],      32, 8,  128, True),

        # Different head dim
        ([32, 64],      32, 8,  64,  True),

        # Single-token sequences (the edge case)
        ([1, 1, 1],     32, 8,  128, True),
        ([1, 1, 1],     32, 32, 128, True),

        # Mixed short/long
        ([4, 128],      32, 8,  128, True),

        # Non-causal multi-seq
        ([32, 64],      32, 8,  128, False),

        # Very short
        ([2, 2],        32, 8,  128, True),
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
    print(f"SUMMARY: {num_passed} passed, {num_failed} failed "
          f"out of {num_passed + num_failed} tests")
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