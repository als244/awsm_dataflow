"""
parse_logs.py
-------------
Parses experiment log files produced by run_train_sweep.py and outputs a CSV
with training arguments and results (throughput, memory usage).

Usage:
    python parse_logs.py <log_root_dir> [-o output.csv]

The expected directory layout is:
    <log_root_dir>/<model_name>/<logfile>.log

Filename format (from run_train_sweep.py):
    seqlen_{}_spb_{}_ga_{}_zero_{}_salf_{}_offact_{}.log
"""

import argparse
import csv
import math
import os
import re
import sys


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

FILENAME_PATTERN = re.compile(
    r"seqlen_(?P<seq_len>\d+)"
    r"_spb_(?P<seqs_per_batch>\d+)"
    r"_ga_(?P<grad_accum_steps>\d+)"
    r"_zero_(?P<zero_stage>\d+)"
    r"_salf_(?P<save_act_layer_frac>[0-9.]+)"
    r"_offact_(?P<offload_act>\w+)"
    r"\.log$"
)

# ---------------------------------------------------------------------------
# Log content parsing
# ---------------------------------------------------------------------------

# Matches the final throughput/memory block printed on success, e.g.:
#   Throughput: 7588.644334506086 Tok/sec
#   Peak Host Memory Reserved: 111.20 GB
#   Peak Device Memory Reserved: 45.00 GB
THROUGHPUT_RE = re.compile(r"Throughput:\s+([\d.]+)\s+Tok/sec")
HOST_MEM_RE = re.compile(r"Peak Host Memory Reserved:\s+([\d.]+)\s+GB")
DEVICE_MEM_RE = re.compile(r"Peak Device Memory Reserved:\s+([\d.]+)\s+GB")


def parse_filename(filename: str) -> dict | None:
    """Extract training arguments from a log filename."""
    m = FILENAME_PATTERN.search(filename)
    if not m:
        return None
    d = m.groupdict()
    return {
        "seq_len": int(d["seq_len"]),
        "seqs_per_batch": int(d["seqs_per_batch"]),
        "grad_accum_steps": int(d["grad_accum_steps"]),
        "zero_stage": int(d["zero_stage"]),
        "save_act_layer_frac": float(d["save_act_layer_frac"]),
        "offload_activations": d["offload_act"] == "True",
    }


def parse_log_content(text: str) -> dict:
    """Extract results from log file content. Returns NaN for missing values."""
    throughput = float("nan")
    host_mem = float("nan")
    device_mem = float("nan")

    m = THROUGHPUT_RE.search(text)
    if m:
        throughput = float(m.group(1))

    m = HOST_MEM_RE.search(text)
    if m:
        host_mem = float(m.group(1))

    m = DEVICE_MEM_RE.search(text)
    if m:
        device_mem = float(m.group(1))

    return {
        "throughput_tok_per_sec": throughput,
        "peak_host_memory_gb": host_mem,
        "peak_device_memory_gb": device_mem,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "model_name",
    "sequence_length",
    "sequences_per_batch",
    "gradient_accumulation_steps",
    "zero_stage",
    "saved_activation_layer_fraction",
    "offload_activations",
    "throughput_tok_per_sec",
    "peak_host_memory_gb",
    "peak_device_memory_gb",
]


def collect_rows(root_dir: str) -> list[dict]:
    rows = []
    for model_name in sorted(os.listdir(root_dir)):
        model_dir = os.path.join(root_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        for filename in sorted(os.listdir(model_dir)):
            if not filename.endswith(".log"):
                continue
            params = parse_filename(filename)
            if params is None:
                print(f"WARNING: skipping unrecognized filename: {filename}", file=sys.stderr)
                continue

            filepath = os.path.join(model_dir, filename)
            with open(filepath, "r", errors="replace") as f:
                content = f.read()

            results = parse_log_content(content)

            rows.append({
                "model_name": model_name,
                "sequence_length": params["seq_len"],
                "sequences_per_batch": params["seqs_per_batch"],
                "gradient_accumulation_steps": params["grad_accum_steps"],
                "zero_stage": params["zero_stage"],
                "saved_activation_layer_fraction": params["save_act_layer_frac"],
                "offload_activations": params["offload_activations"],
                **results,
            })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Parse experiment log files into a CSV."
    )
    parser.add_argument(
        "log_dir",
        help="Root directory containing per-model subdirectories of .log files.",
    )
    parser.add_argument(
        "-o", "--output",
        default="experiment_results.csv",
        help="Output CSV path. Default: experiment_results.csv",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"ERROR: {args.log_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    rows = collect_rows(args.log_dir)

    if not rows:
        print("No log files found.", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Count successes vs failures
    n_success = sum(1 for r in rows if not math.isnan(r["throughput_tok_per_sec"]))
    n_failed = len(rows) - n_success
    print(f"Parsed {len(rows)} log files ({n_success} succeeded, {n_failed} failed).")
    print(f"CSV written to: {args.output}")


if __name__ == "__main__":
    main()