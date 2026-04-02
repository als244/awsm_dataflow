"""
run_train_sweep.py
------------------
Sweeps over all combinations of the parameters defined below,
launching train.py for each combination as a subprocess with stdout/stderr
redirected to a dedicated log file.

Key behaviour:
  - (seq_len, seqs_per_step) are paired tuples.
  - For each seqs_per_step value, ALL integer factor pairs
    (seqs_per_batch, grad_accum_steps) such that
    seqs_per_batch * grad_accum_steps == seqs_per_step are enumerated
    and swept over.
  - Remaining dimensions (zero_stage, save_act_layer_frac, offload_act,
    model_name) are crossed against every (seq_len, seqs_per_batch,
    grad_accum_steps) combination.
  - num_steps is fixed at 3 for every run.
  - Each run is launched via the deepspeed launcher with a random master port.
"""

import argparse
import itertools
import subprocess
import os
import sys
import math
import random
from datetime import datetime

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

SEQ_CONFIGS: list[tuple[int, int]] = [
    # (seq_len, seqs_per_step)
    (1024, 512),
    (2048, 256),
    (4096, 128),
    (8192, 64),
    (16384, 32),
    (32768, 16),
    (65536, 8),
    (131072, 4),
]

SWEEP_PARAMS = {
    "zero_stage":           [0, 1, 2, 3],
    "save_act_layer_frac":  [0, 0.125, 0.25, 0.5, 0.625, 0.75, 0.875, 1],
    "offload_act":          [False, True],
    "model_name":           [
        "llama3_8B",
        "olmoe_7Bx1B",
        "dense_15B",
        "sparse_16Bx3B",
    ],
}

# ---------------------------------------------------------------------------
# Fixed overrides (applied to every run, not swept).
# ---------------------------------------------------------------------------

FIXED_PARAMS = {
    "num_steps": 3,
}

# ---------------------------------------------------------------------------
# Runner settings
# ---------------------------------------------------------------------------

TRAIN_SCRIPT = "train.py"
NUM_GPUS     = 1
LOG_BASE_DIR = "experiment_logs"
DRY_RUN      = False

# Port range for random master port selection (avoid well-known ports)
MASTER_PORT_MIN = 29500
MASTER_PORT_MAX = 39999


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_master_port() -> int:
    """Return a random port in the configured range for DeepSpeed master."""
    return random.randint(MASTER_PORT_MIN, MASTER_PORT_MAX)


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    """Return all (a, b) pairs with a * b == n, sorted by a ascending."""
    pairs = []
    for a in range(1, int(math.isqrt(n)) + 1):
        if n % a == 0:
            b = n // a
            pairs.append((a, b))
            if a != b:
                pairs.append((b, a))
    pairs.sort(key=lambda p: p[0])
    return pairs


def _fmt_val(val) -> str:
    if val is None:
        return "None"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        # Avoid ugly floating-point representation in filenames
        if val == int(val):
            return str(int(val))
        return str(val)
    return str(val)


def params_to_log_path(combo: dict) -> str:
    """Build the log file path from a parameter combo dict.

    Layout:
        experiment_logs/{model_name}/seqlen_{}_seqsperbatch_{}_gradaccum_{}_zero_{}_salf_{}_offact_{}.log
    """
    model = combo["model_name"]
    parts = (
        f"seqlen_{_fmt_val(combo['seq_len'])}"
        f"_spb_{_fmt_val(combo['seqs_per_batch'])}"
        f"_ga_{_fmt_val(combo['grad_accum_steps'])}"
        f"_zero_{_fmt_val(combo['zero_stage'])}"
        f"_salf_{_fmt_val(combo['save_act_layer_frac'])}"
        f"_offact_{_fmt_val(combo['offload_act'])}"
    )
    return os.path.join(LOG_BASE_DIR, model, parts + ".log")


def params_to_run_name(combo: dict) -> str:
    """Human-readable run identifier."""
    model = combo["model_name"]
    return (
        f"{model}"
        f"_seqlen_{_fmt_val(combo['seq_len'])}"
        f"_spb_{_fmt_val(combo['seqs_per_batch'])}"
        f"_ga_{_fmt_val(combo['grad_accum_steps'])}"
        f"_zero_{_fmt_val(combo['zero_stage'])}"
        f"_salf_{_fmt_val(combo['save_act_layer_frac'])}"
        f"_offact_{_fmt_val(combo['offload_act'])}"
    )


# These flags use action='store_true' in train.py, so they should be
# included as bare flags (no value) when True, and omitted when False.
STORE_TRUE_FLAGS = {"offload_act", "use_muon"}


def build_cmd(combo: dict, fixed: dict, run_name: str) -> list[str]:
    """Construct the subprocess command list for a single experiment.

    Uses the deepspeed launcher with a random master port per run.
    """
    master_port = _random_master_port()

    cmd = [
        "deepspeed",
        f"--num_gpus={NUM_GPUS}",
        f"--master_port={master_port}",
        TRAIN_SCRIPT,
    ]

    all_params = {**fixed, **combo}

    for key, val in all_params.items():
        if val is None:
            continue
        if key in STORE_TRUE_FLAGS:
            if val:
                cmd.append(f"--{key}")
            # If False, omit entirely — argparse defaults to False
        else:
            cmd.extend([f"--{key}", str(val)])

    return cmd


def all_combos(
    sweep: dict,
    seq_configs: list[tuple[int, int]],
) -> list[dict]:
    """Return list of dicts, one per combination in the full sweep grid.

    For each (seq_len, seqs_per_step) pair, all factor pairs of seqs_per_step
    are enumerated as (seqs_per_batch, grad_accum_steps), then crossed against
    the remaining sweep dimensions.
    """
    # Build the non-seq, non-batch portion of the grid
    keys   = list(sweep.keys())
    values = list(sweep.values())
    other_combos = [dict(zip(keys, c)) for c in itertools.product(*values)]

    result = []
    for seq_len, seqs_per_step in seq_configs:
        factor_pairs = _factor_pairs(seqs_per_step)
        for seqs_per_batch, grad_accum_steps in factor_pairs:
            for other in other_combos:
                result.append({
                    "seq_len": seq_len,
                    "seqs_per_batch": seqs_per_batch,
                    "grad_accum_steps": grad_accum_steps,
                    **other,
                })
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep over experiment configurations and launch train.py for each."
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        metavar="N",
        help="Resume from experiment number N (1-indexed). Default: 1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=DRY_RUN,
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=NUM_GPUS,
        help=f"Number of GPUs per run. Default: {NUM_GPUS}.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(line_buffering=True)

    args   = parse_args()

    global NUM_GPUS
    NUM_GPUS = args.num_gpus

    combos = all_combos(SWEEP_PARAMS, SEQ_CONFIGS)
    total  = len(combos)

    start_from = args.start_from
    dry_run    = args.dry_run

    if start_from < 1 or start_from > total:
        print(f"ERROR: --start-from must be between 1 and {total} (got {start_from})")
        sys.exit(1)

    print(f"Experiment sweep: {total} combination(s)")
    print(f"DeepSpeed GPUs  : {NUM_GPUS}")
    if start_from > 1:
        print(f"Resuming from   : experiment #{start_from}  (skipping first {start_from - 1})")
    print(f"Logs directory  : {LOG_BASE_DIR}/")
    if dry_run:
        print("*** DRY RUN — commands will be printed but not executed ***")
    print()

    results = []

    for idx, combo in enumerate(combos, start=1):
        if idx < start_from:
            continue

        run_name = params_to_run_name(combo)
        cmd      = build_cmd(combo, FIXED_PARAMS, run_name)
        log_path = params_to_log_path(combo)

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        print(f"[{idx}/{total}] run_name : {run_name}")
        print(f"         command  : {' '.join(cmd)}")
        print(f"         log      : {log_path}")

        if dry_run:
            print()
            continue

        start_time = datetime.now()
        print(f"         started  : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        with open(log_path, "w") as log_file:
            log_file.write(f"# run_name : {run_name}\n")
            log_file.write(f"# command  : {' '.join(cmd)}\n")
            log_file.write(f"# started  : {start_time.isoformat()}\n\n")
            log_file.flush()

            proc = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        end_time = datetime.now()
        elapsed  = end_time - start_time

        if proc.returncode != 0:
            try:
                with open(log_path, "r") as f:
                    log_contents = f.read()
                tail_lines = log_contents.strip().splitlines()[-50:]
                tail = "\n".join(tail_lines)
            except Exception:
                tail = "(could not read log file)"

            print(f"         status   : FAILED (rc={proc.returncode})")
            print(f"         finished : {end_time.strftime('%Y-%m-%d %H:%M:%S')}  (elapsed {elapsed})")
            print(f"         ---- last lines of {log_path} ----")
            print(tail)
            print(f"         ---- end of error output ----\n")
        else:
            print(f"         status   : OK")
            print(f"         finished : {end_time.strftime('%Y-%m-%d %H:%M:%S')}  (elapsed {elapsed})\n")

        results.append((run_name, proc.returncode))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    if not dry_run:
        print("=" * 60)
        print("SWEEP SUMMARY")
        if start_from > 1:
            print(f"(resumed from experiment #{start_from})")
        print("=" * 60)
        failed = [(name, rc) for name, rc in results if rc != 0]
        for name, rc in results:
            mark = "✓" if rc == 0 else "✗"
            print(f"  {mark}  {name}  (rc={rc})")
        print()
        if failed:
            print(f"{len(failed)}/{len(results)} run(s) FAILED:")
            for name, rc in failed:
                print(f"    {name}  (rc={rc})")
            sys.exit(1)
        else:
            print(f"All {len(results)} run(s) completed successfully.")


if __name__ == "__main__":
    main()