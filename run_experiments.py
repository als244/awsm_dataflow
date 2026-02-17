"""
run_experiments.py
------------------
Sweeps over all combinations of the SWEEP parameters defined below,
launching train.py for each combination as a subprocess with stdout/stderr
redirected to a dedicated log file.

To add or remove a sweep dimension:
  1. Add/remove an entry in SWEEP_PARAMS.
  2. The key must exactly match the argparse flag name used in train.py
     (without the leading "--").
  3. Set the value to a list of options to try.

Any parameter NOT listed in SWEEP_PARAMS will use train.py's own default.
To hard-pin a non-swept parameter to a specific value, add it to
FIXED_PARAMS instead.
"""

import itertools
import subprocess
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Sweep configuration
# Edit these to define your experiment grid.
# ---------------------------------------------------------------------------

# (seq_len, seqs_per_step) are swept together as paired tuples since they are
# co-dependent — adjusting one typically requires adjusting the other to
# maintain a target token budget per step.
SEQ_CONFIGS: list[tuple[int, int]] = [
    # (seq_len, seqs_per_step)
    (1024, 512),
    # (2048, 256),
    # (4096, 128),
    (8192, 64),
    # (16384, 32),
    # (32768, 16),
    (65536, 8),
    # (131072, 4),
    # (262144, 2),
]

SWEEP_PARAMS = {
    # SEQ_CONFIGS is handled separately below — do not add seq_len or
    # seqs_per_step here, they will be injected automatically.
    "max_gpu_mem_gb": [16, 18, 20, 22, 24, 26, 28, 30, 40, 50, 60, 70, None],
    # "model_choice":   ["llama3_8B", "olmoe_7Bx1B", "dense_15B", "sparse_16Bx3B", "qwen3_32B", "qwen3_30Bx3B"],
    "model_choice":   ["llama3_8B", "olmoe_7Bx1B", "dense_15B", "sparse_16Bx3B"],
    "force_saved_act_level": [None, 0, 3],
}

# ---------------------------------------------------------------------------
# Fixed overrides (applied to every run, not swept over).
# Leave empty if you want train.py defaults for everything else.
# ---------------------------------------------------------------------------

FIXED_PARAMS = {
    "max_steps": 5,
}

# ---------------------------------------------------------------------------
# Runner settings
# ---------------------------------------------------------------------------

TRAIN_SCRIPT   = "bench_train.py"          # Path to train.py relative to this script
PYTHON         = sys.executable      # Use the same Python interpreter
LOG_BASE_DIR   = "experiment_logs"   # Root directory for all log files
DRY_RUN        = False               # If True, print commands without running


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_val(val) -> str:
    """Format a parameter value for use in run names and paths."""
    if val is None:
        return "None"
    return str(val)


def params_to_run_name(combo: dict) -> str:
    """Build a run name from a parameter combo dict.

    Format: {model_choice}_seqlen_{seq_len}_seqsperstep_{seqs_per_step}_maxgpumemgb_{max_gpu_mem_gb}_maxhostmemgb_{max_host_mem_gb}_forcesavedactlevel_{force_saved_act_level}
    """
    model_choice          = _fmt_val(combo.get("model_choice"))
    seq_len               = _fmt_val(combo.get("seq_len"))
    seqs_per_step         = _fmt_val(combo.get("seqs_per_step"))
    max_gpu_mem_gb        = _fmt_val(combo.get("max_gpu_mem_gb"))
    max_host_mem_gb       = _fmt_val(combo.get("max_host_mem_gb"))
    force_saved_act_level = _fmt_val(combo.get("force_saved_act_level"))

    return (
        f"{model_choice}"
        f"_seqlen_{seq_len}"
        f"_seqsperstep_{seqs_per_step}"
        f"_maxgpumemgb_{max_gpu_mem_gb}"
        f"_maxhostmemgb_{max_host_mem_gb}"
        f"_forcesavedactlevel_{force_saved_act_level}"
    )


def params_to_log_path(combo: dict) -> str:
    """Build the log file path from a parameter combo dict.

    Format: experiment_logs/{model_choice}/seqlen_{seq_len}_seqsperstep_{seqs_per_step}_maxgpumemgb_{max_gpu_mem_gb}_maxhostmemgb_{max_host_mem_gb}_forcesavedactlevel_{force_saved_act_level}.log
    """
    model_choice          = _fmt_val(combo.get("model_choice"))
    seq_len               = _fmt_val(combo.get("seq_len"))
    seqs_per_step         = _fmt_val(combo.get("seqs_per_step"))
    max_gpu_mem_gb        = _fmt_val(combo.get("max_gpu_mem_gb"))
    max_host_mem_gb       = _fmt_val(combo.get("max_host_mem_gb"))
    force_saved_act_level = _fmt_val(combo.get("force_saved_act_level"))

    subdir = os.path.join(
        LOG_BASE_DIR,
        model_choice,
    )
    filename = (
        f"seqlen_{seq_len}"
        f"_seqsperstep_{seqs_per_step}"
        f"_maxgpumemgb_{max_gpu_mem_gb}"
        f"_maxhostmemgb_{max_host_mem_gb}"
        f"_forcesavedactlevel_{force_saved_act_level}"
        f".log"
    )
    return os.path.join(subdir, filename)


def build_cmd(combo: dict, fixed: dict, run_name: str) -> list[str]:
    """Construct the subprocess command list for a single experiment."""
    cmd = [PYTHON, TRAIN_SCRIPT]

    all_params = {**fixed, **combo, "run_name": run_name}

    for key, val in all_params.items():
        if val is None:
            # Omit the flag entirely so train.py uses its own default (None)
            continue
        cmd.extend([f"--{key}", str(val)])

    return cmd


def all_combos(sweep: dict, seq_configs: list[tuple[int, int]]) -> list[dict]:
    """Return list of dicts, one per combination in the sweep grid.

    seq_configs are paired tuples of (seq_len, seqs_per_step) that are iterated
    together (not crossed against each other), then crossed against all other
    sweep dimensions.
    """
    # Build the non-seq portion of the grid
    if sweep:
        keys         = list(sweep.keys())
        values       = list(sweep.values())
        other_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    else:
        other_combos = [{}]

    # Cross seq_configs with the rest of the grid
    result = []
    for seq_len, seqs_per_step in seq_configs:
        for other in other_combos:
            result.append({"seq_len": seq_len, "seqs_per_step": seqs_per_step, **other})
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    combos = all_combos(SWEEP_PARAMS, SEQ_CONFIGS)
    total  = len(combos)

    print(f"Experiment sweep: {total} combination(s)")
    print(f"Logs directory  : {LOG_BASE_DIR}/")
    if DRY_RUN:
        print("*** DRY RUN — commands will be printed but not executed ***")
    print()

    results = []  # list of (run_name, returncode)

    for idx, combo in enumerate(combos, start=1):
        run_name = params_to_run_name(combo)
        cmd      = build_cmd(combo, FIXED_PARAMS, run_name)
        log_path = params_to_log_path(combo)

        # Ensure the model-specific subdirectory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        print(f"[{idx}/{total}] run_name : {run_name}")
        print(f"         command  : {' '.join(cmd)}")
        print(f"         log      : {log_path}")

        if DRY_RUN:
            print()
            continue

        with open(log_path, "w") as log_file:
            # Write a header into the log so it's self-documenting
            log_file.write(f"# run_name : {run_name}\n")
            log_file.write(f"# command  : {' '.join(cmd)}\n")
            log_file.write(f"# started  : {datetime.now().isoformat()}\n\n")
            log_file.flush()

            proc = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,   # merge stderr into the same file
            )

        if proc.returncode != 0:
            # Re-read the tail of the log to surface the error in the console
            try:
                with open(log_path, "r") as f:
                    log_contents = f.read()
                # Show last 50 lines to capture the traceback
                tail_lines = log_contents.strip().splitlines()[-50:]
                tail = "\n".join(tail_lines)
            except Exception:
                tail = "(could not read log file)"

            print(f"         status   : FAILED (rc={proc.returncode})")
            print(f"         ---- last lines of {log_path} ----")
            print(tail)
            print(f"         ---- end of error output ----\n")
        else:
            print(f"         status   : OK\n")

        results.append((run_name, proc.returncode))

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    if not DRY_RUN:
        print("=" * 60)
        print("SWEEP SUMMARY")
        print("=" * 60)
        failed = [(name, rc) for name, rc in results if rc != 0]
        for name, rc in results:
            mark = "✓" if rc == 0 else "✗"
            print(f"  {mark}  {name}  (rc={rc})")
        print()
        if failed:
            print(f"{len(failed)}/{total} run(s) FAILED:")
            for name, rc in failed:
                print(f"    {name}  (rc={rc})")
            sys.exit(1)
        else:
            print(f"All {total} run(s) completed successfully.")


if __name__ == "__main__":
    main()