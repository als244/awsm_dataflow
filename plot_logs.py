#!/usr/bin/env python3
"""
Parse training benchmark log files and generate plots + CSV summary.

Usage:
    python plot_logs.py <root_directory> [--output_dir OUTPUT_DIR]

Directory structure expected:
    root_directory/
        model_name_1/
            seqlen_{}_seqsperstep_{}_maxgpumemgib_{}_maxhostmemgib_{}_forcesavedactlevel_{}.log
        (also supports legacy format with maxgpumemgb / maxhostmemgb)
            ...
        model_name_2/
            ...
"""

import argparse
import ast
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# ── Research-paper (TikZ-like) style configuration ──────────────────────────

def setup_style():
    """Configure matplotlib for a clean, LaTeX research-paper aesthetic.

    Uses matplotlib's built-in mathtext with STIX fonts to approximate
    Computer Modern without requiring a full LaTeX installation.
    """
    plt.rcParams.update({
        # Use built-in mathtext (STIX looks like Computer Modern)
        "text.usetex": False,
        "mathtext.fontset": "stix",
        # Font family — Latin Modern Roman is the LaTeX default
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Latin Modern Roman",
                        "DejaVu Serif", "Times New Roman"],
        "font.size": 10,
        # Axes
        "axes.linewidth": 0.6,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titlepad": 10,
        "axes.spines.top": True,
        "axes.spines.right": True,
        # Ticks — inward like TikZ/pgfplots
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        # Grid
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        # Legend
        "legend.fontsize": 8.5,
        "legend.title_fontsize": 9.5,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.7",
        "legend.fancybox": False,
        "legend.borderpad": 0.5,
        "legend.handlelength": 2.0,
        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        # Figure — column-width for a two-column paper
        "figure.figsize": (5.5, 3.8),
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

setup_style()


# ── Filename parsing ────────────────────────────────────────────────────────

FILENAME_PATTERN = re.compile(
    r"seqlen_(?P<seqlen>\d+)"
    r"_seqsperstep_(?P<seqsperstep>\d+)"
    r"_maxgpumemgi?b_(?P<gpu_budget>[^_]+)"
    r"_maxhostmemgi?b_(?P<host_budget>[^_]+)"
    r"_forcesavedactlevel_(?P<save_level>[^.]+)"
    r"\.log$"
)


def parse_filename(fname: str) -> dict | None:
    m = FILENAME_PATTERN.search(fname)
    if not m:
        return None
    d = m.groupdict()
    d["seqlen"] = int(d["seqlen"])
    d["seqsperstep"] = int(d["seqsperstep"])
    d["gpu_budget"] = None if d["gpu_budget"] == "None" else float(d["gpu_budget"])
    d["host_budget"] = None if d["host_budget"] == "None" else float(d["host_budget"])
    d["save_level"] = None if d["save_level"] == "None" else int(d["save_level"])
    return d


# ── Log-content parsing helpers ─────────────────────────────────────────────

def _find_step3_metrics(text: str) -> dict:
    """Extract Step Time, Tokens/sec, Effective TFLOPS from [Step 3] block."""
    out = {}
    # Find the [Step 3] block
    step3_start = text.find("[Step 3]")
    if step3_start == -1:
        return out
    block = text[step3_start: step3_start + 1500]  # generous window

    m = re.search(r"Step Time:\s*([\d.]+)\s*sec", block)
    if m:
        out["step_time_sec"] = float(m.group(1))

    m = re.search(r"Throughput\s*---\s*([\d.]+)\s*Tokens/sec", block)
    if m:
        out["tokens_per_sec"] = float(m.group(1))

    m = re.search(r"([\d.]+)\s*Effective TFLOPS", block)
    if m:
        out["effective_tflops"] = float(m.group(1))

    m = re.search(r"Avg\. Loss\s*---\s*([\d.]+)", block)
    if m:
        out["loss"] = float(m.group(1))

    m = re.search(r"Max Alloc/Reserve\s*([\d.]+)/([\d.]+)\s*GiB", block)
    if m:
        out["max_alloc_gib"] = float(m.group(1))
        out["max_reserve_gib"] = float(m.group(2))

    return out


def _find_working_set_config(text: str) -> dict:
    """Parse the Working Set Config dict block."""
    out = {}
    marker = "-------- Working Set Config --------"
    idx = text.find(marker)
    if idx == -1:
        return out
    # Find the dict that follows
    rest = text[idx + len(marker):]
    # Find the opening brace
    brace_start = rest.find("{")
    if brace_start == -1:
        return out
    # Balance braces
    depth = 0
    brace_end = brace_start
    for i in range(brace_start, len(rest)):
        if rest[i] == "{":
            depth += 1
        elif rest[i] == "}":
            depth -= 1
            if depth == 0:
                brace_end = i
                break
    dict_str = rest[brace_start: brace_end + 1]
    try:
        d = ast.literal_eval(dict_str)
    except Exception:
        return out

    keys_of_interest = [
        "n_gpu_layers", "n_gpu_grads", "n_gpu_opt_layers",
        "max_training_chunks", "max_chunk_size", "max_seq_len",
        "target_round_tokens",
    ]
    for k in keys_of_interest:
        if k in d:
            out[k] = d[k]
    return out


def _find_act_buffer_info(text: str) -> dict:
    """Parse # GPU Full Act Slots, # Host Act Slots, buffer sizes."""
    out = {}
    m = re.search(r"#\s*GPU Full Act Slots:\s*(\d+)", text)
    if m:
        out["gpu_act_slots"] = int(m.group(1))

    # Fallback: sometimes it's printed differently
    if "gpu_act_slots" not in out:
        m = re.search(r"#\s*GPU Act Slots.*?:\s*(\d+)", text)
        if m:
            out["gpu_act_slots"] = int(m.group(1))

    m = re.search(r"#\s*Host Act Slots:\s*(\d+)", text)
    if m:
        out["host_act_slots"] = int(m.group(1))

    m = re.search(r"#\s*GPU Act Buffer Size:\s*([\d.]+)\s*GB", text)
    if m:
        out["gpu_act_buffer_gb"] = float(m.group(1))

    m = re.search(r"#\s*Host Act Buffer Size:\s*([\d.]+)\s*GB", text)
    if m:
        out["host_act_buffer_gb"] = float(m.group(1))

    return out


def _find_level_combos(text: str) -> dict:
    """Parse Level 3/2/1/0 (layer, chunk) combos."""
    out = {}
    for lvl in [3, 2, 1, 0]:
        m = re.search(
            rf"Level\s+{lvl}:\s*(\d+)\s*\(layer,\s*chunk\)\s*combos", text
        )
        if m:
            out[f"level_{lvl}_combos"] = int(m.group(1))
    return out


def _find_recompute_info(text: str) -> dict:
    """Parse Final Recompute Time (numerator) and Final Recompute Frac."""
    out = {}
    m = re.search(
        r"Final Recompute Time:\s*([\d.]+)\s*ms\s*/\s*([\d.]+)\s*ms.*?"
        r"Final Recompute Frac:\s*([\d.]+)",
        text,
    )
    if m:
        out["final_recompute_time_ms"] = float(m.group(1))
        out["final_recompute_total_ms"] = float(m.group(2))
        out["final_recompute_frac"] = float(m.group(3))
    return out


# ── Main parse function for a single log file ──────────────────────────────

def parse_log(filepath: str) -> dict:
    """Return a dict of all parsed fields, or a dict with 'error' key."""
    try:
        with open(filepath, "r", errors="replace") as f:
            text = f.read()
    except Exception as e:
        return {"error": str(e)}

    result = {}

    step3 = _find_step3_metrics(text)
    if not step3:
        result["error"] = "Step 3 metrics not found"
        return result

    result.update(step3)
    result.update(_find_working_set_config(text))
    result.update(_find_act_buffer_info(text))
    result.update(_find_level_combos(text))
    result.update(_find_recompute_info(text))

    return result


# ── CSV output ──────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    # From filename
    "model", "seqlen", "seqsperstep", "gpu_budget", "host_budget", "save_level",
    # Step 3 metrics
    "step_time_sec", "tokens_per_sec", "effective_tflops", "loss",
    "max_alloc_gib", "max_reserve_gib",
    # Working set config
    "n_gpu_layers", "n_gpu_grads", "n_gpu_opt_layers",
    "max_training_chunks", "max_chunk_size", "max_seq_len", "target_round_tokens",
    # Act buffer info
    "gpu_act_slots", "host_act_slots", "gpu_act_buffer_gb", "host_act_buffer_gb",
    # Level combos
    "level_3_combos", "level_2_combos", "level_1_combos", "level_0_combos",
    # Recompute info
    "final_recompute_time_ms", "final_recompute_total_ms", "final_recompute_frac",
    # Status
    "error",
]


def build_rows(root_dir: str) -> list[dict]:
    rows = []
    root = Path(root_dir)
    if not root.is_dir():
        print(f"Error: {root_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for log_file in sorted(model_dir.iterdir()):
            if not log_file.name.endswith(".log"):
                continue
            fname_info = parse_filename(log_file.name)
            if fname_info is None:
                continue

            row = {"model": model_name}
            row.update(fname_info)

            parsed = parse_log(str(log_file))
            row.update(parsed)
            rows.append(row)

    return rows


def write_csv(rows: list[dict], output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {output_path}")


# ── Plotting ────────────────────────────────────────────────────────────────

# Muted academic color palette (colorblind-friendly)
COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


# Marker shapes and line styles cycled per sequence length in combined mode
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]

# Y-metric configuration (labels use LaTeX math where appropriate)
Y_METRICS = {
    "tflops": {
        "key": "effective_tflops",
        "label": "Effective TFLOPS",
        "file_tag": "tflops",
    },
    "tok_per_sec": {
        "key": "tokens_per_sec",
        "label": "Tokens / sec",
        "file_tag": "tokpersec",
    },
}


def _format_seqlen(seqlen: int) -> str:
    """Format sequence length: 8192 -> '8K', 1024 -> '1K', 512 -> '512'."""
    if seqlen >= 1024 and seqlen % 1024 == 0:
        return f"{seqlen // 1024}K"
    return str(seqlen)


# Map directory names to professional display names.
# Add entries here as new models appear.
MODEL_DISPLAY_NAMES = {
    "llama3_8B":      "Llama3 8B",
    "llama3_8b":      "Llama3 8B",
    "dense_15B":      "Dense 15B",
    "dense_15b":      "Dense 15B",
    "olmoe_7Bx1B":    "OLMoE 7Bx1B",
    "olmoe_7bx1b":    "OLMoE 7Bx1B",
    "sparse_16Bx3B":  "Sparse 16Bx3B",
    "sparse_16bx3b":  "Sparse 16Bx3B",
}


def _display_model(raw_name: str) -> str:
    """Return a professional display name for a model directory name."""
    if raw_name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[raw_name]
    # Fallback: replace underscores with spaces, title-case
    return raw_name.replace("_", " ")



# ── Style constants (shared by PNG and PDF rendering) ───────────────────────
P_ALPHA = 0.92
P_LW = 1.6
P_MS = 8.0

S_ALPHA = 0.30
S_LW = 1.0
S_MS = 5.0


def _plot_on_axes(
    ax, ax2, entries, *,
    y_key, y_label, title_str,
    combine_seqlens, no_gpu_mem_limit_value, NONE_X,
    show_legend=True, compact=False,
    show_primary_ylabel=True, show_secondary_ylabel=True,
    show_primary_yticks=True, show_secondary_yticks=True,
):
    """Render a single (model, seqlen) chart onto the given axes pair.

    Returns (legend_handles, legend_labels) for external legend use.
    """
    from matplotlib.lines import Line2D

    unique_seqlens = sorted(set(e["seqlen"] for e in entries))
    seqlen_to_idx = {sl: i for i, sl in enumerate(unique_seqlens)}

    by_save_level = defaultdict(list)
    for e in entries:
        by_save_level[e["save_level"]].append(e)

    save_levels = sorted(
        by_save_level.keys(),
        key=lambda x: (-1, "") if x is None else (0, x),
    )

    plotted_combos = []

    for ci, sl in enumerate(save_levels):
        color = COLORS[ci % len(COLORS)]
        if sl is None:
            sl_label = "Automatic"
        elif sl == 0:
            sl_label = "Minimal Saving"
        elif sl == 3:
            sl_label = "Fully Saving"
        else:
            sl_label = f"Level {sl}"

        by_seqlen = defaultdict(list)
        for e in by_save_level[sl]:
            by_seqlen[e["seqlen"]].append(e)

        for sq in sorted(by_seqlen.keys()):
            si = seqlen_to_idx[sq]
            mkr = MARKERS[si % len(MARKERS)]
            ls = LINESTYLES[si % len(LINESTYLES)]

            pts = sorted(by_seqlen[sq],
                         key=lambda r: (r["gpu_budget"] or float("inf")))
            x_vals, y_vals, frac_vals = [], [], []
            for p in pts:
                gb = p["gpu_budget"]
                yv = p.get(y_key)
                if yv is None:
                    continue
                raw_frac = p.get("final_recompute_frac")
                try:
                    frac = float(raw_frac)
                except (TypeError, ValueError):
                    frac = 0.0
                x_vals.append(gb if gb is not None else NONE_X)
                y_vals.append(yv)
                frac_vals.append(frac)

            if not x_vals:
                continue

            if not combine_seqlens:
                mkr = "o"
                ls = "-"

            label = (f"{sl_label}, seq={sq}" if combine_seqlens
                     else sl_label)

            p_lw = P_LW * (0.75 if compact else 1.0)
            p_ms = P_MS * (0.75 if compact else 1.0)
            s_lw = S_LW * (0.75 if compact else 1.0)
            s_ms = S_MS * (0.75 if compact else 1.0)

            ax.plot(x_vals, y_vals,
                    color=color, lw=p_lw, alpha=P_ALPHA,
                    linestyle=ls, marker=mkr, markersize=p_ms,
                    markeredgecolor="white", markeredgewidth=0.4,
                    zorder=4)

            ax2.plot(x_vals, frac_vals,
                     color=color, lw=s_lw, alpha=S_ALPHA,
                     linestyle="--", marker="x", markersize=s_ms,
                     markeredgewidth=0.8, zorder=2)

            plotted_combos.append((sl, sq, color, mkr, ls, label))

    # ── Axes formatting ─────────────────────────────────────────
    ax.set_xlabel("GPU Memory Budget (GiB)")
    if show_primary_ylabel:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")
    if not show_primary_yticks:
        ax.tick_params(axis="y", labelleft=False)

    if show_secondary_ylabel:
        ax2.set_ylabel("Fraction of Fwd FLOPs Recomputed", color="0.55")
    else:
        ax2.set_ylabel("")
    ax2.tick_params(axis="y", colors="0.55")
    if not show_secondary_yticks:
        ax2.tick_params(axis="y", labelright=False)
    ax2.set_ylim(-0.05, 1.05)
    ax2.spines["right"].set_color("0.55")
    ax.set_title(title_str, fontsize=12 if not compact else 10)

    all_x = sorted(set(
        (e["gpu_budget"] if e["gpu_budget"] is not None else NONE_X)
        for e in entries if e.get(y_key) is not None
    ))
    tick_labels = [
        "No Limit" if (v == -1 and no_gpu_mem_limit_value is None)
        else f"{v:g}"
        for v in all_x
    ]
    ax.set_xticks(all_x)
    ax.set_xticklabels(tick_labels)
    ax.grid(True)

    if compact:
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax2.yaxis.label.set_size(8)
        # Ensure enough y-ticks even in compact subplots
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))

    # ── Tok/sec reference lines on primary y-ticks (TFLOPS mode only) ──
    if y_key == "effective_tflops":
        # Find one reference point with both tflops and tok/sec
        ref_tflops, ref_toksec = None, None
        for e in entries:
            tf = e.get("effective_tflops")
            ts = e.get("tokens_per_sec")
            if tf is not None and ts is not None and tf > 0:
                ref_tflops, ref_toksec = tf, ts
                break

        if ref_tflops is not None:
            ratio = ref_toksec / ref_tflops

            fig = ax.get_figure()
            fig.canvas.draw()
            yticks = ax.get_yticks()
            ymin, ymax = ax.get_ylim()

            annot_fs = 6.0 if compact else 7.0

            for yt in yticks:
                if yt < ymin or yt > ymax:
                    continue
                tok_val = yt * ratio
                if tok_val < 0:
                    continue
                if tok_val >= 1000:
                    tok_str = f"{tok_val/1000:.1f}K tok/s"
                else:
                    tok_str = f"{tok_val:.0f} tok/s"
                ax.axhline(y=yt, color="0.80", lw=0.4, ls=":",
                           zorder=0)
                ax.annotate(
                    tok_str,
                    xy=(0.0, yt), xycoords=("axes fraction", "data"),
                    fontsize=annot_fs, color="0.50",
                    ha="left", va="bottom",
                    xytext=(3, 1), textcoords="offset points",
                )

    # ── Build legend handles ────────────────────────────────────
    seen = set()
    legend_handles, legend_labels = [], []
    for (sl, sq, color, mkr, ls, lbl) in plotted_combos:
        if lbl in seen:
            continue
        seen.add(lbl)
        h = Line2D([], [],
                   color=color, lw=P_LW, linestyle=ls,
                   marker=mkr, markersize=P_MS,
                   markerfacecolor=color, markeredgecolor="white",
                   markeredgewidth=0.4, alpha=P_ALPHA)
        legend_handles.append(h)
        legend_labels.append(lbl)

    h_sec = Line2D([], [],
                   color="0.55", lw=S_LW, linestyle="--",
                   marker="x", markersize=S_MS,
                   markeredgewidth=0.8, alpha=S_ALPHA)

    if show_legend:
        all_h = list(legend_handles) + [h_sec]
        all_l = list(legend_labels) + ["Frac of Fwd FLOPs Recomputed (right axis)"]

        # Place legend below the plot so it never covers data
        ax.legend(all_h, all_l,
                  title="Saved Activations Policy",
                  loc="upper center", ncol=min(len(all_h), 4),
                  bbox_to_anchor=(0.5, -0.13),
                  handletextpad=0.5, columnspacing=1.2)

    # Return un-padded handles for PDF legend (it does its own layout)
    legend_handles.append(h_sec)
    legend_labels.append("Frac of Fwd FLOPs Recomputed (right axis)")

    return legend_handles, legend_labels


def _choose_grid(n: int) -> tuple[int, int]:
    """Choose (rows, cols) for n subplots on one page."""
    if n <= 1:
        return (1, 1)
    elif n == 2:
        return (1, 2)
    elif n == 3:
        return (1, 3)
    elif n == 4:
        return (2, 2)
    elif n <= 6:
        return (2, 3)
    elif n <= 9:
        return (3, 3)
    else:
        cols = 3
        rows_needed = (n + cols - 1) // cols
        return (rows_needed, cols)


def make_plots(
    rows: list[dict],
    output_dir: str,
    no_gpu_mem_limit_value: float | None = None,
    y_metric: str = "tflops",
    combine_seqlens: bool = False,
):
    """Create one PNG per (model, seqlen) plus one gridded PDF page per model."""
    metric = Y_METRICS[y_metric]
    y_key = metric["key"]
    y_label = metric["label"]
    file_tag = metric["file_tag"]

    NONE_X = no_gpu_mem_limit_value if no_gpu_mem_limit_value is not None else -1

    good = [r for r in rows if "error" not in r or not r["error"]]

    groups = defaultdict(list)
    for r in good:
        key = r["model"] if combine_seqlens else (r["model"], r["seqlen"])
        groups[key].append(r)

    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []

    # Collect entries grouped by model for PDF grid pages
    # model_name -> [(seqlen_or_None, entries, title_str), ...]
    model_charts: dict[str, list[tuple]] = defaultdict(list)

    common_kw = dict(
        y_key=y_key, y_label=y_label,
        combine_seqlens=combine_seqlens,
        no_gpu_mem_limit_value=no_gpu_mem_limit_value,
        NONE_X=NONE_X,
    )

    for group_key, entries in sorted(groups.items()):
        if combine_seqlens:
            model = group_key
            title_str = f"{_display_model(model)}, All Sequence Lengths"
        else:
            model, seqlen = group_key
            title_str = (
                f"{_display_model(model)}, Seqlen {_format_seqlen(seqlen)}"
            )

        # ── Individual PNG ──────────────────────────────────────────
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        _plot_on_axes(ax, ax2, entries, title_str=title_str,
                      show_legend=True, compact=False, **common_kw)
        fig.tight_layout()

        safe_model = model.replace("/", "_").replace(" ", "_")
        if combine_seqlens:
            fname = f"{file_tag}_{safe_model}_combined.png"
        else:
            fname = f"{file_tag}_{safe_model}_seqlen{seqlen}.png"
        path = os.path.join(output_dir, fname)
        fig.savefig(path, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(path)
        print(f"  Saved plot: {path}")

        # Stash for PDF
        model_charts[safe_model].append((entries, title_str))

    # ── Compile per-model PDF (one page per model, gridded subplots) ────
    pdf_paths = []
    for safe_model, chart_list in model_charts.items():
        n = len(chart_list)
        nrows, ncols = _choose_grid(n)

        # Page size: landscape A4-ish for grids, portrait-ish for single
        if ncols >= 2:
            page_w = 11.0
            page_h = 8.5
        else:
            page_w = 7.0
            page_h = 5.0

        fig, axes_flat = plt.subplots(
            nrows, ncols, figsize=(page_w, page_h),
            sharey=True,
            squeeze=False,
        )

        # We need to create twinx for each subplot
        all_legend_handles, all_legend_labels = [], []
        seen_labels = set()
        ax2_list = []

        for idx, (entries, title_str) in enumerate(chart_list):
            r, c = divmod(idx, ncols)
            ax = axes_flat[r][c]
            ax2 = ax.twinx()
            ax2_list.append((r, c, ax2))

            # Only show axis LABELS on edges; ticks always visible
            is_leftmost = (c == 0)
            is_rightmost = (c == ncols - 1) or (idx == n - 1)

            lh, ll = _plot_on_axes(
                ax, ax2, entries, title_str=title_str,
                show_legend=False, compact=(n > 1),
                show_primary_ylabel=is_leftmost,
                show_primary_yticks=True,
                show_secondary_ylabel=is_rightmost,
                show_secondary_yticks=True,
                **common_kw,
            )

            # Accumulate unique legend entries across all subplots
            for h, l in zip(lh, ll):
                if l not in seen_labels:
                    seen_labels.add(l)
                    all_legend_handles.append(h)
                    all_legend_labels.append(l)

        # Hide unused subplot slots
        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes_flat[r][c].set_visible(False)

        # sharey=True hides tick labels on non-leftmost axes; re-enable them.
        # Also force more y-ticks since compact subplots default to too few.
        from matplotlib.ticker import MaxNLocator
        for idx in range(n):
            r, c = divmod(idx, ncols)
            ax = axes_flat[r][c]
            ax.tick_params(axis="y", labelleft=True)
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10])
            )

        # Legend below the subplots — primary entries on one row,
        # Frac of Fwd FLOPs Recomputed on a second row via column-major trick.
        # Using a single legend with all entries and ncol = n_primary.
        from matplotlib.lines import Line2D as _L2D

        primary_h = [h for h, l in zip(all_legend_handles, all_legend_labels)
                     if "Fwd FLOPs Recomputed" not in l]
        primary_l = [l for l in all_legend_labels
                     if "Fwd FLOPs Recomputed" not in l]
        sec_h = [h for h, l in zip(all_legend_handles, all_legend_labels)
                 if "Fwd FLOPs Recomputed" in l]
        sec_l = [l for l in all_legend_labels
                 if "Fwd FLOPs Recomputed" in l]

        all_h = primary_h + sec_h
        all_l = primary_l + sec_l

        fig.legend(
            all_h, all_l,
            title="Saved Activations Policy",
            loc="upper center",
            ncol=min(len(all_h), 5),
            fontsize=8, title_fontsize=9,
            framealpha=0.92, edgecolor="0.7",
            bbox_to_anchor=(0.5, 0.09),
            handletextpad=0.5, columnspacing=1.5,
        )

        # Add model name as a super-title
        display_name = _display_model(
            next((m for m in set(r["model"] for r in good)
                  if m.replace("/", "_").replace(" ", "_") == safe_model),
                 safe_model)
        )
        fig.suptitle(display_name, fontsize=14, fontweight="bold", y=0.98)

        fig.tight_layout(rect=[0, 0.12, 1, 0.93])

        pdf_name = f"{file_tag}_{safe_model}_all.pdf"
        pdf_path = os.path.join(output_dir, pdf_name)
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        pdf_paths.append(pdf_path)
        print(f"  Saved PDF:  {pdf_path}")

    # ── Merge all per-model PDFs into a single report ───────────────────
    report_path = None
    if pdf_paths:
        from pypdf import PdfWriter, PdfReader
        report_name = f"{file_tag}_report.pdf"
        report_path = os.path.join(output_dir, report_name)
        writer = PdfWriter()
        for p in pdf_paths:
            reader = PdfReader(p)
            for page in reader.pages:
                writer.add_page(page)
        with open(report_path, "wb") as f:
            writer.write(f)
        print(f"  Saved report: {report_path}")

    return plot_paths, pdf_paths, report_path


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parse training benchmark logs and generate plots."
    )
    parser.add_argument(
        "root_dir",
        help="Root directory containing model subdirectories with .log files.",
    )
    parser.add_argument(
        "--output_dir", "-o",
        default=None,
        help="Directory for output CSV and plots (default: <root_dir>/output).",
    )
    parser.add_argument(
        "--no_gpu_mem_limit_value", "-n",
        type=float, default=None,
        help="Numeric x-axis value to use for gpu_budget=None (e.g. 80). "
             "If not set, 'No Limit' appears as a categorical tick.",
    )
    parser.add_argument(
        "--y_metric",
        choices=["tflops", "tok_per_sec"],
        default="tflops",
        help="Y-axis metric: 'tflops' (default) or 'tok_per_sec'.",
    )
    parser.add_argument(
        "--combine_seqlens",
        action="store_true",
        help="Combine all sequence lengths for the same model onto one chart. "
             "Each seqlen gets a unique marker shape and line style.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    rows = build_rows(args.root_dir)
    if not rows:
        print("No log files found. Check your directory structure.", file=sys.stderr)
        sys.exit(1)

    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    write_csv(rows, csv_path)

    plot_paths, pdf_paths, report_path = make_plots(
        rows, output_dir,
        no_gpu_mem_limit_value=args.no_gpu_mem_limit_value,
        y_metric=args.y_metric,
        combine_seqlens=args.combine_seqlens,
    )
    print(f"\nDone. {len(rows)} logs parsed, {len(plot_paths)} plots, "
          f"{len(pdf_paths)} model PDFs generated."
          + (f"\nReport: {report_path}" if report_path else ""))


if __name__ == "__main__":
    main()