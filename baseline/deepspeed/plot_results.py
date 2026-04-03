"""
plot_results.py
---------------
Reads the CSV produced by parse_logs.py and generates a multi-page PDF
with one scatter plot per (model, sequence_length) combination.

Each plot shows:
  - X axis: Peak Device Memory (GB)
  - Y axis: Throughput (Tok/sec)
  - Points colored by configuration parameters

Usage:
    python plot_results.py experiment_results.csv [-o plots.pdf]
"""

import argparse
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np


def make_label(row: pd.Series) -> str:
    """Build a compact label string from the config columns."""
    parts = []
    parts.append(f"spb={int(row['sequences_per_batch'])}")
    parts.append(f"ga={int(row['gradient_accumulation_steps'])}")
    parts.append(f"z{int(row['zero_stage'])}")
    parts.append(f"salf={row['saved_activation_layer_fraction']}")
    if row["offload_activations"]:
        parts.append("offload")
    return " ".join(parts)


def plot_group(ax, df_group, model_name, seq_len):
    """Plot a single (model, seq_len) group onto the given axes."""
    # Drop rows where results are NaN (failed runs)
    valid = df_group.dropna(subset=["throughput_tok_per_sec", "peak_device_memory_gb"])
    failed_count = len(df_group) - len(valid)

    # Color by zero_stage, marker by offload_activations
    zero_stages = sorted(valid["zero_stage"].unique())
    colors = {z: c for z, c in zip(zero_stages, plt.cm.tab10.colors)}

    marker_map = {True: "^", False: "o"}

    for _, row in valid.iterrows():
        zs = row["zero_stage"]
        off = row["offload_activations"]
        ax.scatter(
            row["peak_device_memory_gb"],
            row["throughput_tok_per_sec"],
            c=[colors[zs]],
            marker=marker_map[off],
            s=50,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.85,
            zorder=3,
        )

    # Legend entries for zero stage (color)
    legend_handles = []
    for zs in zero_stages:
        h = ax.scatter([], [], c=[colors[zs]], marker="o", s=50, edgecolors="black", linewidths=0.4)
        legend_handles.append((h, f"ZeRO {int(zs)}"))

    # Legend entries for offload (marker shape)
    for off, mkr in marker_map.items():
        h = ax.scatter([], [], c="gray", marker=mkr, s=50, edgecolors="black", linewidths=0.4)
        label = "offload act" if off else "no offload"
        legend_handles.append((h, label))

    if legend_handles:
        ax.legend(
            [h for h, _ in legend_handles],
            [l for _, l in legend_handles],
            fontsize=7,
            loc="best",
            framealpha=0.8,
        )

    title = f"{model_name}  —  seq_len = {int(seq_len)}"
    if failed_count:
        title += f"  ({failed_count} failed run{'s' if failed_count > 1 else ''} omitted)"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Peak Device Memory Reserved (GB)", fontsize=9)
    ax.set_ylabel("Throughput (Tok/sec)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results from CSV to PDF.")
    parser.add_argument("csv_file", help="Path to the CSV produced by parse_logs.py")
    parser.add_argument("-o", "--output", default="experiment_plots.pdf", help="Output PDF path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    # Ensure boolean column is parsed correctly
    if df["offload_activations"].dtype == object:
        df["offload_activations"] = df["offload_activations"].map({"True": True, "False": False})

    groups = list(df.groupby(["model_name", "sequence_length"]))
    groups.sort(key=lambda g: (g[0][0], g[0][1]))

    n_plots = len(groups)
    if n_plots == 0:
        print("No data to plot.", file=sys.stderr)
        sys.exit(1)

    # Layout: up to 2 plots per page
    plots_per_page = 2
    n_pages = math.ceil(n_plots / plots_per_page)

    with PdfPages(args.output) as pdf:
        plot_idx = 0
        for page_num in range(n_pages):
            n_on_page = min(plots_per_page, n_plots - plot_idx)
            fig, axes = plt.subplots(n_on_page, 1, figsize=(8.5, 5.5 * n_on_page))
            if n_on_page == 1:
                axes = [axes]

            for ax in axes:
                (model_name, seq_len), df_group = groups[plot_idx]
                plot_group(ax, df_group, model_name, seq_len)
                plot_idx += 1

            fig.tight_layout(pad=2.0)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Generated {n_plots} plot(s) across {n_pages} page(s) → {args.output}")


if __name__ == "__main__":
    main()