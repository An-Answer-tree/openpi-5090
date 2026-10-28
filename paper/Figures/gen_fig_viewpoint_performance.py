# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.3", "numpy==1.26.4"]
# ///
"""Draw Figure 1 from archived single-view SFT evaluation counts.

Run with `uv run --script paper/Figures/gen_fig_viewpoint_performance.py`.
Alternatively, run with the existing openpi Conda environment's Python.
The adjacent JSON records the plotted counts and their archive sources.
Exports retain the placeholder's 15.75 x 4 cm size, without tight cropping.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    """Export a vector PDF and a 600 dpi preview with identical dimensions."""
    plt.switch_backend("Agg")
    figure_dir = Path(__file__).resolve().parent
    data = json.loads((figure_dir / "Figure1_viewpoint_performance.json").read_text())
    rates = np.array([view["successes"] for view in data["views"]], dtype=float)
    rates *= 100 / data["episodes_per_suite"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "hatch.linewidth": 0.35,
        }
    )
    fig, ax = plt.subplots(figsize=(data["width_cm"] / 2.54, data["height_cm"] / 2.54))
    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.195, top=0.80)

    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    hatches = ["", "//", "\\\\", ".."]
    positions = np.arange(len(data["suite_order"]))
    bar_width = 0.18
    for index, view in enumerate(data["views"]):
        offset = (index - (len(data["views"]) - 1) / 2) * bar_width
        bars = ax.bar(
            positions + offset,
            rates[index],
            width=bar_width * 0.94,
            color=colors[index],
            hatch=hatches[index],
            edgecolor="#FFFFFF",
            linewidth=0.45,
            label=view["label"],
            zorder=3,
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.5, color="#222222")

    ax.set_ylabel("Success rate (%)", labelpad=4)
    ax.set_xticks(positions, data["suite_labels"])
    ax.set_ylim(0, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.55, len(positions) - 0.45)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=2.5, width=0.6, pad=3)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.45)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.54, 1.01),
        ncol=4,
        frameon=False,
        handlelength=1.4,
        handleheight=0.85,
        handletextpad=0.5,
        columnspacing=1.6,
    )
    fig.savefig(figure_dir / "Figure1_viewpoint_performance.pdf", facecolor="white")
    fig.savefig(figure_dir / "Figure1_viewpoint_performance.png", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
