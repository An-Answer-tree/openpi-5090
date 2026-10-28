# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.3", "numpy==1.26.4"]
# ///
"""Draw Figure 1 from archived single-view SFT evaluation counts.

Run with `uv run --script paper/Figures/gen_fig_viewpoint_performance.py`.
Alternatively, run with the existing openpi Conda environment's Python.
The adjacent JSON records the plotted counts and their archive sources.
Exports retain the placeholder's 15.75 x 4 cm size, without tight cropping.
Each suite has a magnified panel with an explicitly marked, nonzero baseline.
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
    fig, axes = plt.subplots(1, 4, figsize=(data["width_cm"] / 2.54, data["height_cm"] / 2.54))
    fig.subplots_adjust(left=0.057, right=0.998, bottom=0.098, top=0.835, wspace=0.25)

    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    hatches = ["", "//", "\\\\", ".."]
    y_ranges = [(60, 90), (60, 90), (60, 90), (20, 80)]
    for suite_index, ax in enumerate(axes):
        suite_rates = rates[:, suite_index]
        for view_index, view in enumerate(data["views"]):
            bars = ax.bar(
                view_index,
                suite_rates[view_index],
                width=0.82,
                color=colors[view_index],
                hatch=hatches[view_index],
                edgecolor="#FFFFFF",
                linewidth=0.45,
                label=view["label"],
                zorder=3,
            )
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.5, color="#222222")

        low, high = y_ranges[suite_index]
        ax.set_ylim(low, high)
        ax.set_yticks(np.linspace(low, high, 4))
        ax.set_xlim(-0.65, 3.65)
        ax.set_xticks([])
        ax.set_xlabel(data["suite_labels"][suite_index], labelpad=1.5)
        ax.tick_params(axis="y", length=2.5, width=0.6, pad=3)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.45)
        ax.text(
            0.5,
            1.015,
            f"Gap: {np.ptp(suite_rates):.1f} pp",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#222222",
        )
        # The break mark makes the cropped baseline explicit in every panel.
        ax.plot(
            [0, 0],
            [0, 0],
            transform=ax.transAxes,
            marker=[(-1, -0.5), (1, 0.5)],
            markersize=7,
            markeredgewidth=0.8,
            color="#222222",
            linestyle="none",
            clip_on=False,
            zorder=5,
        )

    axes[0].set_ylabel("Success rate (%)", labelpad=2)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.54, 0.99),
        ncol=4,
        frameon=False,
        handlelength=1.4,
        handleheight=0.85,
        handletextpad=0.5,
        columnspacing=1.6,
        borderpad=0,
        borderaxespad=0,
    )
    fig.savefig(figure_dir / "Figure1_viewpoint_performance.pdf", facecolor="white")
    fig.savefig(figure_dir / "Figure1_viewpoint_performance.png", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
