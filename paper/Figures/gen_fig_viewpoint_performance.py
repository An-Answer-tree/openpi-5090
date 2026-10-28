# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.3", "numpy==1.26.4", "pillow>=10"]
# ///
"""Draw Figure 1 from archived single-view SFT evaluation counts.

Run with `uv run --script paper/Figures/gen_fig_viewpoint_performance.py`.
Alternatively, run with the existing openpi Conda environment's Python.
The adjacent JSON records the plotted counts and their archive sources.
The four images show one synchronized state, re-rendered at 1536 x 1536.
Dot plots compare suite-level success rates independently of that episode.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
        }
    )
    fig = plt.figure(figsize=(data["width_cm"] / 2.54, data["height_cm"] / 2.54))
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    markers = ["o", "s", "^", "D"]
    fig.text(0.004, 0.99, "(a) Synchronized camera views", va="top", fontsize=9.5, fontweight="bold")
    image_width = 0.239
    image_height = image_width * data["width_cm"] / data["height_cm"]
    for i, view in enumerate(data["views"]):
        left = 0.005 + i * 0.251
        ax = fig.add_axes([left, 0.414, image_width, image_height])
        with Image.open(figure_dir / "Figure1_views" / f"{view['label'].lower()}view.png") as frame:
            ax.imshow(frame, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(colors[i])
            spine.set_linewidth(1.1)
        ax.text(
            0.035,
            0.965,
            f"{view['label']} view",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=colors[i],
            fontweight="bold",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.93, "pad": 2.5},
        )

    fig.text(0.004, 0.397, "(b) Single-view SFT success rate (%)", va="top", fontsize=9.5, fontweight="bold")
    y_positions = np.arange(4)[::-1]
    for suite_index, suite in enumerate(data["suite_labels"]):
        ax = fig.add_axes([0.067 + suite_index * 0.245, 0.067, 0.190, 0.224])
        low, high = (60, 90) if suite_index < 3 else (20, 80)
        label_offset = (high - low) * 0.055
        ax.hlines(y_positions, low, high, colors="#E9E9E9", linewidth=0.65, zorder=1)
        for i, value in enumerate(rates[:, suite_index]):
            ax.scatter(value, y_positions[i], s=27, marker=markers[i], color=colors[i], zorder=3)
            ax.text(value + label_offset, y_positions[i], f"{value:.1f}", va="center", fontsize=8, color="#222222")
        ax.set_xlim(low, high + 0.17 * (high - low))
        ax.set_xticks(np.linspace(low, high, 4))
        ax.set_ylim(-0.55, 3.55)
        ax.set_yticks(y_positions, [view["label"] for view in data["views"]] if suite_index == 0 else [])
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", length=2.5, width=0.6, pad=2, labelsize=8)
        ax.set_title(suite, pad=5, fontsize=9.5, fontweight="bold")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#888888")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.45)
    fig.savefig(figure_dir / "Figure1_viewpoint_performance.pdf", dpi=600, facecolor="white")
    fig.savefig(figure_dir / "Figure1_viewpoint_performance.png", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
