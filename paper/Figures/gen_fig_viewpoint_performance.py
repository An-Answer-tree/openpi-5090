# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.3", "numpy==1.26.4", "pillow>=10"]
# ///
"""Draw Figure 1 from archived single-view SFT evaluation counts.

Run with `uv run --script paper/Figures/gen_fig_viewpoint_performance.py`.
Alternatively, run with the existing openpi Conda environment's Python.
The adjacent JSON records the plotted counts and their archive sources.
The four images show one synchronized state, re-rendered at 1536 x 1536.
The compact image grid identifies the views; suite-level bar plots show
their performance differences, with explicitly marked cropped baselines.
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
    fig.text(0.1645, 0.99, "(a) Camera views", ha="center", va="top", fontsize=9, fontweight="bold")
    image_width = 0.152
    image_height = image_width * data["width_cm"] / data["height_cm"]
    for i, view in enumerate(data["views"]):
        left = 0.005 + (i % 2) * 0.167
        bottom = 0.529 if i < 2 else 0.087
        ax = fig.add_axes([left, bottom, image_width, image_height])
        with Image.open(figure_dir / "Figure1_views" / f"{view['label'].lower()}view.png") as frame:
            ax.imshow(frame, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(colors[i])
            spine.set_linewidth(0.8)
        fig.text(
            left + image_width / 2,
            bottom - 0.015,
            view["label"],
            ha="center",
            va="top",
            color=colors[i],
            fontweight="bold",
            fontsize=8.5,
        )

    fig.text(0.694, 0.99, "(b) Single-view SFT success rate (%)", ha="center", va="top", fontsize=9, fontweight="bold")
    for suite_index, suite in enumerate(data["suite_labels"]):
        left = 0.395 + (suite_index % 2) * 0.325
        bottom = 0.569 if suite_index < 2 else 0.116
        ax = fig.add_axes([left, bottom, 0.273, 0.292])
        low, high = (60, 90) if suite_index < 3 else (20, 80)
        for i, value in enumerate(rates[:, suite_index]):
            bars = ax.bar(i, value, width=0.68, color=colors[i], edgecolor="white", linewidth=0.45, zorder=3)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, color="#222222")
        ax.set_xlim(-0.65, 3.65)
        ax.set_xticks(np.arange(4), [view["label"] for view in data["views"]], fontsize=8)
        ax.set_ylim(low, high)
        ax.set_yticks(np.linspace(low, high, 4))
        ax.tick_params(axis="y", length=2.5, pad=3, width=0.6)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_title(suite, pad=6, fontsize=9.5, fontweight="bold")
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.45)
        ax.plot(
            [0],
            [0],
            transform=ax.transAxes,
            marker=[(-1, -0.5), (1, 0.5)],
            markersize=6,
            markeredgewidth=0.8,
            color="#333333",
            linestyle="none",
            clip_on=False,
            zorder=5,
        )
    fig.savefig(figure_dir / "Figure1_viewpoint_performance.pdf", dpi=600, facecolor="white")
    fig.savefig(figure_dir / "Figure1_viewpoint_performance.png", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
