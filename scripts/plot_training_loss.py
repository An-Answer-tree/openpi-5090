"""Plots training loss curves parsed from OpenPI Slurm logs."""

import argparse
import pathlib
import re

import matplotlib.pyplot as plt

loss_pattern = re.compile(r"Step (\d+):.*\bloss=([0-9.eE+-]+)")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-gpu-log", type=pathlib.Path, required=True)
    parser.add_argument("--four-gpu-log", type=pathlib.Path, required=True)
    parser.add_argument("--four-gpu-cosine-log", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def read_loss(log_path: pathlib.Path) -> tuple[list[int], list[float]]:
    """Reads logged training steps and losses."""
    matches = loss_pattern.findall(log_path.read_text())
    return [int(step) for step, _ in matches], [float(loss) for _, loss in matches]


def plot_loss(
    two_gpu_log: pathlib.Path,
    four_gpu_log: pathlib.Path,
    four_gpu_cosine_log: pathlib.Path,
    output: pathlib.Path,
) -> None:
    """Plots losses by optimization step and samples processed."""
    two_gpu_steps, two_gpu_losses = read_loss(two_gpu_log)
    four_gpu_steps, four_gpu_losses = read_loss(four_gpu_log)
    cosine_steps, cosine_losses = read_loss(four_gpu_cosine_log)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    series = (
        (two_gpu_steps, two_gpu_losses, 16, "2 GPUs, BS 16, default LR", "#0072B2", "-", "o"),
        (four_gpu_steps, four_gpu_losses, 32, "4 GPUs, BS 32, default LR", "#D55E00", "--", "s"),
        (cosine_steps, cosine_losses, 32, "4 GPUs, BS 32, cosine LR", "#009E73", "-.", "^"),
    )

    for steps, losses, batch_size, label, color, line_style, marker in series:
        plot_kwargs = {
            "color": color,
            "label": label,
            "linestyle": line_style,
            "linewidth": 1.4,
            "marker": marker,
            "markevery": 30,
            "markersize": 2.8,
        }
        axes[0].plot(steps, losses, **plot_kwargs)
        samples = [step * batch_size for step in steps]
        axes[1].plot(samples, losses, **plot_kwargs)

    axes[0].set_xlabel("Optimization step")
    axes[1].set_xlabel("Samples processed")
    axes[0].set_ylabel("Training loss")
    for axis in axes:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Runs the loss plotting command."""
    args = parse_args()
    plot_loss(args.two_gpu_log, args.four_gpu_log, args.four_gpu_cosine_log, args.output)


if __name__ == "__main__":
    main()
