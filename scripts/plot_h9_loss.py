"""Plots H9 ACPD-v2 training metrics from an OpenPI Slurm log."""

import argparse
import pathlib
import re

import matplotlib.pyplot as plt


STEP_PATTERN = re.compile(r"Step (?P<step>\d+):(?P<metrics>.*)")
METRIC_PATTERN = re.compile(r"(?P<name>[a-z0-9_]+)=(?P<value>[0-9.eE+-]+)")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def read_metrics(log_path: pathlib.Path) -> tuple[list[int], dict[str, list[float]]]:
    """Reads step metrics from a Slurm log."""
    steps = []
    records = []
    for line in log_path.read_text(errors="replace").splitlines():
        step_match = STEP_PATTERN.search(line)
        if step_match is None:
            continue
        metrics = {
            match.group("name"): float(match.group("value"))
            for match in METRIC_PATTERN.finditer(step_match.group("metrics"))
        }
        steps.append(int(step_match.group("step")))
        records.append(metrics)

    names = sorted({name for record in records for name in record})
    values = {name: [record[name] for record in records] for name in names}
    return steps, values


def plot_h9_loss(log_path: pathlib.Path, output: pathlib.Path) -> None:
    """Plots the H9 primary and auxiliary training curves."""
    steps, values = read_metrics(log_path)
    colors = {
        "loss": "#0072B2",
        "supervised_loss": "#D55E00",
        "student_task_loss": "#009E73",
        "teacher_task_loss": "#CC79A7",
        "acpd_loss": "#E69F00",
        "weighted_acpd_loss": "#D55E00",
        "action_corr_loss": "#56B4E9",
        "exact_contribution_cosine": "#009E73",
        "exact_contribution_gate": "#CC79A7",
    }

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharex=True)
    panels = (
        (
            axes[0],
            "Primary losses",
            "Loss",
            ("loss", "supervised_loss", "student_task_loss", "teacher_task_loss"),
        ),
        (
            axes[1],
            "Auxiliary losses",
            "Loss",
            ("acpd_loss", "weighted_acpd_loss", "action_corr_loss"),
        ),
        (
            axes[2],
            "Contribution signal",
            "Value",
            ("exact_contribution_cosine", "exact_contribution_gate"),
        ),
    )

    for axis, title, ylabel, names in panels:
        for name in names:
            axis.plot(
                steps,
                values[name],
                color=colors[name],
                linewidth=1.4,
                marker="o",
                markersize=2.2,
                markevery=5,
                label=name.replace("_", " "),
            )
        axis.set_title(title)
        axis.set_xlabel("Optimization step")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)

    fig.suptitle("H9 ACPD-v2 training curves (physical global BS32)", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Runs the H9 loss plotting command."""
    args = parse_args()
    plot_h9_loss(args.log, args.output)


if __name__ == "__main__":
    main()
