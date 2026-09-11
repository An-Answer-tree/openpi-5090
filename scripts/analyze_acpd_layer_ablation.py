"""Analyzes the ACPD layer-ablation training logs."""

import argparse
import csv
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np

step_pattern = re.compile(r"Step (\d+): (.*)")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="Run as LABEL=LOG_PATH.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    return parser.parse_args()


def read_metrics(log_path: pathlib.Path) -> list[dict[str, float]]:
    """Reads step metrics from an OpenPI Slurm log."""
    metrics = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = step_pattern.search(line)
        if match is None:
            continue
        row = {"step": float(match.group(1))}
        for item in match.group(2).split(", "):
            name, value = item.split("=", maxsplit=1)
            row[name] = float(value)
        metrics.append(row)
    return metrics


def summarize(label: str, metrics: list[dict[str, float]]) -> dict[str, float | str]:
    """Computes the pre-registered convergence metrics."""
    primary = [row["supervised_loss"] for row in metrics if 100 <= row["step"] <= 4_900]
    final = [row["supervised_loss"] for row in metrics if 4_000 <= row["step"] <= 4_900]
    return {
        "run": label,
        "logged_points": len(metrics),
        "last_step": int(metrics[-1]["step"]),
        "supervised_loss_mean_100_4900": float(np.mean(primary)),
        "supervised_loss_mean_4000_4900": float(np.mean(final)),
        "final_total_loss": metrics[-1]["loss"],
        "final_action_corr_loss": metrics[-1]["action_corr_loss"],
        "final_acpd_prediction_loss": metrics[-1]["acpd_prediction_loss"],
        "final_acpd_variance_loss": metrics[-1]["acpd_variance_loss"],
    }


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    """Writes rows using the union of their keys."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_supervised_loss(runs: dict[str, list[dict[str, float]]], output_dir: pathlib.Path) -> None:
    """Plots supervised loss against optimizer step."""
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axis = plt.subplots(figsize=(6.4, 3.6))
    for (label, metrics), color in zip(runs.items(), colors, strict=True):
        axis.plot(
            [row["step"] for row in metrics],
            [row["supervised_loss"] for row in metrics],
            label=label,
            color=color,
            linewidth=1.4,
        )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Supervised flow loss")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "supervised_loss.png", dpi=220)
    fig.savefig(output_dir / "supervised_loss.pdf")
    plt.close(fig)


def main() -> None:
    """Runs layer-ablation analysis."""
    args = parse_args()
    runs = {}
    for run in args.run:
        label, log_path = run.split("=", maxsplit=1)
        runs[label] = read_metrics(pathlib.Path(log_path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = [{"run": label, **row} for label, metrics in runs.items() for row in metrics]
    write_csv(args.output_dir / "metrics.csv", raw_rows)
    write_csv(args.output_dir / "summary.csv", [summarize(label, metrics) for label, metrics in runs.items()])
    plot_supervised_loss(runs, args.output_dir)


if __name__ == "__main__":
    main()
