"""Compares matched SFT and ACPD-v2 training losses."""

import argparse
import json
import pathlib
import re
import statistics

import matplotlib.pyplot as plt

STEP_PATTERN = re.compile(r"Step (?P<step>\d+):(?P<metrics>.*)")
METRIC_PATTERN = re.compile(r"(?P<name>[a-z0-9_]+)=(?P<value>[0-9.eE+-]+)")
WINDOWS = (
    ("0-5K", 100, 4_900),
    ("5K-10K", 5_000, 9_900),
    ("10K-15K", 10_000, 14_900),
    ("15K-20K", 15_000, 19_900),
    ("20K-25K", 20_000, 24_900),
    ("25K-30K", 25_000, 29_900),
)
ANCHOR_STEPS = (100, 1_000, 5_000, 10_000, 15_000, 20_000, 25_000, 29_900)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-log", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--acpd-log", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    return parser.parse_args()


def read_metrics(log_path: pathlib.Path) -> dict[int, dict[str, float]]:
    """Reads step-indexed metrics from an OpenPI Slurm log."""
    records = {}
    for line in log_path.read_text(errors="replace").splitlines():
        step_match = STEP_PATTERN.search(line)
        if step_match is None:
            continue
        records[int(step_match.group("step"))] = {
            match.group("name"): float(match.group("value"))
            for match in METRIC_PATTERN.finditer(step_match.group("metrics"))
        }
    return records


def merge_metrics(log_paths: list[pathlib.Path]) -> dict[int, dict[str, float]]:
    """Merges consecutive logs, preferring later logs at duplicate steps."""
    records = {}
    for log_path in log_paths:
        records.update(read_metrics(log_path))
    return records


def mean_metric(
    records: dict[int, dict[str, float]],
    metric: str,
    start: int,
    end: int,
) -> float:
    """Returns a metric mean over an inclusive step window."""
    values = [row[metric] for step, row in records.items() if start <= step <= end]
    return statistics.fmean(values)


def moving_average(values: list[float], radius: int = 3) -> list[float]:
    """Returns a centered moving average without changing sequence length."""
    return [statistics.fmean(values[max(0, index - radius) : index + radius + 1]) for index in range(len(values))]


def build_summary(
    sft: dict[int, dict[str, float]],
    acpd: dict[int, dict[str, float]],
) -> dict[str, dict | list]:
    """Builds matched window and anchor summaries."""
    windows = []
    for label, start, end in WINDOWS:
        sft_loss = mean_metric(sft, "loss", start, end)
        acpd_supervised = mean_metric(acpd, "supervised_loss", start, end)
        acpd_total = mean_metric(acpd, "loss", start, end)
        acpd_contribution = mean_metric(acpd, "weighted_acpd_loss", start, end)
        acpd_acl = acpd_total - acpd_supervised - acpd_contribution
        windows.append(
            {
                "window": label,
                "sft_supervised_loss": sft_loss,
                "acpd_supervised_loss": acpd_supervised,
                "acpd_minus_sft_percent": 100.0 * (acpd_supervised / sft_loss - 1.0),
                "acpd_total_loss": acpd_total,
                "acpd_weighted_contribution_loss": acpd_contribution,
                "acpd_weighted_acl_loss": acpd_acl,
                "acpd_contribution_cosine": mean_metric(acpd, "exact_contribution_cosine", start, end),
                "acpd_gate": mean_metric(acpd, "exact_contribution_gate", start, end),
            }
        )

    anchors = []
    for step in ANCHOR_STEPS:
        sft_loss = sft[step]["loss"]
        acpd_supervised = acpd[step]["supervised_loss"]
        anchors.append(
            {
                "step": step,
                "sft_supervised_loss": sft_loss,
                "acpd_supervised_loss": acpd_supervised,
                "acpd_minus_sft_percent": 100.0 * (acpd_supervised / sft_loss - 1.0),
                "acpd_contribution_cosine": acpd[step]["exact_contribution_cosine"],
                "acpd_gate": acpd[step]["exact_contribution_gate"],
            }
        )

    matched_steps = [step for step in sorted(set(sft) & set(acpd)) if 100 <= step <= 29_900]
    sft_loss = [sft[step]["loss"] for step in matched_steps]
    acpd_supervised = [acpd[step]["supervised_loss"] for step in matched_steps]
    relative_differences = [
        100.0 * (acpd_value / sft_value - 1.0) for acpd_value, sft_value in zip(acpd_supervised, sft_loss, strict=True)
    ]
    gates = [acpd[step]["exact_contribution_gate"] for step in matched_steps]
    peak_gate_index = max(range(len(gates)), key=gates.__getitem__)
    final_step = matched_steps[-1]
    final_supervised = acpd[final_step]["supervised_loss"]
    final_contribution = acpd[final_step]["weighted_acpd_loss"]
    final_acl = acpd[final_step]["loss"] - final_supervised - final_contribution
    overall = {
        "matched_points": len(matched_steps),
        "supervised_loss_correlation": statistics.correlation(sft_loss, acpd_supervised),
        "mean_sft_supervised_loss": statistics.fmean(sft_loss),
        "mean_acpd_supervised_loss": statistics.fmean(acpd_supervised),
        "mean_acpd_minus_sft_percent": statistics.fmean(relative_differences),
        "mean_absolute_acpd_minus_sft_percent": statistics.fmean(map(abs, relative_differences)),
        "peak_gate_step": matched_steps[peak_gate_index],
        "peak_gate": gates[peak_gate_index],
        "final_step": final_step,
        "final_contribution_cosine": acpd[final_step]["exact_contribution_cosine"],
        "final_gate": acpd[final_step]["exact_contribution_gate"],
        "final_objective_fractions": {
            "supervised": final_supervised / acpd[final_step]["loss"],
            "weighted_contribution": final_contribution / acpd[final_step]["loss"],
            "weighted_acl": final_acl / acpd[final_step]["loss"],
        },
    }
    return {"overall": overall, "windows": windows, "anchors": anchors}


def plot_comparison(
    sft: dict[int, dict[str, float]],
    acpd: dict[int, dict[str, float]],
    output: pathlib.Path,
) -> None:
    """Plots matched primary losses and ACPD-v2 auxiliary diagnostics."""
    steps = sorted(set(sft) & set(acpd))
    sft_loss = [sft[step]["loss"] for step in steps]
    acpd_supervised = [acpd[step]["supervised_loss"] for step in steps]
    relative_difference = [100.0 * (a / b - 1.0) for a, b in zip(acpd_supervised, sft_loss, strict=True)]
    acpd_contribution = [acpd[step]["weighted_acpd_loss"] for step in steps]
    acpd_acl = [acpd[step]["loss"] - acpd[step]["supervised_loss"] - acpd[step]["weighted_acpd_loss"] for step in steps]
    contribution_cosine = [acpd[step]["exact_contribution_cosine"] for step in steps]
    gate = [acpd[step]["exact_contribution_gate"] for step in steps]

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.4), sharex=True)

    axes[0, 0].plot(steps, moving_average(sft_loss), color="#0072B2", label="SFT supervised")
    axes[0, 0].plot(steps, moving_average(acpd_supervised), color="#D55E00", label="ACPD-v2 supervised")
    axes[0, 0].set_title("Matched supervised flow loss")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[0, 1].plot(steps, moving_average(relative_difference), color="#009E73")
    axes[0, 1].set_title("ACPD-v2 relative to SFT")
    axes[0, 1].set_ylabel("Supervised-loss difference (%)")

    axes[1, 0].plot(steps, moving_average(acpd_supervised), color="#0072B2", label="Supervised")
    axes[1, 0].plot(steps, moving_average(acpd_contribution), color="#E69F00", label="0.2 x contribution")
    axes[1, 0].plot(steps, moving_average(acpd_acl), color="#CC79A7", label="0.5 x ACL")
    axes[1, 0].set_title("ACPD-v2 objective components")
    axes[1, 0].set_xlabel("Optimizer step")
    axes[1, 0].set_ylabel("Weighted loss")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(steps, moving_average(contribution_cosine), color="#009E73", label="Contribution cosine")
    axes[1, 1].set_title("Contribution learning and injection")
    axes[1, 1].set_xlabel("Optimizer step")
    axes[1, 1].set_ylabel("Cosine")
    gate_axis = axes[1, 1].twinx()
    gate_axis.plot(steps, moving_average(gate), color="#CC79A7", label="Injection gate")
    gate_axis.set_ylabel("tanh gate")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    gate_handles, gate_labels = gate_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles + gate_handles, labels + gate_labels, frameon=False, loc="lower right")

    for axis in axes.flat:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Runs the matched loss comparison."""
    args = parse_args()
    sft = merge_metrics(args.sft_log)
    acpd = read_metrics(args.acpd_log)
    summary = build_summary(sft, acpd)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    plot_comparison(sft, acpd, args.output)


if __name__ == "__main__":
    main()
