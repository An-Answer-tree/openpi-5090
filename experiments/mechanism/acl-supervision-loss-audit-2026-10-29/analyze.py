"""Audit existing ACL-only/SFT logs and archived ACL gradient diagnostics."""

import csv
import json
import math
from pathlib import Path
import re
import statistics

REPOSITORY = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "results"
SFT_LOGS = [
    REPOSITORY / "slurm-log/pi05-bv-sft-bs64-5k_130285.out",
    REPOSITORY / "slurm-log/pi05-bv-sft-bs64-r30k_130762.out",
    REPOSITORY / "slurm-log/pi05-bv-sft-bs64-r60k_132791.out",
]
ACL_LOG = Path(
    "/opt/liutong/openpi-5090-research/acpd-v2-acl-only-bs64-35k/slurm-log/pi05-bv-acl-only-bs64-35k_135724.out"
)
STEP = re.compile(r"Step (\d+):([^\r\n]*)")
METRIC = re.compile(r"([a-z0-9_]+)=([0-9.eE+-]+)")


def read_metrics(path: Path) -> dict[int, dict[str, float]]:
    """Read logged metrics without traversing the surrounding log directory."""
    return {
        int(match[1]): {key: float(value) for key, value in METRIC.findall(match[2])}
        for match in STEP.finditer(path.read_text(errors="replace"))
    }


def main() -> None:
    """Summarize training observations without launching new model evaluations."""
    sft = {}
    for path in SFT_LOGS:
        sft.update(read_metrics(path))
    acl = read_metrics(ACL_LOG)
    steps = sorted(step for step in set(sft) & set(acl) if 100 <= step < 35_000)
    rows = [
        {
            "step": step,
            "sft_flow_loss": sft[step]["loss"],
            "acl_only_flow_loss": acl[step]["supervised_loss"],
            "acl_loss": acl[step]["action_corr_loss"],
            "teacher_student_correlation": 1 - acl[step]["action_corr_loss"],
        }
        for step in steps
    ]
    windows = []
    for start in range(0, 35_000, 5_000):
        selected = [row for row in rows if start <= row["step"] < start + 5_000]
        means = {key: statistics.fmean(row[key] for row in selected) for key in rows[0] if key != "step"}
        windows.append(
            {
                "start_step": start,
                "end_step_exclusive": start + 5_000,
                "logged_points": len(selected),
                **means,
                "flow_relative_difference_percent": 100 * (means["acl_only_flow_loss"] / means["sft_flow_loss"] - 1),
            }
        )
    gradient_root = Path("/opt/liutong/openpi-5090-research/acpd-v2-gradient-conflict")
    gradients = {}
    for checkpoint in ["h9-5k", "h9-30k"]:
        path = gradient_root / checkpoint / "summary.json"
        data = json.loads(path.read_text())
        raw = [json.loads(line) for line in (path.parent / "per_batch.jsonl").read_text().splitlines()]
        total_cosines = []
        pcd_acl_cosines = []
        dot_errors = []
        for row in raw:
            ratio, cosine = row["privileged_norm_ratio"], row["privileged_cosine"]
            total_cosines.append((1 + ratio * cosine) / math.sqrt(1 + ratio * ratio + 2 * ratio * cosine))
            cue, action, combined = row["contribution_norm"], row["action_corr_norm"], row["privileged_norm"]
            pcd_acl_cosines.append((combined**2 - cue**2 - action**2) / (2 * cue * action))
            dot_errors.append(
                abs(
                    ratio * cosine
                    - row["contribution_norm_ratio"] * row["contribution_cosine"]
                    - row["action_corr_norm_ratio"] * row["action_corr_cosine"]
                )
            )
        assert max(dot_errors) < 1e-4, "Archived weighted gradient sums failed the dot-product identity."
        gradients[checkpoint] = {
            "source": str(path),
            "student_params": data["student_params"],
            "batch_size": data["batch_size"],
            "num_batches": data["num_batches"],
            "components": {
                name: {
                    "flow_cosine_median": data["metrics"][f"{key}_cosine"]["median"],
                    "flow_norm_ratio_median": data["metrics"][f"{key}_norm_ratio"]["median"],
                    "observed_conflict_fraction": data["metrics"][f"{key}_conflict"]["mean"],
                }
                for name, key in [("PCD", "contribution"), ("ACL", "action_corr"), ("combined_auxiliary", "privileged")]
            },
            "derived_total_flow_cosine_median": statistics.median(total_cosines),
            "derived_pcd_acl_cosine_median": statistics.median(pcd_acl_cosines),
            "weighted_gradient_sum_dot_identity_max_error": max(dot_errors),
        }
    sft_mean = statistics.fmean(row["sft_flow_loss"] for row in rows)
    acl_mean = statistics.fmean(row["acl_only_flow_loss"] for row in rows)
    summary = {
        "study_type": "Exploratory reanalysis of existing training logs and gradient diagnostics; no new training or evaluation.",
        "sources": {"sft_logs": list(map(str, SFT_LOGS)), "acl_only_log": str(ACL_LOG)},
        "matched_logged_points": len(rows),
        "flow_loss_correlation": statistics.correlation(
            [row["sft_flow_loss"] for row in rows], [row["acl_only_flow_loss"] for row in rows]
        ),
        "sft_mean_flow_loss": sft_mean,
        "acl_only_mean_flow_loss": acl_mean,
        "mean_flow_relative_difference_percent": 100 * (acl_mean / sft_mean - 1),
        "acl_loss_reduction_between_endpoint_windows_percent": 100
        * (1 - windows[-1]["acl_loss"] / windows[0]["acl_loss"]),
        "windows": windows,
        "archived_full_acpd_gradient_diagnostics": gradients,
        "interpretation_limits": [
            "Training losses are recorded on training batches, rounded to four decimals, and aligned by optimizer step, not reevaluated on held-out paired samples.",
            "SFT did not log teacher/student correlations; within-run ACL reduction does not establish an improvement over SFT on this metric.",
            "Gradient diagnostics measure individual loss gradients at full ACPD checkpoints with 200 BS32 probe batches, not the ACL-only training trajectory.",
            "No claim of faster convergence, improved generalization, or causal explanation of success is established by these observations.",
        ],
    }
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "matched_losses.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ["windows", "sources"]}, indent=2))


if __name__ == "__main__":
    main()
