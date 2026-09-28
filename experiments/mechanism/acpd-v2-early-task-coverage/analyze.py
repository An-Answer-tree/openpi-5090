"""Summarizes early ACPD-v2 task-level gains from paired LIBERO logs."""

import json
import pathlib

import numpy as np
from scipy.stats import spearmanr

from scripts.distillation_acpd.analyze_paired_eval import parse_episode_results

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PAIRED_RESULTS = {
    5000: _ROOT / "experiments/baseline/sft-backview-bs64-5k/results/h9_vs_h12_paired_analysis.json",
    10000: _ROOT / "experiments/student/acpd-v2-h9-early-trajectory/results/h9_vs_sft_10k_paired_analysis.json",
    15000: _ROOT / "experiments/student/acpd-v2-h9-early-trajectory/results/h9_vs_sft_15k_paired_analysis.json",
}
_SUITES = ("spatial", "object", "goal", "libero_10")
_EPISODES = set(range(50))


def _task_differences(paired_result: pathlib.Path) -> dict[str, float]:
    """Returns treatment-minus-baseline success for each matched task."""
    config = json.loads(paired_result.read_text())
    baseline_dir = pathlib.Path(config["baseline_dir"]) / "logs"
    treatment_dir = pathlib.Path(config["treatment_dir"]) / "logs"
    differences = {}
    for suite in _SUITES:
        baseline = parse_episode_results(baseline_dir / f"{suite}.log")
        treatment = parse_episode_results(treatment_dir / f"{suite}.log")
        if baseline.keys() != treatment.keys():
            raise ValueError(f"Unmatched episode keys in {suite}: {paired_result}")
        tasks = {task for task, _ in baseline}
        for task in tasks:
            episodes = {episode for name, episode in baseline if name == task}
            if episodes != _EPISODES:
                raise ValueError(f"Incomplete episodes for {suite}/{task}: {paired_result}")
            delta = sum(int(treatment[task, episode]) - int(baseline[task, episode]) for episode in _EPISODES)
            differences[f"{suite}/{task}"] = delta / len(_EPISODES)
    return differences


def _counts(differences: dict[str, float]) -> dict[str, int]:
    return {
        "positive": sum(value > 0 for value in differences.values()),
        "negative": sum(value < 0 for value in differences.values()),
        "zero": sum(value == 0 for value in differences.values()),
    }


def analyze() -> dict:
    """Compares task-level gains at 5K, 10K, and 15K."""
    by_step = {step: _task_differences(path) for step, path in _PAIRED_RESULTS.items()}
    keys = set(by_step[10000])
    if any(set(differences) != keys for differences in by_step.values()):
        raise ValueError("Task sets differ between checkpoints")

    steps = {}
    for step, differences in by_step.items():
        steps[str(step)] = {
            "pooled_difference": float(np.mean(list(differences.values()))),
            "all_tasks": _counts(differences),
            "suites": {
                suite: {
                    **_counts({key: value for key, value in differences.items() if key.startswith(f"{suite}/")}),
                    "difference": float(
                        np.mean([value for key, value in differences.items() if key.startswith(f"{suite}/")])
                    ),
                }
                for suite in _SUITES
            },
            "task_differences": dict(sorted(differences.items())),
        }

    early = by_step[10000]
    later = by_step[15000]
    ordered = sorted(keys)
    changes = {key: later[key] - early[key] for key in ordered}
    largest_change = max(changes, key=lambda key: abs(changes[key]))
    return {
        "steps": steps,
        "positive_10k_remaining_positive_15k": sum(early[key] > 0 and later[key] > 0 for key in keys),
        "positive_10k_becoming_nonpositive_15k": sum(early[key] > 0 and later[key] <= 0 for key in keys),
        "spearman_10k_15k": float(
            spearmanr([early[key] for key in ordered], [later[key] for key in ordered]).statistic
        ),
        "largest_change": {"task": largest_change, "difference": changes[largest_change]},
    }


def main() -> None:
    result = analyze()
    output = pathlib.Path(__file__).parent / "results/task_coverage.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for step, values in result["steps"].items():
        print(step, values["all_tasks"], f"pooled={values['pooled_difference']:+.2%}")
    print("10K positive retained at 15K:", result["positive_10k_remaining_positive_15k"])
    print("10K to 15K Spearman:", result["spearman_10k_15k"])


if __name__ == "__main__":
    main()
