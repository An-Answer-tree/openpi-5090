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


def _task_episode_differences(paired_result: pathlib.Path) -> dict[str, np.ndarray]:
    """Returns paired treatment-minus-baseline outcomes for each task."""
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
            differences[f"{suite}/{task}"] = np.array(
                [int(treatment[task, episode]) - int(baseline[task, episode]) for episode in sorted(_EPISODES)],
                dtype=np.int8,
            )
    return differences


def _counts(differences: dict[str, float]) -> dict[str, int]:
    return {
        "positive": sum(value > 0 for value in differences.values()),
        "negative": sum(value < 0 for value in differences.values()),
        "zero": sum(value == 0 for value in differences.values()),
    }


def _bootstrap_coverage(by_step: dict[int, dict[str, np.ndarray]], tasks: list[str]) -> dict:
    """Resamples matched episode indices across checkpoints."""
    steps = (5000, 10000, 15000)
    outcomes = np.stack([[by_step[step][task] for task in tasks] for step in steps])
    rng = np.random.default_rng(42)
    indices = rng.integers(len(_EPISODES), size=(2_000, len(tasks), len(_EPISODES)))
    sampled = np.take_along_axis(outcomes[:, None], indices[None], axis=-1).mean(axis=-1)
    positive = sampled > 0

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.percentile(values, [2.5, 97.5])]

    return {
        "seed": 42,
        "resamples": 2_000,
        "positive_task_count_ci95": {
            str(step): interval(positive[index].sum(axis=-1)) for index, step in enumerate(steps)
        },
        "positive_10k_retained_15k_ci95": interval((positive[1] & positive[2]).sum(axis=-1)),
    }


def analyze() -> dict:
    """Compares task-level gains at 5K, 10K, and 15K."""
    outcomes = {step: _task_episode_differences(path) for step, path in _PAIRED_RESULTS.items()}
    by_step = {
        step: {task: float(np.mean(values)) for task, values in task_outcomes.items()}
        for step, task_outcomes in outcomes.items()
    }
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
        "bootstrap": _bootstrap_coverage(outcomes, ordered),
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
