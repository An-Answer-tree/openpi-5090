"""Compares paired H9 and SFT success over the 5K-30K training trajectory."""

import json
import pathlib

import numpy as np

from scripts.distillation_acpd.analyze_paired_eval import parse_episode_results

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PAIRS = {
    5000: "experiments/baseline/sft-backview-bs64-5k/results/h9_vs_h12_paired_analysis.json",
    10000: "experiments/student/acpd-v2-h9-early-trajectory/results/h9_vs_sft_10k_paired_analysis.json",
    15000: "experiments/student/acpd-v2-h9-early-trajectory/results/h9_vs_sft_15k_paired_analysis.json",
    20000: "experiments/student/acpd-v2-h9-early-trajectory/results/h9_vs_sft_20k_paired_analysis.json",
    25000: "experiments/baseline/sft-backview-bs64-5k/results/h9_vs_h12_25k_paired_analysis.json",
    30000: "experiments/baseline/sft-backview-bs64-5k/results/h9_vs_h12_30k_paired_analysis.json",
}
_SUITES = ("spatial", "object", "goal", "libero_10")
_STEPS = tuple(_PAIRS)
_TRIALS = tuple(range(50))
_RESAMPLES = 10_000


def _curve_average(values: np.ndarray, start: int = 0) -> np.ndarray:
    """Returns the trapezoidal mean for equally spaced checkpoints."""
    values = values[start:]
    weights = np.ones(len(values))
    weights[1:-1] = 2
    return np.average(values, axis=0, weights=weights)


def _load_outcomes() -> dict[str, np.ndarray]:
    """Loads paired per-task differences with shape [checkpoint, episode]."""
    by_step = {}
    for step, relative_path in _PAIRS.items():
        pair = json.loads((_ROOT / relative_path).read_text())
        baseline_dir = pathlib.Path(pair["baseline_dir"]) / "logs"
        treatment_dir = pathlib.Path(pair["treatment_dir"]) / "logs"
        tasks = {}
        for suite in _SUITES:
            baseline = parse_episode_results(baseline_dir / f"{suite}.log")
            treatment = parse_episode_results(treatment_dir / f"{suite}.log")
            if baseline.keys() != treatment.keys():
                raise ValueError(f"Unmatched episodes at {step} in {suite}")
            for task in {name for name, _ in baseline}:
                episodes = {episode for name, episode in baseline if name == task}
                if episodes != set(_TRIALS):
                    raise ValueError(f"Incomplete episodes at {step} for {suite}/{task}")
                tasks[f"{suite}/{task}"] = np.array(
                    [int(treatment[task, episode]) - int(baseline[task, episode]) for episode in _TRIALS],
                    dtype=np.int8,
                )
        observed = np.mean(np.stack(list(tasks.values())))
        if not np.isclose(observed, pair["pooled"]["difference"]):
            raise ValueError(f"Pooled success difference disagrees with {relative_path}")
        by_step[step] = tasks

    task_names = set(by_step[_STEPS[0]])
    if len(task_names) != 40 or any(set(tasks) != task_names for tasks in by_step.values()):
        raise ValueError("Task sets differ between checkpoints")
    return {task: np.stack([by_step[step][task] for step in _STEPS]) for task in sorted(task_names)}


def _summarize(samples: np.ndarray, observed: float) -> dict[str, float | list[float]]:
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return {"difference": observed, "ci95": [float(lower), float(upper)]}


def analyze() -> dict:
    """Computes task-stratified paired learning-curve area differences."""
    outcomes = _load_outcomes()
    rng = np.random.default_rng(42)
    task_areas = {}
    sampled_areas = {suite: np.zeros(_RESAMPLES) for suite in _SUITES}
    sampled_late_areas = {suite: np.zeros(_RESAMPLES) for suite in _SUITES}

    for task, differences in outcomes.items():
        suite = task.split("/", maxsplit=1)[0]
        task_areas[task] = float(_curve_average(differences.mean(axis=1)))
        indices = rng.integers(len(_TRIALS), size=(_RESAMPLES, len(_TRIALS)))
        sampled = np.take(differences, indices, axis=1).mean(axis=-1)
        sampled_areas[suite] += _curve_average(sampled)
        sampled_late_areas[suite] += _curve_average(sampled, start=1)

    suite_results = {}
    for suite in _SUITES:
        areas = [value for task, value in task_areas.items() if task.startswith(f"{suite}/")]
        suite_results[suite] = _summarize(sampled_areas[suite] / 10, float(np.mean(areas)))

    pooled_samples = sum(sampled_areas.values()) / len(outcomes)
    late_samples = sum(sampled_late_areas.values()) / len(outcomes)
    return {
        "steps": _STEPS,
        "tasks": len(outcomes),
        "episodes_per_task_per_step": len(_TRIALS),
        "bootstrap_resamples": _RESAMPLES,
        "bootstrap_seed": 42,
        "pooled_5k_30k": _summarize(pooled_samples, float(np.mean(list(task_areas.values())))),
        "pooled_10k_30k": _summarize(
            late_samples,
            float(np.mean([_curve_average(values.mean(axis=1), start=1) for values in outcomes.values()])),
        ),
        "suites_5k_30k": suite_results,
        "task_area_counts": {
            "positive": sum(value > 0 for value in task_areas.values()),
            "negative": sum(value < 0 for value in task_areas.values()),
            "zero": sum(value == 0 for value in task_areas.values()),
        },
        "task_areas": task_areas,
        "input_pairs": list(_PAIRS.values()),
    }


def main() -> None:
    result = analyze()
    output = pathlib.Path(__file__).parent / "results/learning_curve_area.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in ("pooled_5k_30k", "pooled_10k_30k", "task_area_counts")}, indent=2))


if __name__ == "__main__":
    main()
