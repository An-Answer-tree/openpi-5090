"""Compares two LIBERO evaluations on paired task initial states."""

import dataclasses
import json
import pathlib
import re

import numpy as np
import tyro

_TASK_PATTERN = re.compile(r"INFO:root:Task: (.+); episode: (\d+)")
_SUCCESS_PATTERN = re.compile(r"INFO:root:Success: (True|False)")
_SUITE_LOGS = {
    "spatial": "spatial.log",
    "object": "object.log",
    "goal": "goal.log",
    "libero_10": "libero_10.log",
}


@dataclasses.dataclass(frozen=True)
class Args:
    """Arguments for paired LIBERO evaluation analysis."""

    baseline_dir: pathlib.Path
    treatment_dir: pathlib.Path
    output: pathlib.Path
    bootstrap_samples: int = 10_000
    seed: int = 42
    expected_trials_per_task: int = 50


EpisodeKey = tuple[str, int]


def parse_episode_results(log_path: pathlib.Path) -> dict[EpisodeKey, bool]:
    """Parses task, episode, and success triples from one evaluator log."""
    results = {}
    episode_key = None
    for line in log_path.read_text(errors="replace").splitlines():
        if match := _TASK_PATTERN.search(line):
            episode_key = (match.group(1), int(match.group(2)))
        elif match := _SUCCESS_PATTERN.search(line):
            if episode_key is None:
                raise ValueError(f"Success without a preceding episode in {log_path}")
            if episode_key in results:
                raise ValueError(f"Duplicate episode {episode_key} in {log_path}")
            results[episode_key] = match.group(1) == "True"
            episode_key = None
    if episode_key is not None:
        raise ValueError(f"Missing success result for {episode_key} in {log_path}")
    return results


def _group_differences(
    baseline: dict[EpisodeKey, bool],
    treatment: dict[EpisodeKey, bool],
    expected_trials_per_task: int,
) -> list[np.ndarray]:
    if baseline.keys() != treatment.keys():
        raise ValueError("Baseline and treatment episode keys do not match")

    task_differences = {}
    for key in sorted(baseline):
        task_name, episode = key
        task_differences.setdefault(task_name, {})[episode] = int(treatment[key]) - int(baseline[key])

    grouped = []
    expected_episodes = set(range(expected_trials_per_task))
    for task_name, differences in task_differences.items():
        if differences.keys() != expected_episodes:
            raise ValueError(f"Task {task_name!r} does not contain episodes 0-{expected_trials_per_task - 1}")
        grouped.append(np.array([differences[index] for index in range(expected_trials_per_task)]))
    return grouped


def summarize_paired_results(
    baseline: dict[EpisodeKey, bool],
    treatment: dict[EpisodeKey, bool],
    *,
    bootstrap_samples: int,
    expected_trials_per_task: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Computes paired success difference and a task-stratified interval."""
    grouped = _group_differences(baseline, treatment, expected_trials_per_task)
    bootstrap_totals = np.zeros(bootstrap_samples)
    for differences in grouped:
        indices = rng.integers(
            len(differences),
            size=(bootstrap_samples, len(differences)),
        )
        bootstrap_totals += differences[indices].sum(axis=1)

    num_episodes = sum(len(differences) for differences in grouped)
    difference = (sum(treatment.values()) - sum(baseline.values())) / num_episodes
    lower, upper = np.percentile(bootstrap_totals / num_episodes, [2.5, 97.5])
    return {
        "tasks": len(grouped),
        "episodes": num_episodes,
        "baseline_success": sum(baseline.values()) / num_episodes,
        "treatment_success": sum(treatment.values()) / num_episodes,
        "difference": difference,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def analyze(args: Args) -> dict:
    """Analyzes every suite and the pooled paired results."""
    rng = np.random.default_rng(args.seed)
    all_baseline = {}
    all_treatment = {}
    suite_metrics = {}
    for suite, filename in _SUITE_LOGS.items():
        baseline = parse_episode_results(args.baseline_dir / "logs" / filename)
        treatment = parse_episode_results(args.treatment_dir / "logs" / filename)
        suite_metrics[suite] = summarize_paired_results(
            baseline,
            treatment,
            bootstrap_samples=args.bootstrap_samples,
            expected_trials_per_task=args.expected_trials_per_task,
            rng=rng,
        )
        all_baseline.update({(f"{suite}/{task}", episode): success for (task, episode), success in baseline.items()})
        all_treatment.update({(f"{suite}/{task}", episode): success for (task, episode), success in treatment.items()})

    return {
        "baseline_dir": str(args.baseline_dir),
        "treatment_dir": str(args.treatment_dir),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "suites": suite_metrics,
        "pooled": summarize_paired_results(
            all_baseline,
            all_treatment,
            bootstrap_samples=args.bootstrap_samples,
            expected_trials_per_task=args.expected_trials_per_task,
            rng=rng,
        ),
    }


def _print_table(metrics: dict) -> None:
    print("| Suite | Baseline | Full ACPD | Difference | Paired 95% CI |")
    print("|---|---:|---:|---:|---:|")
    rows = {**metrics["suites"], "pooled": metrics["pooled"]}
    for suite, values in rows.items():
        print(
            f"| {suite} | {values['baseline_success']:.2%} | {values['treatment_success']:.2%} | "
            f"{values['difference']:+.2%} | [{values['ci95_lower']:+.2%}, {values['ci95_upper']:+.2%}] |"
        )


def main(args: Args) -> None:
    metrics = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    _print_table(metrics)


if __name__ == "__main__":
    main(tyro.cli(Args))
