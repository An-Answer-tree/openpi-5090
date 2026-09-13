import pathlib

import numpy as np
import pytest

from scripts.distillation_acpd import analyze_paired_eval


def _write_log(path: pathlib.Path, tasks: dict[str, list[bool]]) -> None:
    lines = []
    for task, successes in tasks.items():
        for episode, success in enumerate(successes):
            lines.extend(
                (
                    f"INFO:root:Task: {task}; episode: {episode}",
                    f"INFO:root:Success: {success}",
                )
            )
    path.write_text("\n".join(lines) + "\n")


def test_summarize_paired_results(tmp_path: pathlib.Path) -> None:
    baseline_path = tmp_path / "baseline.log"
    treatment_path = tmp_path / "treatment.log"
    _write_log(baseline_path, {"task a": [False, True, False], "task b": [True, False, False]})
    _write_log(treatment_path, {"task a": [True, True, False], "task b": [True, True, False]})

    metrics = analyze_paired_eval.summarize_paired_results(
        analyze_paired_eval.parse_episode_results(baseline_path),
        analyze_paired_eval.parse_episode_results(treatment_path),
        bootstrap_samples=1_000,
        expected_trials_per_task=3,
        rng=np.random.default_rng(42),
    )

    assert metrics["tasks"] == 2
    assert metrics["episodes"] == 6
    assert metrics["baseline_success"] == pytest.approx(2 / 6)
    assert metrics["treatment_success"] == pytest.approx(4 / 6)
    assert metrics["difference"] == pytest.approx(2 / 6)
    assert metrics["ci95_lower"] >= 0.0


def test_rejects_unpaired_episodes(tmp_path: pathlib.Path) -> None:
    baseline_path = tmp_path / "baseline.log"
    treatment_path = tmp_path / "treatment.log"
    _write_log(baseline_path, {"task": [False, True]})
    _write_log(treatment_path, {"task": [True]})

    with pytest.raises(ValueError, match="episode keys do not match"):
        analyze_paired_eval.summarize_paired_results(
            analyze_paired_eval.parse_episode_results(baseline_path),
            analyze_paired_eval.parse_episode_results(treatment_path),
            bootstrap_samples=10,
            expected_trials_per_task=2,
            rng=np.random.default_rng(42),
        )
