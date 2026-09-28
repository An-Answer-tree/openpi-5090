"""Tests the oracle gate reliability analysis."""

import numpy as np

from scripts.distillation_acpd.analyze_gate_reliability import analyze_reliability


def test_aligned_and_misaligned_groups_have_expected_direction():
    errors = np.array(
        [
            [1.0, 0.0, 0.0, 0.5],
            [2.0, 0.0, 0.0, 1.5],
            [1.0, 0.0, 0.0, 1.5],
            [2.0, 0.0, 0.0, 2.5],
        ]
    )
    metrics = np.array([[0.1, 0.5], [0.1, 0.5], [0.1, -0.5], [0.1, -0.5]])
    result = analyze_reliability(errors, metrics, np.array([1, 1, 2, 2]))

    aligned = result["episode_groups"]["oracle_aligned_cosine_ge_0"]
    misaligned = result["episode_groups"]["oracle_misaligned_cosine_lt_0"]
    assert aligned["episodes"] == 1
    assert misaligned["episodes"] == 1
    assert aligned["mean_on_minus_off_mse"] < 0
    assert misaligned["mean_on_minus_off_mse"] > 0
