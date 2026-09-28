"""Tests paired episode analysis for the frozen gate probe."""

import numpy as np

from scripts.distillation_acpd.gate_effect_probe import _analyze


def test_analyze_uses_episode_means_and_paired_differences():
    episodes = np.array([0, 0, 1])
    errors = np.array([[1.0, 0.9, 0.8, 0.7], [1.0, 0.9, 0.8, 0.7], [3.0, 2.9, 2.8, 2.7]])
    metrics = np.array([[0.1, 0.5], [0.1, 0.5], [0.3, -0.5]])
    result = _analyze(errors, metrics, episodes, samples=100, seed=42)
    assert result["episode_mean_mse"]["off"] == 2.0
    assert np.isclose(result["branch_vs_off"]["on"]["mse_difference_vs_off"], -0.3)
    assert result["branch_vs_off"]["on"]["paired_ci95"][1] < 0
    assert np.isclose(result["velocity_delta_rms_ratio"], 0.2)
    assert not result["benchmark_success_available"]
