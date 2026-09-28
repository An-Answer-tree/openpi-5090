"""Tests for H18 action-readout oracle analysis."""

import numpy as np

from scripts.distillation_acpd.analyze_action_readout_oracle import analyze


def test_oracle_detects_complementary_episode_winners(tmp_path):
    errors = np.asarray(
        [
            [1.0, 0.5, 1.3, 1.1],
            [1.0, 0.5, 1.3, 1.2],
            [1.0, 0.9, 0.4, 1.1],
            [1.0, 0.9, 0.4, 1.2],
        ],
        dtype=np.float32,
    )
    input_path = tmp_path / "errors.npz"
    np.savez(input_path, errors=errors, episode_index=np.asarray([0, 0, 1, 1]))
    result = analyze(input_path, bootstrap_samples=100, seed=42)
    assert result["best_fixed_arm"] == "hidden_only"
    assert result["episode_winner_fraction"]["hidden_layer10"] == 0.5
    assert result["episode_oracle_mean_mse"] < result["fixed_episode_mean_mse"]["hidden_only"]
