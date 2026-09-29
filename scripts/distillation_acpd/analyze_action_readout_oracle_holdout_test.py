"""Tests H18 held-out oracle analysis."""

import numpy as np

from scripts.distillation_acpd.analyze_action_readout_oracle_holdout import analyze


def test_holdout_oracle_uses_the_other_half(tmp_path):
    errors = np.asarray(
        [
            [1.0, 0.4, 1.0, 1.1],
            [1.0, 0.3, 1.2, 1.1],
            [1.0, 0.5, 1.0, 1.1],
            [1.0, 0.4, 1.2, 1.1],
        ],
        dtype=np.float32,
    )
    input_path = tmp_path / "errors.npz"
    np.savez(input_path, errors=errors, episode_index=np.asarray([0, 0, 0, 0]))
    result = analyze(input_path, bootstrap_samples=100, seed=42)
    assert result["eligible_episodes"] == 1
    assert result["holdout_oracle_minus_baseline"] < 0.0
