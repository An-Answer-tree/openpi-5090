"""Tests for paired view-branch contrasts."""

import numpy as np

from scripts.distillation_acpd.analyze_gate_views import analyze_view_branches


def test_view_contrasts_use_equal_episode_weight() -> None:
    errors = np.array(
        [
            [3.0, 2.0, 2.0, 2.0],
            [3.0, 2.0, 4.0, 3.0],
            [3.0, 2.0, 4.0, 3.0],
            [3.0, 2.0, 4.0, 3.0],
        ]
    )
    episodes = np.array([0, 1, 1, 1])

    result = analyze_view_branches(errors, episodes, bootstrap_samples=100, seed=42)

    assert result["samples"] == 4
    assert result["episodes"] == 2
    assert result["contrasts"]["wrist_minus_agent"]["mean"] == 1.0
    assert result["contrasts"]["on_minus_agent"]["mean"] == 0.5
    assert result["contrasts"]["interaction"]["mean"] == 0.5
