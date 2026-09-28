"""Tests view-only gate observable analysis."""

import numpy as np

from scripts.distillation_acpd.analyze_gate_observables import analyze_observables


def test_observable_analysis_reports_held_out_auc():
    errors = np.array(
        [
            [1.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 1.5],
            [1.0, 0.0, 0.0, 1.5],
            [1.0, 0.0, 0.0, 1.5],
            [1.0, 0.0, 0.0, 1.5],
        ]
    )
    metrics = np.array(
        [
            [0.1, 0.0, 0.9, 0.1, 0.1],
            [0.1, 0.0, 0.9, 0.1, 0.1],
            [0.1, 0.0, 0.8, 0.2, 0.2],
            [0.1, 0.0, 0.8, 0.2, 0.2],
            [0.1, 0.0, -0.8, 0.8, 0.8],
            [0.1, 0.0, -0.8, 0.8, 0.8],
            [0.1, 0.0, -0.9, 0.9, 0.9],
            [0.1, 0.0, -0.9, 0.9, 0.9],
        ]
    )
    result = analyze_observables(errors, metrics, np.repeat(np.arange(4), 2))
    assert result["episodes"] == 4
    assert result["benefit_episodes"] == 2
    for values in result["observable_results"].values():
        assert len(values["held_out_auc_per_fold"]) == 2
        assert len(values["held_out_policy_per_fold"]) == 2
