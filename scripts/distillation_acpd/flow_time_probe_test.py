"""Tests the paired flow-time summary."""

import numpy as np

from scripts.distillation_acpd.flow_time_probe import _summarize_bin


def test_summary_pairs_models_within_episode():
    rows = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0],
            [5.0, 4.0, 7.0, 6.0],
        ]
    )
    summary = _summarize_bin(rows, np.array([1, 1, 2]), seed=42)

    assert summary["samples"] == 3
    assert summary["episodes"] == 2
    assert summary["sft_mse7"] == 3.5
    assert summary["acpd_mse7"] == 3.5
    assert summary["acpd_minus_sft_mse7"] == 0.0
