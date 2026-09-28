"""Tests target-free observable router analysis."""

import numpy as np

from scripts.distillation_acpd.analyze_observable_action_router import _summarize_inputs


def test_summarize_inputs_pools_three_visible_feature_groups():
    inputs = np.ones((3, 2, 4, 96), dtype=np.float32)
    summary = _summarize_inputs(inputs)
    assert summary.shape == (2, 288)
    np.testing.assert_allclose(summary, 1.0)
