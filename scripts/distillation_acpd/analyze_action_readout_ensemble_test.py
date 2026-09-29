"""Tests fixed action readout ensemble analysis helpers."""

import numpy as np

from scripts.distillation_acpd.analyze_action_readout_ensemble import _readout_outputs


def test_readout_outputs_preserve_velocity_with_zero_heads():
    inputs = np.zeros((3, 2, 4, 6), dtype=np.float32)
    velocity = np.ones((2, 4, 2), dtype=np.float32)
    params = {
        "input_kernel": np.zeros((3, 6, 3), dtype=np.float32),
        "input_bias": np.zeros((3, 3), dtype=np.float32),
        "output_kernel": np.zeros((3, 3, 2), dtype=np.float32),
        "output_bias": np.zeros((3, 2), dtype=np.float32),
    }
    expected = np.broadcast_to(velocity[None], (3, 2, 4, 2))
    np.testing.assert_allclose(_readout_outputs(inputs, velocity, params), expected)
