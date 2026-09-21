import jax.numpy as jnp
import numpy as np

from scripts.distillation_acpd.gradient_conflict_probe import _gradient_pair_metrics


def test_gradient_pair_metrics_detects_alignment_and_conflict():
    task = {"x": jnp.asarray([3.0, 4.0])}

    aligned = _gradient_pair_metrics(task, {"x": jnp.asarray([6.0, 8.0])})
    opposed = _gradient_pair_metrics(task, {"x": jnp.asarray([-3.0, -4.0])})

    np.testing.assert_allclose(aligned["cosine"], 1.0, atol=1e-6)
    np.testing.assert_allclose(aligned["norm_ratio"], 2.0, atol=1e-6)
    np.testing.assert_allclose(aligned["conflict"], 0.0)
    np.testing.assert_allclose(opposed["cosine"], -1.0, atol=1e-6)
    np.testing.assert_allclose(opposed["conflict"], 1.0)
