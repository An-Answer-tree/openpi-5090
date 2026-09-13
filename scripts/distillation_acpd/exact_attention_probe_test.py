import jax
import jax.numpy as jnp
import numpy as np

from scripts.distillation_acpd.exact_attention_probe import ExactAttentionProbeConfig
from scripts.distillation_acpd.exact_attention_probe import _analyze
from scripts.distillation_acpd.exact_attention_probe import _init_probe_params
from scripts.distillation_acpd.exact_attention_probe import _probe_loss
from scripts.distillation_acpd.exact_attention_probe import _split_episodes


def _config(**kwargs):
    return ExactAttentionProbeConfig(
        student_init_params="student",
        teacher_params="teacher",
        assets_dir="assets",
        result_dir="results",
        **kwargs,
    )


def test_episode_split_is_disjoint_and_deterministic():
    first_train, first_validation = _split_episodes(20, 0.1, 42)
    second_train, second_validation = _split_episodes(20, 0.1, 42)

    assert (first_train, first_validation) == (second_train, second_validation)
    assert len(first_validation) == 2
    assert set(first_train).isdisjoint(first_validation)


def test_probe_loss_and_analysis_identify_predictable_targets():
    config = _config(bootstrap_samples=100)
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(8, 1, 2, 2, 4)).astype(np.float32)
    targets = np.repeat(targets, len(config.layers), axis=1)
    predictions = np.repeat(targets[:, None], len(config.probe_seeds), axis=1)
    shuffled_targets = np.roll(targets, 1, axis=0)
    total_attention = targets.sum(axis=2)
    advantages = np.arange(8, dtype=np.float32)
    episode_indices = np.repeat(np.arange(4), 2)

    rows, decisions, selected_layer, hard_samples = _analyze(
        config,
        predictions,
        targets,
        shuffled_targets,
        total_attention,
        advantages,
        episode_indices,
        1.0,
    )

    assert len(rows) == 6
    assert all(decision["usable"] for decision in decisions)
    assert selected_layer == 6
    assert hard_samples == 2

    params = _init_probe_params(config, hidden_dim=4, num_views=2)
    student_hidden = jax.random.normal(jax.random.key(0), (8, 3, 2, 4))
    loss, metrics = _probe_loss(params, student_hidden, jnp.asarray(targets))
    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["probe_cosine"])
