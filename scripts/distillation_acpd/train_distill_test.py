import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models.pi0_distill_acpd import AcpdHead
from scripts.distillation_acpd.train_distill import _acpd_prediction_loss
from scripts.distillation_acpd.train_distill import _acpd_variance_loss
from scripts.distillation_acpd.train_distill import _action_corr_loss
from scripts.distillation_acpd.train_distill import _add_scaled_gradients
from scripts.distillation_acpd.train_distill import _micro_step_train_rng
from scripts.distillation_acpd.train_distill import _per_sample_prediction_error
from scripts.distillation_acpd.train_distill import _scale_gradients


def test_acpd_prediction_loss_is_zero_for_equal_cues():
    cue = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=jnp.float32)

    loss, cosine = _acpd_prediction_loss(cue, cue)

    np.testing.assert_allclose(loss, 0.0, atol=1e-6)
    np.testing.assert_allclose(cosine, 1.0, atol=1e-6)


def test_acpd_variance_loss_penalizes_constant_cues():
    constant = jnp.ones((2, 3, 2), dtype=jnp.float32)
    diverse = jnp.asarray(
        [
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
            [[0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]],
        ],
        dtype=jnp.float32,
    )

    constant_loss, constant_std = _acpd_variance_loss(constant)
    diverse_loss, diverse_std = _acpd_variance_loss(diverse)

    assert float(constant_loss) > float(diverse_loss)
    assert float(constant_std) < float(diverse_std)


def test_acpd_head_trains_selector_and_predictor_but_detaches_teacher_inputs():
    head = AcpdHead(4, 4, 4, 8, rngs=nnx.Rngs(0))
    student_hidden = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 10.0
    teacher_visual = jnp.arange(32, dtype=jnp.float32).reshape(2, 4, 4) / 10.0
    teacher_hidden = jnp.flip(student_hidden, axis=-1)

    def head_loss(model):
        predicted, privileged, _ = model(student_hidden, teacher_visual, teacher_hidden, detach_query=True)
        return _acpd_prediction_loss(predicted, privileged)[0] + 0.1 * _acpd_variance_loss(privileged)[0]

    _, grads = nnx.value_and_grad(head_loss)(head)
    assert float(optax.global_norm(grads["privileged_predictor"])) > 0.0
    assert float(optax.global_norm(grads["visual_memory_projector"])) > 0.0
    assert float(optax.global_norm(grads["query_projector"])) > 0.0

    def input_loss(student, visual, teacher):
        predicted, privileged, _ = head(student, visual, teacher, detach_query=True)
        return _acpd_prediction_loss(predicted, privileged)[0]

    student_grad, visual_grad, teacher_grad = jax.grad(input_loss, argnums=(0, 1, 2))(
        student_hidden,
        teacher_visual,
        teacher_hidden,
    )
    assert float(jnp.linalg.norm(student_grad)) > 0.0
    np.testing.assert_allclose(visual_grad, 0.0, atol=1e-6)
    np.testing.assert_allclose(teacher_grad, 0.0, atol=1e-6)


def test_action_corr_loss_matches_pearson_correlation():
    velocity = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]])

    np.testing.assert_allclose(_action_corr_loss(velocity, velocity), 0.0, atol=1e-6)
    np.testing.assert_allclose(_action_corr_loss(velocity, -velocity), 2.0, atol=1e-6)


def test_action_corr_loss_ignores_libero_padding_dimensions():
    student = jnp.asarray([[[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]]])
    teacher = jnp.asarray([[[1.0, 2.0, -5.0], [3.0, 4.0, 7.0]]])

    np.testing.assert_allclose(_action_corr_loss(student, teacher, task_action_dim=2), 0.0, atol=1e-6)


def test_per_sample_prediction_error_reduces_non_batch_dimensions():
    prediction = jnp.asarray([[[1.0, 2.0]], [[3.0, 5.0]]])
    target = jnp.asarray([[[0.0, 0.0]], [[1.0, 1.0]]])

    np.testing.assert_allclose(_per_sample_prediction_error(prediction, target), [2.5, 10.0])


def test_gradient_accumulation_averages_independent_microbatches():
    first = {"x": jnp.asarray([2.0, 4.0])}
    second = {"x": jnp.asarray([6.0, 8.0])}
    accumulated = _scale_gradients(first, scale=0.5)
    accumulated = _add_scaled_gradients(accumulated, second, scale=0.5)

    np.testing.assert_allclose(accumulated["x"], [4.0, 6.0])
    first_rng = _micro_step_train_rng(jax.random.key(0), 3, 0, gradient_accumulation_steps=4)
    second_rng = _micro_step_train_rng(jax.random.key(0), 3, 1, gradient_accumulation_steps=4)
    assert not np.array_equal(first_rng, second_rng)
