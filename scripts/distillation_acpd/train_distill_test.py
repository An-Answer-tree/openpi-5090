import dataclasses
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models.pi0_distill_acpd import AcpdHead
from openpi.models.pi0_distill_acpd import AcpdPi0Config
from openpi.models.pi0_distill_acpd import ExactContributionHead
from openpi.training import checkpoints
from openpi.training import config as training_config
from openpi.training.utils import TrainState
from scripts.distillation_acpd.train_distill import DistillTrainConfig
from scripts.distillation_acpd.train_distill import FrozenModelState
from scripts.distillation_acpd.train_distill import _acpd_loss_weight
from scripts.distillation_acpd.train_distill import _acpd_prediction_loss
from scripts.distillation_acpd.train_distill import _acpd_variance_loss
from scripts.distillation_acpd.train_distill import _action_corr_loss
from scripts.distillation_acpd.train_distill import _add_scaled_gradients
from scripts.distillation_acpd.train_distill import _data_start_batch
from scripts.distillation_acpd.train_distill import _exact_contribution_loss
from scripts.distillation_acpd.train_distill import _make_student_train_config
from scripts.distillation_acpd.train_distill import _make_teacher_train_config
from scripts.distillation_acpd.train_distill import _micro_step_train_rng
from scripts.distillation_acpd.train_distill import _per_sample_prediction_error
from scripts.distillation_acpd.train_distill import _restore_from_checkpoint
from scripts.distillation_acpd.train_distill import _scale_gradients
from scripts.distillation_acpd.train_distill import compute_gradients


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


def test_exact_contribution_head_is_zero_gated_and_detaches_flow_gradient():
    head = ExactContributionHead(4, rngs=nnx.Rngs(0))
    student_hidden = jnp.ones((2, 3, 4), dtype=jnp.float32)
    final_hidden = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)

    assert head.predict(student_hidden).shape == (2, 2, 3, 4)
    np.testing.assert_allclose(head.fuse(final_hidden, student_hidden), final_hidden)

    head.predictor.kernel.value = jnp.ones_like(head.predictor.kernel.value)
    head.predictor.bias.value = jnp.zeros_like(head.predictor.bias.value)

    def flow_loss(model):
        return jnp.mean(model.fuse(final_hidden, student_hidden))

    _, gradients = nnx.value_and_grad(flow_loss)(head)
    np.testing.assert_allclose(optax.global_norm(gradients["predictor"]), 0.0)
    assert float(optax.global_norm(gradients["gate"])) > 0.0


def test_exact_contribution_head_can_receive_task_gradient():
    head = ExactContributionHead(4, rngs=nnx.Rngs(0))
    student_hidden = jnp.ones((2, 3, 4), dtype=jnp.float32)
    final_hidden = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    head.gate.value = jnp.asarray(0.5, dtype=jnp.float32)

    def flow_loss(model):
        return jnp.mean(model.fuse(final_hidden, student_hidden, detach_prediction=False))

    _, gradients = nnx.value_and_grad(flow_loss)(head)
    assert float(optax.global_norm(gradients["predictor"])) > 0.0


def test_exact_contribution_head_has_stable_graph_metadata():
    first = nnx.graphdef(ExactContributionHead(4, rngs=nnx.Rngs(0)))
    second = nnx.graphdef(ExactContributionHead(4, rngs=nnx.Rngs(1)))

    assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(second)


def test_exact_contribution_loss_is_zero_for_equal_targets():
    target = jnp.arange(48, dtype=jnp.float32).reshape(2, 2, 3, 4) + 1.0

    loss, cosine, target_power = _exact_contribution_loss(target, target)

    np.testing.assert_allclose(loss, 0.0, atol=1e-6)
    np.testing.assert_allclose(cosine, 1.0, atol=1e-6)
    assert float(target_power) > 0.0


def test_acpd_v2_policy_configs_deploy_selected_layer():
    for config_name, layer, fusion_location, injection in (
        ("pi05_libero_backview_acpd_v2_lora", 9, "final", True),
        ("pi05_libero_backview_acpd_v2_layer10_lora", 10, "final", True),
        ("pi05_libero_backview_acpd_v2_layer10_aligned_lora", 10, "aligned_attention", True),
        ("pi05_libero_backview_acpd_v2_layer10_loss_only_lora", 10, "final", False),
    ):
        config = training_config.get_config(config_name)

        assert isinstance(config.model, AcpdPi0Config)
        assert config.model.align_layers == (layer,)
        assert config.model.exact_contribution_fusion
        assert config.model.exact_contribution_injection == injection
        assert config.model.exact_contribution_fusion_location == fusion_location
        assert not config.model.create_acpd_heads


def test_acpd_v2_train_and_eval_models_have_matching_parameter_trees():
    for config_name, layer, fusion_location, injection in (
        ("pi05_libero_backview_acpd_v2_lora", 9, "final", True),
        ("pi05_libero_backview_acpd_v2_layer10_lora", 10, "final", True),
        ("pi05_libero_backview_acpd_v2_layer10_aligned_lora", 10, "aligned_attention", True),
        ("pi05_libero_backview_acpd_v2_layer10_loss_only_lora", 10, "final", False),
    ):
        distill_config = DistillTrainConfig(
            student_init_params="base/params",
            teacher_params="teacher/params",
            assets_dir="assets",
            checkpoint_base_dir="checkpoints",
            align_layers=(layer,),
            exact_contribution_fusion=True,
            exact_contribution_injection=injection,
            exact_contribution_fusion_location=fusion_location,
        )

        student_config = _make_student_train_config(distill_config, create_acpd_heads=False)
        teacher_config = _make_teacher_train_config(distill_config, student_config)
        eval_config = training_config.get_config(config_name)

        assert student_config.model == eval_config.model
        assert isinstance(teacher_config.model, AcpdPi0Config)
        assert not teacher_config.model.exact_contribution_fusion
        assert not teacher_config.model.create_acpd_heads


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


def test_resume_data_loader_accounts_for_gradient_accumulation():
    config = DistillTrainConfig(
        student_init_params="base/params",
        teacher_params="teacher/params",
        assets_dir="assets",
        checkpoint_base_dir="checkpoints",
        gradient_accumulation_steps=4,
        resume_data_loader=True,
    )

    assert _data_start_batch(config, 30_000, resuming=True) == 120_000


def test_contribution_schedule_preserves_constant_default_and_clamps_endpoints():
    config = DistillTrainConfig()
    steps = jnp.asarray([0, 9_999, 10_000, 12_500, 15_000, 20_000], dtype=jnp.int32)
    constant = jax.jit(jax.vmap(lambda step: _acpd_loss_weight(config, step)))
    np.testing.assert_allclose(constant(steps), 0.2)

    scheduled_config = dataclasses.replace(config, acpd_loss_end_weight=0.05)
    scheduled = jax.jit(jax.vmap(lambda step: _acpd_loss_weight(scheduled_config, step)))
    expected = np.asarray([0.2, 0.2, 0.2, 0.125, 0.05, 0.05])
    np.testing.assert_allclose(scheduled(steps), expected, rtol=1e-6)
    # Check that the live weight scales a differentiable objective under JIT.
    gradients = jax.jit(
        jax.vmap(jax.grad(lambda x, step: _acpd_loss_weight(scheduled_config, step) * x**2), in_axes=(None, 0))
    )(2.0, steps)
    np.testing.assert_allclose(gradients, 4.0 * expected, rtol=1e-6)


def test_scheduled_weight_scales_actual_training_gradients():
    class ToyModel(nnx.Module):
        def __init__(self):
            self.exact_contribution_head = ExactContributionHead(7, rngs=nnx.Rngs(0))

        def compute_train_outputs(self, rng, observation, noisy_actions, time, *, train):
            return jnp.zeros_like(noisy_actions), None, (jnp.ones_like(noisy_actions),)

        def compute_attention_contributions(self, rng, observation, noisy_actions, time, *, train):
            batch, horizon, dim = noisy_actions.shape
            return jnp.zeros_like(noisy_actions), jnp.ones((batch, 1, 2, horizon, dim)), None, None

    model_def, params = nnx.split(ToyModel())
    tx = optax.adam(1e-3)
    state = TrainState(
        step=jnp.asarray(10_000),
        params=params,
        model_def=model_def,
        opt_state=tx.init(params),
        tx=tx,
        ema_decay=None,
    )
    teacher = FrozenModelState(params=params, model_def=model_def)
    config = DistillTrainConfig(
        exact_contribution_fusion=True,
        acpd_loss_end_weight=0.05,
        supervised_loss_weight=0.0,
        action_corr_loss_weight=0.0,
    )
    train_config = types.SimpleNamespace(trainable_filter=nnx.Param)
    batch = (None, None, jnp.zeros((2, 3, 7)))
    compute = jax.jit(
        lambda state: compute_gradients(
            config,
            train_config,
            jax.random.key(42),
            state,
            teacher,
            batch,
            jnp.asarray(0),
        )
    )
    initial_grads, initial_info = compute(state)
    final_grads, final_info = compute(dataclasses.replace(state, step=jnp.asarray(15_000)))
    assert float(optax.global_norm(initial_grads)) > 0.0
    for initial, final in zip(jax.tree.leaves(initial_grads), jax.tree.leaves(final_grads), strict=True):
        np.testing.assert_allclose(final, initial * 0.25, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(final_info["weighted_acpd_loss"], initial_info["weighted_acpd_loss"] * 0.25)
    np.testing.assert_allclose(final_info["acpd_loss"], initial_info["acpd_loss"])
    np.testing.assert_allclose(final_info["acpd_loss_weight"], 0.05)


def test_external_resume_restores_exact_step_and_adam_without_source_writes(tmp_path):
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    model_def, params = nnx.split(model)
    tx = optax.adam(1e-3)
    grads = jax.tree.map(jnp.ones_like, params)
    updates, opt_state = tx.update(grads, tx.init(params), params)
    state = TrainState(
        step=jnp.asarray(10_000),
        params=optax.apply_updates(params, updates),
        model_def=model_def,
        opt_state=opt_state,
        tx=tx,
        ema_decay=None,
    )
    loader = types.SimpleNamespace(data_config=lambda: types.SimpleNamespace(norm_stats=None, asset_id=None))
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "source", keep_period=1, overwrite=False, resume=False
    )
    checkpoints.save_state(manager, state, loader, 9_999)
    checkpoints.save_state(manager, dataclasses.replace(state, step=jnp.asarray(15_000)), loader, 14_999)
    manager.wait_until_finished()
    manager.close()
    # This traversal is limited to the tiny temporary test checkpoint.
    source_files = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in (tmp_path / "source").rglob("*")}
    template = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state)
    restored = _restore_from_checkpoint(str(tmp_path / "source" / "9999"), template)
    assert int(restored.step) == 10_000
    for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(actual, expected)
    actual_updates, _ = restored.tx.update(grads, restored.opt_state, restored.params)
    expected_updates, _ = state.tx.update(grads, state.opt_state, state.params)
    for actual, expected in zip(jax.tree.leaves(actual_updates), jax.tree.leaves(expected_updates), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert source_files == {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in source_files}
    assert _data_start_batch(DistillTrainConfig(resume_data_loader=True), int(restored.step), resuming=True) == 10_000
