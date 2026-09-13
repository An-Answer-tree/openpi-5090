"""Tests whether a weak-view student can predict a visual-only teacher message."""

from collections.abc import Mapping
import dataclasses
import functools
import logging

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import tyro

import openpi.models.model as _model
from openpi.shared import array_typing as at
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from scripts.distillation_acpd.train_distill import DistillTrainConfig
from scripts.distillation_acpd.train_distill import _create_paired_data_loader
from scripts.distillation_acpd.train_distill import _init_frozen_model_state
from scripts.distillation_acpd.train_distill import _make_student_train_config
from scripts.distillation_acpd.train_distill import _make_teacher_train_config
from scripts.train import init_logging
from scripts.train import init_train_state


@dataclasses.dataclass(frozen=True)
class VisualProbeConfig:
    """Configuration for the short visual-message probe."""

    student_init_params: str = tyro.MISSING
    teacher_params: str = tyro.MISSING
    assets_dir: str = tyro.MISSING
    dataset_repo_id: str = "libero_multiview_tuned_6view_lerobot"
    student_view_key: str = "backview_image"
    teacher_view_key: str = "agentview_image"
    wrist_view_key: str = "wrist_image"
    batch_size: int = 32
    num_train_steps: int = 500
    log_interval: int = 50
    num_workers: int = 8
    learning_rate: float = 1e-3
    seed: int = 42
    fsdp_devices: int = 2


def _normalize(x: at.Array, eps: float = 1e-6) -> at.Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def _visual_message(teacher_visual: at.Array, teacher_hidden: at.Array) -> at.Array:
    """Builds a fixed action-conditioned message from teacher visual tokens."""
    teacher_visual = teacher_visual.astype(jnp.float32)
    teacher_hidden = teacher_hidden.astype(jnp.float32)
    if teacher_visual.shape[-1] != teacher_hidden.shape[-1]:
        if teacher_visual.shape[-1] % teacher_hidden.shape[-1] != 0:
            raise ValueError("Teacher visual and action widths must be compatible.")
        ratio = teacher_visual.shape[-1] // teacher_hidden.shape[-1]
        teacher_visual = teacher_visual.reshape(
            *teacher_visual.shape[:-1], teacher_hidden.shape[-1], ratio
        ).mean(axis=-1)
    visual = _normalize(teacher_visual)
    query = _normalize(teacher_hidden)
    logits = jnp.einsum("bhd,bvd->bhv", query, visual) * teacher_hidden.shape[-1] ** -0.5
    attention = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bhv,bvd->bhd", attention, teacher_visual)


def _extract_features(
    rng: at.KeyArrayLike,
    student_state: training_utils.TrainState,
    teacher_state,
    batch: tuple[_model.Observation, _model.Observation, _model.Actions],
) -> tuple[at.Array, at.Array]:
    """Runs frozen teacher/student forwards and returns probe features."""
    student_model = nnx.merge(student_state.model_def, student_state.params)
    teacher_model = nnx.merge(teacher_state.model_def, teacher_state.params)
    student_model.eval()
    teacher_model.eval()
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    teacher_observation, student_observation, actions = batch
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    noisy_actions = time_expanded * noise + (1 - time_expanded) * actions

    _, _, student_hiddens = student_model.compute_train_outputs(
        preprocess_rng, student_observation, noisy_actions, time, train=False
    )
    _, teacher_visual, teacher_hiddens = teacher_model.compute_train_outputs(
        preprocess_rng, teacher_observation, noisy_actions, time, train=False
    )
    target = _visual_message(teacher_visual, teacher_hiddens[0])
    return jax.lax.stop_gradient(student_hiddens[0].astype(jnp.float32)), jax.lax.stop_gradient(target)


def _probe_loss(
    params: Mapping[str, at.Array],
    student_hidden: at.Array,
    target: at.Array,
) -> tuple[at.Array, dict[str, at.Array]]:
    prediction = jnp.einsum("bhd,df->bhf", student_hidden, params["weight"]) + params["bias"]
    prediction = _normalize(prediction)
    normalized_target = _normalize(target)
    correct_cosine = jnp.mean(jnp.sum(prediction * normalized_target, axis=-1))
    shuffled_target = jnp.roll(normalized_target, 1, axis=0)
    shuffled_cosine = jnp.mean(jnp.sum(prediction * shuffled_target, axis=-1))
    metrics = {
        "probe_loss": 1.0 - correct_cosine,
        "correct_cosine": correct_cosine,
        "shuffled_cosine": shuffled_cosine,
        "cosine_gap": correct_cosine - shuffled_cosine,
        "target_std": jnp.mean(jnp.std(target, axis=(0, 1))),
    }
    return metrics["probe_loss"], metrics


def _probe_step(
    params: Mapping[str, at.Array],
    opt_state: optax.OptState,
    student_hidden: at.Array,
    target: at.Array,
    optimizer: optax.GradientTransformation,
) -> tuple[Mapping[str, at.Array], optax.OptState, dict[str, at.Array]]:
    (loss, metrics), grads = jax.value_and_grad(_probe_loss, has_aux=True)(params, student_hidden, target)
    del loss
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, {**metrics, "probe_grad_norm": optax.global_norm(grads)}


def main(config: VisualProbeConfig):
    """Runs the visual-message probe without saving a checkpoint."""
    init_logging()
    if config.batch_size % jax.device_count() != 0:
        raise ValueError("Batch size must be divisible by the number of visible devices.")

    distill_config = DistillTrainConfig(
        student_init_params=config.student_init_params,
        teacher_params=config.teacher_params,
        dataset_repo_id=config.dataset_repo_id,
        student_view_key=config.student_view_key,
        teacher_view_key=config.teacher_view_key,
        teacher_use_wrist_image=True,
        wrist_view_key=config.wrist_view_key,
        assets_dir=config.assets_dir,
        checkpoint_base_dir="/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/probes",
        name="pi05_libero_backview_visual_message_probe",
        exp_name="visual_message_probe_500",
        batch_size=config.batch_size,
        num_train_steps=config.num_train_steps,
        num_workers=config.num_workers,
        fsdp_devices=config.fsdp_devices,
        align_layers=(6,),
        wandb_enabled=False,
    )
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    student_config = _make_student_train_config(distill_config)
    teacher_config = _make_teacher_train_config(distill_config, student_config)
    data_loader = _create_paired_data_loader(
        distill_config, student_config, sharding_=data_sharding, shuffle=True
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info("Initialized paired data loader:\n%s", training_utils.array_tree_to_info(batch))

    rng = jax.random.key(config.seed)
    feature_rng, student_init_rng, teacher_init_rng, probe_rng = jax.random.split(rng, 4)
    student_state, student_state_sharding = init_train_state(student_config, student_init_rng, mesh, resume=False)
    teacher_state, teacher_state_sharding = _init_frozen_model_state(
        teacher_config, teacher_init_rng, mesh, config.teacher_params
    )
    jax.block_until_ready((student_state, teacher_state))

    pextract_features = jax.jit(
        _extract_features,
        in_shardings=(replicated_sharding, student_state_sharding, teacher_state_sharding, data_sharding),
        out_shardings=(data_sharding, data_sharding),
    )
    probe_dim = 1024
    probe_params = {
        "weight": 0.01 * jax.random.normal(probe_rng, (probe_dim, probe_dim)),
        "bias": jnp.zeros((probe_dim,), dtype=jnp.float32),
    }
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(probe_params)
    pprobe_step = jax.jit(
        functools.partial(_probe_step, optimizer=optimizer),
        in_shardings=(replicated_sharding, replicated_sharding, data_sharding, data_sharding),
        out_shardings=(replicated_sharding, replicated_sharding, replicated_sharding),
    )

    final_metrics = None
    for step in tqdm.tqdm(range(config.num_train_steps), total=config.num_train_steps):
        with sharding.set_mesh(mesh):
            student_hidden, target = pextract_features(
                jax.random.fold_in(feature_rng, step), student_state, teacher_state, batch
            )
            probe_params, opt_state, metrics = pprobe_step(probe_params, opt_state, student_hidden, target)
        if step % config.log_interval == 0 or step == config.num_train_steps - 1:
            final_metrics = jax.device_get(metrics)
            logging.info(
                "Step %d: %s",
                step,
                ", ".join(f"{key}={value:.4f}" for key, value in sorted(final_metrics.items())),
            )
        batch = next(data_iter)
    logging.info("Final visual-message probe metrics: %s", final_metrics)


if __name__ == "__main__":
    main(tyro.cli(VisualProbeConfig))
