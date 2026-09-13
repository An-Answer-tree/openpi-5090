"""Probes exact privileged-view attention contributions on held-out episodes."""

from collections.abc import Mapping
import dataclasses
import functools
import json
import logging
import math
import os
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from lerobot.common.datasets import lerobot_dataset
import numpy as np
import optax
import tyro

from openpi.models import gemma
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.training import sharding
from scripts.distillation_acpd.train_distill import DistillTrainConfig
from scripts.distillation_acpd.train_distill import FrozenModelState
from scripts.distillation_acpd.train_distill import _create_paired_data_loader
from scripts.distillation_acpd.train_distill import _init_frozen_model_state
from scripts.distillation_acpd.train_distill import _make_student_train_config
from scripts.distillation_acpd.train_distill import _make_teacher_train_config
from scripts.train import init_logging


@dataclasses.dataclass(frozen=True)
class ExactAttentionProbeConfig:
    """Configuration for the exact privileged-attention probe."""

    student_init_params: str = tyro.MISSING
    teacher_params: str = tyro.MISSING
    assets_dir: str = tyro.MISSING
    result_dir: str = tyro.MISSING
    dataset_repo_id: str = "libero_multiview_tuned_6view_lerobot"
    student_view_key: str = "backview_image"
    teacher_view_key: str = "agentview_image"
    wrist_view_key: str = "wrist_image"
    layers: tuple[int, ...] = (6, 9, 12)
    probe_seeds: tuple[int, ...] = (11, 29, 47)
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_steps: int = 500
    num_validation_batches: int = 32
    validation_fraction: float = 0.1
    learning_rate: float = 1e-3
    log_interval: int = 50
    bootstrap_samples: int = 2_000
    seed: int = 42
    fsdp_devices: int = 2


def _split_episodes(total_episodes: int, validation_fraction: float, seed: int):
    episode_ids = np.random.default_rng(seed).permutation(total_episodes)
    validation_count = max(1, math.ceil(total_episodes * validation_fraction))
    return episode_ids[:-validation_count].tolist(), episode_ids[-validation_count:].tolist()


def _extract_features(
    rng: at.KeyArrayLike,
    student_state: FrozenModelState,
    teacher_state: FrozenModelState,
    batch: tuple[_model.Observation, _model.Observation, _model.Actions],
):
    student_model = nnx.merge(student_state.model_def, student_state.params)
    teacher_model = nnx.merge(teacher_state.model_def, teacher_state.params)
    student_model.eval()
    teacher_model.eval()

    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    teacher_observation, student_observation, actions = batch
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    noisy_actions = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
    target_velocity = noise - actions

    student_velocity, _, student_hiddens = student_model.compute_train_outputs(
        preprocess_rng, student_observation, noisy_actions, time, train=False
    )
    teacher_velocity, contributions, shuffled_contributions, total_attention = (
        teacher_model.compute_attention_contributions(
            preprocess_rng, teacher_observation, noisy_actions, time, train=False
        )
    )
    student_error = jnp.mean(jnp.square(student_velocity[..., :7] - target_velocity[..., :7]), axis=(1, 2))
    teacher_error = jnp.mean(jnp.square(teacher_velocity[..., :7] - target_velocity[..., :7]), axis=(1, 2))
    student_hidden = jnp.stack(student_hiddens, axis=1).astype(jnp.float32)
    return jax.tree.map(
        jax.lax.stop_gradient,
        (
            student_hidden,
            contributions.astype(jnp.float32),
            shuffled_contributions.astype(jnp.float32),
            total_attention.astype(jnp.float32),
            student_error - teacher_error,
        ),
    )


def _init_probe_params(config: ExactAttentionProbeConfig, hidden_dim: int, num_views: int):
    shape = (len(config.probe_seeds), len(config.layers), num_views, hidden_dim, hidden_dim)
    weights = [
        jax.random.normal(jax.random.key(seed), shape[1:]) * (0.01 * hidden_dim**-0.5)
        for seed in config.probe_seeds
    ]
    return {
        "weight": jnp.stack(weights),
        "bias": jnp.zeros(shape[:-2] + (hidden_dim,), dtype=jnp.float32),
    }


def _predict(params: Mapping[str, at.Array], student_hidden: at.Array) -> at.Array:
    return (
        jnp.einsum("blhd,rlvde->brlvhe", student_hidden, params["weight"])
        + params["bias"][None, :, :, :, None, :]
    )


def _probe_loss(params: Mapping[str, at.Array], student_hidden: at.Array, target: at.Array):
    prediction = _predict(params, student_hidden)
    squared_error = jnp.mean(jnp.square(prediction - target[:, None]), axis=(0, 4, 5))
    target_power = jnp.mean(jnp.square(target), axis=(0, 3, 4))
    normalized_mse = squared_error / jnp.maximum(target_power[None], 1e-8)

    prediction_norm = prediction / jnp.maximum(jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-8)
    target_norm = target / jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
    cosine = jnp.mean(jnp.sum(prediction_norm * target_norm[:, None], axis=-1), axis=(0, 4))
    return jnp.mean(normalized_mse), {
        "probe_loss": jnp.mean(normalized_mse),
        "probe_cosine": jnp.mean(cosine),
    }


def _probe_gradients(params, student_hidden, target):
    (loss, metrics), gradients = jax.value_and_grad(_probe_loss, has_aux=True)(params, student_hidden, target)
    del loss
    return gradients, metrics


def _apply_gradients(params, opt_state, gradients, optimizer):
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    return optax.apply_updates(params, updates), opt_state


def _normalize(array: np.ndarray) -> np.ndarray:
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


def _bootstrap_interval(
    values: np.ndarray,
    episode_indices: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    episodes = np.unique(episode_indices)
    episode_means = np.asarray([values[episode_indices == episode].mean() for episode in episodes])
    rng = np.random.default_rng(seed)
    draws = rng.choice(episode_means, size=(samples, len(episode_means)), replace=True).mean(axis=1)
    return np.percentile(draws, [2.5, 97.5]).tolist()


def _analyze(
    config: ExactAttentionProbeConfig,
    predictions: np.ndarray,
    targets: np.ndarray,
    shuffled_targets: np.ndarray,
    total_attention: np.ndarray,
    advantages: np.ndarray,
    episode_indices: np.ndarray,
    gradient_norm: float,
):
    correct_cosine = np.sum(_normalize(predictions) * _normalize(targets)[:, None], axis=-1).mean(axis=-1)
    shuffled_cosine = np.sum(
        _normalize(predictions) * _normalize(shuffled_targets)[:, None], axis=-1
    ).mean(axis=-1)
    gaps = correct_cosine - shuffled_cosine

    target_mean = targets.mean(axis=(0, 3), keepdims=True)
    baseline_mse = np.mean(np.square(targets - target_mean), axis=(0, 3, 4))
    probe_mse = np.mean(np.square(predictions - targets[:, None]), axis=(0, 4, 5))
    explained_variance = 1 - probe_mse / np.maximum(baseline_mse[None], 1e-8)
    target_std = targets.std(axis=(0, 3, 4))
    contribution_norm = np.linalg.norm(targets, axis=-1)
    total_norm = np.linalg.norm(total_attention, axis=-1)
    norm_ratio = np.mean(contribution_norm / np.maximum(total_norm[:, :, None], 1e-8), axis=(0, 3))

    hard_mask = advantages >= np.quantile(advantages, 0.75)
    rows = []
    view_names = ("agentview", "wrist")
    for layer_index, layer in enumerate(config.layers):
        for view_index, view in enumerate(view_names):
            per_sample_gap = gaps[:, :, layer_index, view_index].mean(axis=1)
            hard_gap = per_sample_gap[hard_mask]
            row = {
                "layer": layer,
                "view": view,
                "correct_cosine": float(correct_cosine[:, :, layer_index, view_index].mean()),
                "shuffled_cosine": float(shuffled_cosine[:, :, layer_index, view_index].mean()),
                "cosine_gap": float(per_sample_gap.mean()),
                "cosine_gap_ci95": _bootstrap_interval(
                    per_sample_gap,
                    episode_indices,
                    samples=config.bootstrap_samples,
                    seed=config.seed + layer + view_index,
                ),
                "hard_cosine_gap": float(hard_gap.mean()),
                "hard_cosine_gap_ci95": _bootstrap_interval(
                    hard_gap,
                    episode_indices[hard_mask],
                    samples=config.bootstrap_samples,
                    seed=config.seed + 100 + layer + view_index,
                ),
                "explained_variance": float(explained_variance[:, layer_index, view_index].mean()),
                "explained_variance_by_seed": explained_variance[:, layer_index, view_index].tolist(),
                "target_std": float(target_std[layer_index, view_index]),
                "attention_norm_ratio": float(norm_ratio[layer_index, view_index]),
            }
            rows.append(row)

    layer_decisions = []
    for layer_index, layer in enumerate(config.layers):
        per_sample_gap = gaps[:, :, layer_index].mean(axis=(1, 2))
        hard_gap = per_sample_gap[hard_mask]
        gap_interval = _bootstrap_interval(
            per_sample_gap,
            episode_indices,
            samples=config.bootstrap_samples,
            seed=config.seed + layer,
        )
        hard_interval = _bootstrap_interval(
            hard_gap,
            episode_indices[hard_mask],
            samples=config.bootstrap_samples,
            seed=config.seed + 100 + layer,
        )
        mean_explained_variance = float(explained_variance[:, layer_index].mean())
        finite_signal = bool(
            np.isfinite(target_std[layer_index]).all()
            and np.all(target_std[layer_index] > 0)
            and np.isfinite(norm_ratio[layer_index]).all()
            and np.all(norm_ratio[layer_index] > 0)
            and np.isfinite(gradient_norm)
            and gradient_norm > 0
        )
        usable = bool(
            per_sample_gap.mean() >= 0.10
            and gap_interval[0] > 0
            and hard_gap.mean() >= 0.05
            and hard_interval[0] > 0
            and mean_explained_variance > 0
            and finite_signal
        )
        layer_decisions.append(
            {
                "layer": layer,
                "cosine_gap": float(per_sample_gap.mean()),
                "cosine_gap_ci95": gap_interval,
                "hard_cosine_gap": float(hard_gap.mean()),
                "hard_cosine_gap_ci95": hard_interval,
                "explained_variance": mean_explained_variance,
                "usable": usable,
            }
        )

    passing = sorted(
        (decision for decision in layer_decisions if decision["usable"]),
        key=lambda decision: decision["cosine_gap"],
        reverse=True,
    )
    selected_layer = None
    if passing:
        selected_layer = passing[0]["layer"]
        if len(passing) > 1 and passing[0]["cosine_gap"] - passing[1]["cosine_gap"] < 0.02:
            selected_layer = min(decision["layer"] for decision in passing)
    return rows, layer_decisions, selected_layer, int(hard_mask.sum())


def main(config: ExactAttentionProbeConfig):
    """Trains probes and evaluates them on episode-disjoint validation data."""
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
        checkpoint_base_dir=config.result_dir,
        name="pi05_libero_backview_exact_attention_probe",
        exp_name="h7_layers6_9_12",
        batch_size=config.batch_size,
        num_workers=0,
        fsdp_devices=config.fsdp_devices,
        align_layers=config.layers,
        wandb_enabled=False,
    )
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    student_config = _make_student_train_config(distill_config, create_acpd_heads=False)
    teacher_config = _make_teacher_train_config(distill_config, student_config)

    metadata = lerobot_dataset.LeRobotDatasetMetadata(config.dataset_repo_id)
    train_episodes, validation_episodes = _split_episodes(
        metadata.total_episodes, config.validation_fraction, config.seed
    )
    logging.info(
        "Episode split: train=%d, validation=%d",
        len(train_episodes),
        len(validation_episodes),
    )
    train_loader = _create_paired_data_loader(
        distill_config,
        student_config,
        sharding_=data_sharding,
        shuffle=True,
        episodes=train_episodes,
    )
    validation_loader = _create_paired_data_loader(
        distill_config,
        student_config,
        sharding_=data_sharding,
        shuffle=True,
        episodes=validation_episodes,
        include_episode_index=True,
    )

    rng = jax.random.key(config.seed)
    feature_rng, student_rng, teacher_rng = jax.random.split(rng, 3)
    student_state, student_sharding = _init_frozen_model_state(
        student_config, student_rng, mesh, config.student_init_params
    )
    teacher_state, teacher_sharding = _init_frozen_model_state(
        teacher_config, teacher_rng, mesh, config.teacher_params
    )
    jax.block_until_ready((student_state, teacher_state))

    pextract = jax.jit(
        _extract_features,
        in_shardings=(replicated_sharding, student_sharding, teacher_sharding, data_sharding),
        out_shardings=(data_sharding,) * 5,
    )
    hidden_dim = gemma.get_config(student_config.model.action_expert_variant).width
    probe_params = _init_probe_params(config, hidden_dim, num_views=2)
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(probe_params)
    pprobe_gradients = jax.jit(
        _probe_gradients,
        in_shardings=(replicated_sharding, data_sharding, data_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
    )
    papply_gradients = jax.jit(
        functools.partial(_apply_gradients, optimizer=optimizer),
        in_shardings=(replicated_sharding, replicated_sharding, replicated_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
    )
    ppredict = jax.jit(
        _predict,
        in_shardings=(replicated_sharding, data_sharding),
        out_shardings=data_sharding,
    )

    train_iterator = iter(train_loader)
    final_gradient_norm = None
    for step in range(config.num_train_steps):
        accumulated_gradients = jax.tree.map(jnp.zeros_like, probe_params)
        step_metrics = []
        for micro_step in range(config.gradient_accumulation_steps):
            batch = next(train_iterator)
            student_hidden, target, _, _, _ = pextract(
                jax.random.fold_in(feature_rng, step * config.gradient_accumulation_steps + micro_step),
                student_state,
                teacher_state,
                batch,
            )
            gradients, metrics = pprobe_gradients(probe_params, student_hidden, target)
            accumulated_gradients = jax.tree.map(
                lambda accumulated, gradient: accumulated
                + gradient / config.gradient_accumulation_steps,
                accumulated_gradients,
                gradients,
            )
            step_metrics.append(metrics)
        probe_params, opt_state = papply_gradients(probe_params, opt_state, accumulated_gradients)
        if step % config.log_interval == 0 or step == config.num_train_steps - 1:
            metrics = jax.device_get(jax.tree.map(lambda *values: jnp.mean(jnp.stack(values)), *step_metrics))
            final_gradient_norm = float(jax.device_get(optax.global_norm(accumulated_gradients)))
            logging.info(
                "Step %d: loss=%.6f, cosine=%.6f, grad_norm=%.6f",
                step,
                metrics["probe_loss"],
                metrics["probe_cosine"],
                final_gradient_norm,
            )

    validation_iterator = iter(validation_loader)
    validation = [[] for _ in range(6)]
    for validation_step in range(config.num_validation_batches):
        teacher_observation, student_observation, actions, episode_index = next(validation_iterator)
        features = pextract(
            jax.random.fold_in(feature_rng, config.num_train_steps + validation_step),
            student_state,
            teacher_state,
            (teacher_observation, student_observation, actions),
        )
        prediction = ppredict(probe_params, features[0])
        arrays = (prediction, *features[1:], episode_index)
        for values, array in zip(validation, arrays, strict=True):
            values.append(np.asarray(jax.device_get(array)))

    predictions = np.concatenate(validation[0], axis=0)
    targets = np.concatenate(validation[1], axis=0)
    shuffled_targets = np.concatenate(validation[2], axis=0)
    total_attention = np.concatenate(validation[3], axis=0)
    advantages = np.concatenate(validation[4], axis=0)
    episode_indices = np.concatenate(validation[5], axis=0)
    rows, layer_decisions, selected_layer, hard_sample_count = _analyze(
        config,
        predictions,
        targets,
        shuffled_targets,
        total_attention,
        advantages,
        episode_indices,
        final_gradient_norm,
    )
    for row in rows:
        logging.info(
            "H7 layer=%d view=%s gap=%.4f ci95=%s hard_gap=%.4f hard_ci95=%s ev=%.4f norm=%.4f",
            row["layer"],
            row["view"],
            row["cosine_gap"],
            row["cosine_gap_ci95"],
            row["hard_cosine_gap"],
            row["hard_cosine_gap_ci95"],
            row["explained_variance"],
            row["attention_norm_ratio"],
        )

    result = {
        "config": dataclasses.asdict(config),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "validation_samples": len(episode_indices),
        "hard_samples": hard_sample_count,
        "final_gradient_norm": final_gradient_norm,
        "metrics": rows,
        "layer_decisions": layer_decisions,
        "selected_layer": selected_layer,
    }
    result_dir = pathlib.Path(config.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"metrics_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logging.info("Saved H7 metrics to %s; selected_layer=%s", result_path, selected_layer)


if __name__ == "__main__":
    main(tyro.cli(ExactAttentionProbeConfig))
