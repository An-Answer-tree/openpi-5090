"""Measures the local action effect of H9's predicted-contribution gate."""

import dataclasses
import json
import logging
import os
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from lerobot.common.datasets import lerobot_dataset
import numpy as np
import tyro

from openpi.training import sharding
from scripts.distillation_acpd.exact_attention_probe import _split_episodes
from scripts.distillation_acpd.train_distill import DistillTrainConfig
from scripts.distillation_acpd.train_distill import FrozenModelState
from scripts.distillation_acpd.train_distill import _create_paired_data_loader
from scripts.distillation_acpd.train_distill import _init_frozen_model_state
from scripts.distillation_acpd.train_distill import _make_student_train_config
from scripts.train import init_logging

_BRANCHES = ("off", "agent_only", "wrist_only", "on")
_METRIC_NAMES = (
    "velocity_delta_rms_ratio",
    "delta_to_ideal_correction_cosine",
    "view_delta_cosine",
    "view_delta_disagreement",
    "view_norm_imbalance",
)


@dataclasses.dataclass(frozen=True)
class GateEffectProbeConfig:
    """Configuration for a frozen, single-GPU H9 gate probe."""

    checkpoint_params: str
    assets_dir: str
    output_dir: str
    dataset_repo_id: str = "libero_multiview_tuned_6view_lerobot"
    batch_size: int = 8
    num_batches: int = 128
    validation_fraction: float = 0.1
    bootstrap_samples: int = 2000
    seed: int = 42


def _measure(rng, state: FrozenModelState, batch):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    _, observation, actions = batch
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    noise = jax.random.normal(noise_rng, actions.shape)
    flow_time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    noisy_actions = flow_time[:, None, None] * noise + (1 - flow_time[:, None, None]) * actions
    velocity_on, _, hiddens = model.compute_train_outputs(
        preprocess_rng, observation, noisy_actions, flow_time, train=False, include_final_hidden=True
    )
    final_hidden = hiddens[-1]
    prediction = model.exact_contribution_head.predict(hiddens[0])
    gate = jnp.tanh(model.exact_contribution_head.gate.value).astype(final_hidden.dtype)
    velocity_off = model.action_out_proj(final_hidden)[..., :7].astype(jnp.float32)
    velocity_agent = model.action_out_proj(final_hidden + gate * prediction[:, 0].astype(final_hidden.dtype))[
        ..., :7
    ].astype(jnp.float32)
    velocity_wrist = model.action_out_proj(final_hidden + gate * prediction[:, 1].astype(final_hidden.dtype))[
        ..., :7
    ].astype(jnp.float32)
    velocities = jnp.stack([velocity_off, velocity_agent, velocity_wrist, velocity_on[..., :7].astype(jnp.float32)])
    target = (noise - actions)[..., :7].astype(jnp.float32)
    errors = jnp.mean(jnp.square(velocities - target[None]), axis=(2, 3)).T
    delta = velocities[-1] - velocity_off
    correction = target - velocity_off
    delta_norm = jnp.linalg.norm(delta, axis=(1, 2))
    correction_norm = jnp.linalg.norm(correction, axis=(1, 2))
    agent_delta = velocity_agent - velocity_off
    wrist_delta = velocity_wrist - velocity_off
    agent_delta_norm = jnp.linalg.norm(agent_delta, axis=(1, 2))
    wrist_delta_norm = jnp.linalg.norm(wrist_delta, axis=(1, 2))
    view_norm = jnp.maximum(agent_delta_norm + wrist_delta_norm, 1e-8)
    metrics = jnp.stack(
        [
            delta_norm / jnp.maximum(jnp.linalg.norm(velocity_off, axis=(1, 2)), 1e-8),
            jnp.sum(delta * correction, axis=(1, 2)) / jnp.maximum(delta_norm * correction_norm, 1e-8),
            jnp.sum(agent_delta * wrist_delta, axis=(1, 2)) / jnp.maximum(agent_delta_norm * wrist_delta_norm, 1e-8),
            jnp.linalg.norm(agent_delta - wrist_delta, axis=(1, 2)) / view_norm,
            jnp.abs(agent_delta_norm - wrist_delta_norm) / view_norm,
        ],
        axis=1,
    )
    return errors, metrics, gate


def _analyze(errors: np.ndarray, metrics: np.ndarray, episodes: np.ndarray, *, samples: int, seed: int) -> dict:
    """Bootstraps paired MSE differences over episode means."""
    episode_ids = np.unique(episodes)
    episode_errors = np.stack([errors[episodes == episode].mean(axis=0) for episode in episode_ids])
    episode_metrics = np.stack([metrics[episodes == episode].mean(axis=0) for episode in episode_ids])
    means = episode_errors.mean(axis=0)
    indices = np.random.default_rng(seed).integers(len(episode_ids), size=(samples, len(episode_ids)))
    resampled = episode_errors[indices].mean(axis=1)
    comparisons = {}
    for index, name in enumerate(_BRANCHES[1:], start=1):
        difference = means[index] - means[0]
        ci95 = np.percentile(resampled[:, index] - resampled[:, 0], [2.5, 97.5])
        comparisons[name] = {"mse_difference_vs_off": float(difference), "paired_ci95": ci95.tolist()}
    metric_means = episode_metrics.mean(axis=0)
    return {
        "episode_mean_mse": dict(zip(_BRANCHES, means.tolist(), strict=True)),
        "branch_vs_off": comparisons,
        "velocity_delta_rms_ratio": float(metric_means[0]),
        "delta_to_ideal_correction_cosine": float(metric_means[1]),
        "metric_means": dict(zip(_METRIC_NAMES[: metrics.shape[1]], metric_means.tolist(), strict=True)),
        "validation_episodes_sampled": len(episode_ids),
        "benchmark_success_available": False,
    }


def main(config: GateEffectProbeConfig) -> None:
    """Reads an H9 checkpoint and the fixed dataset without modifying either."""
    init_logging()
    if jax.device_count() != 1:
        raise ValueError("The frozen gate probe requires one visible JAX device.")
    output_dir = pathlib.Path(config.output_dir) / os.environ.get("SLURM_JOB_ID", "local")
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.info("Output directory: %s", output_dir)
    distill_config = DistillTrainConfig(
        name="pi05_libero_backview_acpd_v2_layer10",
        exp_name="gate_effect_probe",
        student_init_params=config.checkpoint_params,
        teacher_params=config.checkpoint_params,
        dataset_repo_id=config.dataset_repo_id,
        teacher_view_key="backview_image",
        teacher_use_wrist_image=False,
        assets_dir=config.assets_dir,
        checkpoint_base_dir=config.output_dir,
        align_layers=(10,),
        exact_contribution_fusion=True,
        batch_size=config.batch_size,
        num_workers=0,
        fsdp_devices=1,
        seed=config.seed,
        wandb_enabled=False,
    )
    train_config = _make_student_train_config(distill_config, create_acpd_heads=False)
    metadata = lerobot_dataset.LeRobotDatasetMetadata(config.dataset_repo_id)
    _, validation_episodes = _split_episodes(metadata.total_episodes, config.validation_fraction, config.seed)
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state, state_sharding = _init_frozen_model_state(
        train_config, jax.random.key(config.seed), mesh, config.checkpoint_params
    )
    jax.block_until_ready(state)
    measure = jax.jit(_measure, in_shardings=(replicated, state_sharding, data_sharding))
    loader = _create_paired_data_loader(
        distill_config,
        train_config,
        sharding_=data_sharding,
        shuffle=True,
        episodes=validation_episodes,
        include_episode_index=True,
    )
    error_rows, metric_rows, episode_rows = [], [], []
    loader_iterator = iter(loader)
    with sharding.set_mesh(mesh):
        for index in range(config.num_batches):
            *batch, episode_ids = next(loader_iterator)
            errors, metrics, gate = measure(jax.random.fold_in(jax.random.key(config.seed), index), state, tuple(batch))
            error_rows.append(np.asarray(jax.device_get(errors)))
            metric_rows.append(np.asarray(jax.device_get(metrics)))
            episode_rows.append(np.asarray(jax.device_get(episode_ids)).reshape(-1))
            if (index + 1) % 32 == 0:
                logging.info("Measured %d/%d batches", index + 1, config.num_batches)

    errors = np.concatenate(error_rows)
    metrics = np.concatenate(metric_rows)
    episodes = np.concatenate(episode_rows)
    if not np.isfinite(errors).all() or not np.isfinite(metrics).all():
        raise FloatingPointError("Non-finite gate probe metrics.")
    summary = _analyze(errors, metrics, episodes, samples=config.bootstrap_samples, seed=config.seed)
    summary.update(
        {"config": dataclasses.asdict(config), "gate": float(gate), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez(
        output_dir / "per_sample.npz",
        errors=errors,
        metrics=metrics,
        metric_names=np.asarray(_METRIC_NAMES),
        episode_index=episodes,
    )
    logging.info("Gate effect: %s", summary["branch_vs_off"]["on"])


if __name__ == "__main__":
    main(tyro.cli(GateEffectProbeConfig))
