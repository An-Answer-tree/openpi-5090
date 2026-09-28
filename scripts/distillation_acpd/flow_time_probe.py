"""Compares paired SFT and ACPD-v2 flow errors across fixed time bins."""

import dataclasses
import gc
import json
import logging
import os
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.training import sharding
from scripts.distillation_acpd.train_distill import DistillTrainConfig
from scripts.distillation_acpd.train_distill import FrozenModelState
from scripts.distillation_acpd.train_distill import _create_paired_data_loader
from scripts.distillation_acpd.train_distill import _init_frozen_model_state
from scripts.distillation_acpd.train_distill import _make_student_train_config
from scripts.train import init_logging


@dataclasses.dataclass(frozen=True)
class FlowTimeProbeConfig:
    """Inputs for the read-only 5K flow-time diagnostic."""

    sft_params: str
    acpd_params: str
    assets_dir: str
    output_dir: str
    dataset_repo_id: str = "libero_multiview_tuned_6view_lerobot"
    batch_size: int = 8
    num_batches: int = 64
    seed: int = 42
    all_times_per_sample: bool = False


def _model_errors(rng, model_state: FrozenModelState, batch, batch_index, time_index):
    """Evaluates one frozen model on a deterministic noisy action and target."""
    model = nnx.merge(model_state.model_def, model_state.params)
    model.eval()
    _, observation, actions = batch
    preprocess_rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, actions.shape)
    sample_index = batch_index * actions.shape[0] + jnp.arange(actions.shape[0])
    time_bin = jnp.where(time_index < 0, sample_index % 5, time_index)
    flow_time = (time_bin.astype(jnp.float32) + 0.5) / 5.0
    noisy_actions = flow_time[:, None, None] * noise + (1 - flow_time[:, None, None]) * actions
    target = noise - actions
    velocity, _, _ = model.compute_train_outputs(preprocess_rng, observation, noisy_actions, flow_time, train=False)

    def mse(velocity, action_dim):
        return jnp.mean(jnp.square(velocity[..., :action_dim] - target[..., :action_dim]), axis=(1, 2))

    return (
        time_bin,
        mse(velocity, 7),
        mse(velocity, 32),
        _centered_action_cosine(velocity, target),
    )


def _evaluate_model(
    probe_config: FlowTimeProbeConfig,
    loader_config: DistillTrainConfig,
    model_config,
    params_path: str,
    init_rng: jax.Array,
    sample_rng: jax.Array,
    mesh: jax.sharding.Mesh,
    data_sharding: jax.sharding.Sharding,
    replicated: jax.sharding.Sharding,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs one model pass, returning host-side metrics and sample identifiers."""
    state, state_sharding = _init_frozen_model_state(model_config, init_rng, mesh, params_path)
    jax.block_until_ready(state)
    loader = _create_paired_data_loader(
        loader_config, model_config, sharding_=data_sharding, shuffle=True, include_episode_index=True
    )
    model_errors = jax.jit(
        _model_errors,
        in_shardings=(replicated, state_sharding, data_sharding, replicated, replicated),
    )
    time_rows, error_rows, episode_rows = [], [], []
    with sharding.set_mesh(mesh):
        for batch_index, (*batch, episode_ids) in zip(range(probe_config.num_batches), loader, strict=False):
            for time_index in range(5) if probe_config.all_times_per_sample else (-1,):
                values = model_errors(
                    jax.random.fold_in(sample_rng, batch_index),
                    state,
                    tuple(batch),
                    jnp.asarray(batch_index),
                    jnp.asarray(time_index),
                )
                time_bin, *errors = jax.device_get(values)
                time_rows.append(np.asarray(time_bin))
                error_rows.append(np.stack(errors, axis=-1))
                episode_rows.append(np.asarray(jax.device_get(episode_ids)).reshape(-1))
            if (batch_index + 1) % 16 == 0:
                logging.info("Evaluated %d/%d batches", batch_index + 1, probe_config.num_batches)

    del model_errors, loader, state
    jax.clear_caches()
    gc.collect()
    return np.concatenate(time_rows), np.concatenate(error_rows), np.concatenate(episode_rows)


def _centered_action_cosine(velocity: jax.Array, target: jax.Array) -> jax.Array:
    """Matches ACL's centered cosine on the real seven action dimensions."""
    prediction = velocity[..., :7].astype(jnp.float32).reshape((velocity.shape[0], -1))
    truth = target[..., :7].astype(jnp.float32).reshape((target.shape[0], -1))
    prediction -= jnp.mean(prediction, axis=-1, keepdims=True)
    truth -= jnp.mean(truth, axis=-1, keepdims=True)
    prediction /= jnp.maximum(jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-6)
    truth /= jnp.maximum(jnp.linalg.norm(truth, axis=-1, keepdims=True), 1e-6)
    return jnp.sum(prediction * truth, axis=-1)


def _summarize_bin(rows: np.ndarray, episodes: np.ndarray, seed: int) -> dict[str, float | int | list[float]]:
    """Reports episode-level paired means and a cluster bootstrap interval."""
    unique_episodes = np.unique(episodes)
    episode_means = np.stack([rows[episodes == episode].mean(axis=0) for episode in unique_episodes])
    difference = episode_means[:, 1] - episode_means[:, 0]
    direction_difference = episode_means[:, 5] - episode_means[:, 4]
    indices = np.random.default_rng(seed).integers(len(unique_episodes), size=(2000, len(unique_episodes)))
    ci95 = np.percentile(difference[indices].mean(axis=1), [2.5, 97.5])
    direction_ci95 = np.percentile(direction_difference[indices].mean(axis=1), [2.5, 97.5])
    return {
        "samples": len(rows),
        "episodes": len(unique_episodes),
        "sft_mse7": float(episode_means[:, 0].mean()),
        "acpd_mse7": float(episode_means[:, 1].mean()),
        "acpd_minus_sft_mse7": float(difference.mean()),
        "paired_ci95": ci95.tolist(),
        "sft_mse32": float(episode_means[:, 2].mean()),
        "acpd_mse32": float(episode_means[:, 3].mean()),
        "sft_target_cosine7": float(episode_means[:, 4].mean()),
        "acpd_target_cosine7": float(episode_means[:, 5].mean()),
        "acpd_minus_sft_target_cosine7": float(direction_difference.mean()),
        "target_cosine7_ci95": direction_ci95.tolist(),
    }


def main(config: FlowTimeProbeConfig) -> None:
    """Loads frozen checkpoints in place and writes only small diagnostic arrays."""
    init_logging()
    if jax.device_count() != 1:
        raise ValueError("Flow-time probe requires one visible GPU.")
    output_dir = pathlib.Path(config.output_dir) / os.environ.get("SLURM_JOB_ID", "local")
    output_dir.mkdir(parents=True, exist_ok=False)
    common = {
        "name": "pi05_libero_backview_acpd_v2_layer10",
        "exp_name": "flow_time_5k_probe",
        "student_init_params": config.acpd_params,
        "teacher_params": config.acpd_params,
        "dataset_repo_id": config.dataset_repo_id,
        "teacher_view_key": "backview_image",
        "teacher_use_wrist_image": False,
        "assets_dir": config.assets_dir,
        "checkpoint_base_dir": config.output_dir,
        "align_layers": (10,),
        "batch_size": config.batch_size,
        "num_workers": 0,
        "fsdp_devices": 1,
        "seed": config.seed,
        "wandb_enabled": False,
    }
    acpd_config = DistillTrainConfig(**common, exact_contribution_fusion=True)
    sft_config = dataclasses.replace(acpd_config, exact_contribution_fusion=False)
    acpd_train_config = _make_student_train_config(acpd_config, create_acpd_heads=False)
    sft_train_config = _make_student_train_config(sft_config, create_acpd_heads=False)
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    rng = jax.random.key(config.seed)
    sft_rng, acpd_rng, sample_rng = jax.random.split(rng, 3)
    sft_time_bins, sft_metrics, sft_episodes = _evaluate_model(
        config, acpd_config, sft_train_config, config.sft_params, sft_rng, sample_rng, mesh, data_sharding, replicated
    )
    acpd_time_bins, acpd_metrics, acpd_episodes = _evaluate_model(
        config,
        acpd_config,
        acpd_train_config,
        config.acpd_params,
        acpd_rng,
        sample_rng,
        mesh,
        data_sharding,
        replicated,
    )
    if not (np.array_equal(sft_time_bins, acpd_time_bins) and np.array_equal(sft_episodes, acpd_episodes)):
        raise RuntimeError("SFT and ACPD passes did not replay the same sample order.")
    time_bins = sft_time_bins
    episodes = sft_episodes
    errors = np.column_stack(
        [
            sft_metrics[:, 0],
            acpd_metrics[:, 0],
            sft_metrics[:, 1],
            acpd_metrics[:, 1],
            sft_metrics[:, 2],
            acpd_metrics[:, 2],
        ]
    )
    if not np.isfinite(errors).all():
        raise FloatingPointError("Non-finite flow errors in probe.")
    summary = {
        "scope": "Training-set offline fit audit; not held-out validation or benchmark success.",
        "config": dataclasses.asdict(config),
        "pooled": _summarize_bin(errors, episodes, config.seed),
        "time_bins": {
            str((index + 0.5) / 5): _summarize_bin(
                errors[time_bins == index], episodes[time_bins == index], config.seed
            )
            for index in range(5)
        },
    }
    np.savez(output_dir / "paired_errors.npz", time_bin=time_bins, episode_index=episodes, errors=errors)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("Saved paired flow-time audit to %s", output_dir)


if __name__ == "__main__":
    main(tyro.cli(FlowTimeProbeConfig))
