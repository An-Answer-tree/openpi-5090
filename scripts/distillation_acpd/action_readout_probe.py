"""Tests frozen contribution features with matched small action readouts."""

import dataclasses
import functools
import json
import logging
import os
import pathlib
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from lerobot.common.datasets import lerobot_dataset
import numpy as np
import optax
import tyro

from openpi.models import gemma
from openpi.models.pi0_distill_acpd import _prenorm
from openpi.training import sharding
from scripts.distillation_acpd.exact_attention_probe import _split_episodes
from scripts.distillation_acpd.train_distill import DistillTrainConfig
from scripts.distillation_acpd.train_distill import FrozenModelState
from scripts.distillation_acpd.train_distill import _create_paired_data_loader
from scripts.distillation_acpd.train_distill import _init_frozen_model_state
from scripts.distillation_acpd.train_distill import _make_student_train_config
from scripts.train import init_logging

ARMS = ("hidden_only", "hidden_layer10", "hidden_contribution")


@dataclasses.dataclass(frozen=True)
class ActionReadoutProbeConfig:
    """Configuration for the single-GPU frozen action readout probe."""

    student_params: str
    assets_dir: str
    output_dir: str
    dataset_repo_id: str = "libero_multiview_tuned_6view_lerobot"
    batch_size: int = 8
    num_train_steps: int = 500
    num_validation_batches: int = 128
    validation_fraction: float = 0.1
    hidden_width: int = 128
    learning_rate: float = 1e-3
    log_interval: int = 50
    bootstrap_samples: int = 2000
    seed: int = 42


def _make_inputs(final_hidden, layer_hidden, contributions):
    final_hidden = _prenorm(final_hidden)
    layer_hidden = _prenorm(layer_hidden)
    contributions = _prenorm(contributions)
    zeros = jnp.zeros_like(final_hidden)
    return jnp.stack(
        [
            jnp.concatenate([final_hidden, zeros, zeros], axis=-1),
            jnp.concatenate([final_hidden, layer_hidden, layer_hidden], axis=-1),
            jnp.concatenate([final_hidden, contributions[:, 0], contributions[:, 1]], axis=-1),
        ]
    )


def _extract_features(rng, state: FrozenModelState, batch):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    _, observation, actions = batch
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    noise = jax.random.normal(noise_rng, actions.shape)
    flow_time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    noisy_actions = flow_time[:, None, None] * noise + (1 - flow_time[:, None, None]) * actions
    velocity, _, hiddens = model.compute_train_outputs(
        preprocess_rng, observation, noisy_actions, flow_time, train=False, include_final_hidden=True
    )
    contributions = model.exact_contribution_head.predict(hiddens[0])
    inputs = _make_inputs(hiddens[-1], hiddens[0], contributions)
    return jax.tree.map(
        jax.lax.stop_gradient,
        (inputs, velocity[..., :7].astype(jnp.float32), (noise - actions)[..., :7], flow_time),
    )


def _init_heads(rng, input_dim: int, hidden_width: int):
    kernel = jax.random.normal(rng, (input_dim, hidden_width)) * input_dim**-0.5
    return {
        "input_kernel": jnp.broadcast_to(kernel, (len(ARMS), *kernel.shape)),
        "input_bias": jnp.zeros((len(ARMS), hidden_width)),
        "output_kernel": jnp.zeros((len(ARMS), hidden_width, 7)),
        "output_bias": jnp.zeros((len(ARMS), 7)),
    }


def _corrections(params, inputs):
    hidden = jax.nn.silu(
        jnp.einsum("rbhd,rdk->rbhk", inputs, params["input_kernel"]) + params["input_bias"][:, None, None]
    )
    return jnp.einsum("rbhk,rka->rbha", hidden, params["output_kernel"]) + params["output_bias"][:, None, None]


def _sample_errors(params, inputs, velocity, target):
    corrected = velocity[None] + _corrections(params, inputs)
    return jnp.mean(jnp.square(corrected - target[None]), axis=(2, 3)).T


def _head_loss(params, inputs, velocity, target):
    losses = jnp.mean(_sample_errors(params, inputs, velocity, target), axis=0)
    # Sum, rather than average across arms, preserves each independent head's loss scale.
    return jnp.sum(losses), losses


def _train_step(params, opt_state, inputs, velocity, target, *, optimizer):
    (_, losses), gradients = jax.value_and_grad(_head_loss, has_aux=True)(params, inputs, velocity, target)
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, losses


def _analyze(errors: np.ndarray, episodes: np.ndarray, *, bootstrap_samples: int, seed: int):
    episode_ids = np.unique(episodes)
    episode_errors = np.stack([errors[episodes == episode].mean(axis=0) for episode in episode_ids])
    means = episode_errors.mean(axis=0)
    indices = np.random.default_rng(seed).integers(len(episode_ids), size=(bootstrap_samples, len(episode_ids)))
    boot_means = episode_errors[indices].mean(axis=1)
    names = ("frozen_h9", *ARMS)
    comparisons = {}
    for index, control in enumerate(names[:-1]):
        difference = means[-1] - means[index]
        interval = np.percentile(boot_means[:, -1] - boot_means[:, index], [2.5, 97.5])
        comparisons[control] = {
            "mse_difference": float(difference),
            "paired_ci95": interval.tolist(),
            "relative_mse_reduction": float(-difference / max(means[index], 1e-12)),
        }
    passes = (
        all(
            comparisons[name]["relative_mse_reduction"] >= 0.01 and comparisons[name]["paired_ci95"][1] < 0
            for name in ARMS[:-1]
        )
        and means[-1] < means[0]
    )
    return {
        "frame_mean_mse": dict(zip(names, errors.mean(axis=0).tolist(), strict=True)),
        "episode_mean_mse": dict(zip(names, means.tolist(), strict=True)),
        "contribution_vs_control": comparisons,
        "validation_episodes_sampled": len(episode_ids),
        "passes_offline_screen": bool(passes),
        "benchmark_success_available": False,
    }


def main(config: ActionReadoutProbeConfig) -> None:
    """Trains only readouts; reads the source checkpoint and dataset in place."""
    init_logging()
    if jax.device_count() != 1:
        raise ValueError("This frozen probe requires exactly one visible JAX device.")
    started = time.monotonic()
    output_dir = pathlib.Path(config.output_dir) / os.environ.get("SLURM_JOB_ID", "local")
    output_dir.mkdir(parents=True, exist_ok=False)
    distill_config = DistillTrainConfig(
        name="pi05_libero_backview_acpd_v2_layer10",
        exp_name="h18_action_readout",
        student_init_params=config.student_params,
        teacher_params=config.student_params,
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
    train_episodes, validation_episodes = _split_episodes(
        metadata.total_episodes, config.validation_fraction, config.seed
    )
    manifest = {
        "config": dataclasses.asdict(config),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "arms": ARMS,
        "scope": "Adapter-held-out episodes; backbone saw these episodes. Not a benchmark success test.",
    }
    (output_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    rng = jax.random.key(config.seed)
    state_rng, head_rng, train_rng, validation_rng = jax.random.split(rng, 4)
    state, state_sharding = _init_frozen_model_state(train_config, state_rng, mesh, config.student_params)
    jax.block_until_ready(state)
    logging.info("Frozen H9 checkpoint loaded in place; no teacher or optimizer state loaded.")
    train_loader = _create_paired_data_loader(
        distill_config, train_config, sharding_=data_sharding, shuffle=True, episodes=train_episodes
    )
    hidden_dim = gemma.get_config(train_config.model.action_expert_variant).width
    params = _init_heads(head_rng, 3 * hidden_dim, config.hidden_width)
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    extract = jax.jit(_extract_features, in_shardings=(replicated, state_sharding, data_sharding))
    train_step = jax.jit(functools.partial(_train_step, optimizer=optimizer))
    sample_errors = jax.jit(_sample_errors)
    training_rows = []
    train_iterator = iter(train_loader)
    with sharding.set_mesh(mesh):
        with (output_dir / "training.jsonl").open("w", encoding="utf-8") as log:
            for step in range(config.num_train_steps):
                inputs, velocity, target, _ = extract(jax.random.fold_in(train_rng, step), state, next(train_iterator))
                if step == 0:
                    corrections = np.asarray(jax.device_get(_corrections(params, inputs)))
                    np.testing.assert_array_equal(corrections, np.zeros_like(corrections))
                    logging.info(
                        "Inputs %s; velocity %s; all readouts initially equal frozen H9.", inputs.shape, velocity.shape
                    )
                params, opt_state, losses = train_step(params, opt_state, inputs, velocity, target)
                training_rows.append(np.asarray(jax.device_get(losses)))
                if (step + 1) % config.log_interval == 0 or step == config.num_train_steps - 1:
                    losses = np.mean(training_rows, axis=0)
                    if not np.isfinite(losses).all():
                        raise FloatingPointError(f"Non-finite readout loss at step {step + 1}: {losses}")
                    row = {"step": step + 1, "elapsed_seconds": time.monotonic() - started}
                    row.update(dict(zip(ARMS, losses.tolist(), strict=True)))
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    training_rows.clear()
                    logging.info("Step %d/%d: readout MSE=%s", step + 1, config.num_train_steps, losses)

        validation_loader = _create_paired_data_loader(
            distill_config,
            train_config,
            sharding_=data_sharding,
            shuffle=True,
            episodes=validation_episodes,
            include_episode_index=True,
        )
        validation_iterator = iter(validation_loader)
        error_rows, episode_rows, time_rows = [], [], []
        for batch_index in range(config.num_validation_batches):
            *batch, episode_ids = next(validation_iterator)
            inputs, velocity, target, flow_time = extract(
                jax.random.fold_in(validation_rng, batch_index), state, tuple(batch)
            )
            baseline_error = jnp.mean(jnp.square(velocity - target), axis=(1, 2))
            errors = jnp.concatenate([baseline_error[:, None], sample_errors(params, inputs, velocity, target)], axis=1)
            error_rows.append(np.asarray(jax.device_get(errors)))
            episode_rows.append(np.asarray(jax.device_get(episode_ids)).reshape(-1))
            time_rows.append(np.asarray(jax.device_get(flow_time)))
            if (batch_index + 1) % 32 == 0:
                logging.info("Held-out batches %d/%d", batch_index + 1, config.num_validation_batches)

    errors, episodes = np.concatenate(error_rows), np.concatenate(episode_rows)
    if not np.isfinite(errors).all():
        raise FloatingPointError("Non-finite held-out errors.")
    summary = _analyze(errors, episodes, bootstrap_samples=config.bootstrap_samples, seed=config.seed)
    summary.update({"manifest": manifest, "elapsed_seconds": time.monotonic() - started})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez(
        output_dir / "validation_errors.npz", errors=errors, episode_index=episodes, flow_time=np.concatenate(time_rows)
    )
    np.savez(output_dir / "readout_params.npz", **jax.device_get(params))
    logging.info(
        "Offline screen=%s; saved only readouts and metrics to %s", summary["passes_offline_screen"], output_dir
    )


if __name__ == "__main__":
    main(tyro.cli(ActionReadoutProbeConfig))
