"""Measures ACPD-v2 gradient conflict without updating model parameters."""

import dataclasses
import functools
import json
import logging
import pathlib
import platform
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.sharding as sharding
from scripts.distillation_acpd.train_distill import DistillTrainConfig
from scripts.distillation_acpd.train_distill import _add_scaled_gradients
from scripts.distillation_acpd.train_distill import _create_paired_data_loader
from scripts.distillation_acpd.train_distill import _init_frozen_model_state
from scripts.distillation_acpd.train_distill import _make_student_train_config
from scripts.distillation_acpd.train_distill import _make_teacher_train_config
from scripts.distillation_acpd.train_distill import _scale_gradients
from scripts.distillation_acpd.train_distill import compute_gradients
from scripts.train import init_logging
from scripts.train import init_train_state


@dataclasses.dataclass(frozen=True)
class GradientConflictProbeConfig:
    """Configuration for one checkpoint gradient audit."""

    checkpoint_label: str
    student_params: str
    teacher_params: str
    assets_dir: str
    output_dir: str
    dataset_repo_id: str = "libero_multiview_tuned_6view_lerobot"
    seed: int = 42
    batch_size: int = 32
    num_batches: int = 200
    num_workers: int = 0
    fsdp_devices: int = 2
    acpd_loss_weight: float = 0.2
    action_corr_loss_weight: float = 0.5


def _tree_dot(first: Any, second: Any) -> at.Array:
    products = jax.tree.leaves(
        jax.tree.map(
            lambda x, y: jnp.vdot(x.astype(jnp.float32), y.astype(jnp.float32)),
            first,
            second,
        )
    )
    return sum(products, start=jnp.asarray(0.0, dtype=jnp.float32))


def _gradient_pair_metrics(task_grads: Any, auxiliary_grads: Any) -> dict[str, at.Array]:
    task_norm = optax.global_norm(task_grads).astype(jnp.float32)
    auxiliary_norm = optax.global_norm(auxiliary_grads).astype(jnp.float32)
    cosine = _tree_dot(task_grads, auxiliary_grads) / jnp.maximum(task_norm * auxiliary_norm, 1e-12)
    return {
        "cosine": cosine,
        "conflict": (cosine < 0).astype(jnp.float32),
        "norm": auxiliary_norm,
        "norm_ratio": auxiliary_norm / jnp.maximum(task_norm, 1e-12),
    }


def compute_gradient_metrics(
    distill_config: DistillTrainConfig,
    flow_grads: Any,
    contribution_grads: Any,
    action_corr_grads: Any,
    info: dict[str, at.Array],
) -> dict[str, at.Array]:
    """Computes loss-specific LoRA gradient relationships for one batch."""
    lora_filter = nnx_utils.PathRegex(".*lora.*")
    flow_lora = flow_grads.filter(lora_filter)
    contribution_lora = _scale_gradients(
        contribution_grads.filter(lora_filter),
        scale=distill_config.acpd_loss_weight,
    )
    action_corr_lora = _scale_gradients(
        action_corr_grads.filter(lora_filter),
        scale=distill_config.action_corr_loss_weight,
    )
    privileged_lora = _add_scaled_gradients(contribution_lora, action_corr_lora, scale=1.0)

    metrics = {
        "flow_norm": optax.global_norm(flow_lora),
        "supervised_loss": info["supervised_loss"],
        "acpd_loss": info["acpd_loss"],
        "action_corr_loss": info["action_corr_loss"],
    }
    for name, grads in (
        ("contribution", contribution_lora),
        ("action_corr", action_corr_lora),
        ("privileged", privileged_lora),
    ):
        metrics.update({f"{name}_{key}": value for key, value in _gradient_pair_metrics(flow_lora, grads).items()})
    return metrics


def _summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def _write_results(config: GradientConflictProbeConfig, rows: list[dict[str, float]]) -> None:
    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_batch.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")

    metric_names = sorted(key for key in rows[0] if key != "batch")
    summary = {
        "checkpoint_label": config.checkpoint_label,
        "student_params": config.student_params,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "num_batches": config.num_batches,
        "acpd_loss_weight": config.acpd_loss_weight,
        "action_corr_loss_weight": config.action_corr_loss_weight,
        "metrics": {name: _summarize([row[name] for row in rows]) for name in metric_names},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_distill_config(config: GradientConflictProbeConfig) -> DistillTrainConfig:
    return DistillTrainConfig(
        name="pi05_libero_backview_acpd_v2_layer10",
        exp_name=f"gradient_conflict_{config.checkpoint_label}",
        student_config_name="pi05_libero_backview_lora",
        student_init_params=config.student_params,
        teacher_params=config.teacher_params,
        dataset_repo_id=config.dataset_repo_id,
        student_view_key="backview_image",
        teacher_view_key="agentview_image",
        teacher_use_wrist_image=True,
        wrist_view_key="wrist_image",
        supervised_loss_weight=1.0,
        acpd_loss_weight=config.acpd_loss_weight,
        action_corr_loss_weight=config.action_corr_loss_weight,
        align_layers=(10,),
        exact_contribution_fusion=True,
        assets_dir=config.assets_dir,
        asset_id="libero_multiview",
        checkpoint_base_dir=config.output_dir,
        seed=config.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        fsdp_devices=config.fsdp_devices,
        wandb_enabled=False,
    )


def main(config: GradientConflictProbeConfig) -> None:
    """Runs a read-only gradient audit for one checkpoint."""
    init_logging()
    logging.info("Running gradient audit on %s", platform.node())
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size {config.batch_size} must be divisible by {jax.device_count()} devices.")

    distill_config = _make_distill_config(config)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    student_config = _make_student_train_config(distill_config, create_acpd_heads=False)
    teacher_config = _make_teacher_train_config(distill_config, student_config)

    rng = jax.random.key(config.seed)
    audit_rng, student_rng, teacher_rng = jax.random.split(rng, 3)
    student_state, student_sharding = init_train_state(student_config, student_rng, mesh, resume=False)
    jax.block_until_ready(student_state)
    teacher_state, teacher_sharding = _init_frozen_model_state(
        teacher_config,
        teacher_rng,
        mesh,
        config.teacher_params,
    )
    jax.block_until_ready(teacher_state)

    data_loader = _create_paired_data_loader(
        distill_config,
        student_config,
        sharding_=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    component_gradients = jax.jit(
        functools.partial(compute_gradients, distill_config, student_config),
        in_shardings=(
            replicated_sharding,
            student_sharding,
            teacher_sharding,
            data_sharding,
            replicated_sharding,
            replicated_sharding,
        ),
    )
    gradient_metrics = jax.jit(functools.partial(compute_gradient_metrics, distill_config))

    rows = []
    with sharding.set_mesh(mesh):
        for batch_index in range(config.num_batches):
            batch = next(data_iter)
            batch_rng = jax.random.fold_in(audit_rng, batch_index)
            micro_step = jnp.asarray(0, dtype=jnp.uint32)
            flow_grads, info = component_gradients(
                batch_rng,
                student_state,
                teacher_state,
                batch,
                micro_step,
                jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32),
            )
            contribution_grads, _ = component_gradients(
                batch_rng,
                student_state,
                teacher_state,
                batch,
                micro_step,
                jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),
            )
            action_corr_grads, _ = component_gradients(
                batch_rng,
                student_state,
                teacher_state,
                batch,
                micro_step,
                jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32),
            )
            metrics = gradient_metrics(flow_grads, contribution_grads, action_corr_grads, info)
            row = {key: float(value) for key, value in jax.device_get(metrics).items()}
            row["batch"] = batch_index
            rows.append(row)
            if batch_index == 0 or (batch_index + 1) % 10 == 0:
                logging.info(
                    "Batch %d/%d: privileged cosine=%.4f, conflict=%.0f, norm ratio=%.4f",
                    batch_index + 1,
                    config.num_batches,
                    row["privileged_cosine"],
                    row["privileged_conflict"],
                    row["privileged_norm_ratio"],
                )

    _write_results(config, rows)
    logging.info("Saved gradient audit to %s", config.output_dir)


if __name__ == "__main__":
    main(tyro.cli(GradientConflictProbeConfig))
