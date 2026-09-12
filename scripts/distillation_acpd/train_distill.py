"""Trains a weak-view pi0.5 student with ACPD distillation."""

from collections.abc import Sequence
import dataclasses
import functools
import logging
import pathlib
import platform
from typing import Any

import etils.epath as epath
from flax import struct
from flax import traverse_util
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_distill_acpd as pi0_distill_acpd
from openpi.policies import libero_policy
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
from scripts.train import _load_weights_and_validate
from scripts.train import init_logging
from scripts.train import init_train_state
from scripts.train import init_wandb


@dataclasses.dataclass(frozen=True)
class DistillTrainConfig:
    """Configuration for one weak-view ACPD student."""

    name: str = "pi05_libero_backview_acpd_lora"
    project_name: str = "openpi"
    exp_name: str = "pi05_libero_backview_acpd_lora_fsdp4_bs32_30k"

    student_config_name: str = "pi05_libero_backview_lora"
    student_init_params: str = tyro.MISSING
    teacher_params: str = tyro.MISSING
    dataset_repo_id: str = "libero_multiview_tuned_6view_lerobot"

    student_view_key: str = "backview_image"
    teacher_view_key: str = "agentview_image"
    student_use_wrist_image: bool = False
    teacher_use_wrist_image: bool = True
    wrist_view_key: str = "wrist_image"

    supervised_loss_weight: float = 1.0
    acpd_loss_weight: float = 0.2
    acpd_variance_loss_weight: float = 0.1
    action_corr_loss_weight: float = 0.5
    align_layers: tuple[int, ...] = (6, 12)
    acpd_memory_dim: int = 1024
    acpd_projector_hidden_dim: int = 2048
    acpd_detach_query: bool = True

    assets_dir: str = tyro.MISSING
    asset_id: str = "libero_multiview"
    checkpoint_base_dir: str = tyro.MISSING
    seed: int = 42
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_workers: int = 8
    num_train_steps: int = 30_000
    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6
    log_interval: int = 100
    save_interval: int = 5_000
    keep_period: int | None = 5_000
    save_final_checkpoint: bool = True
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    fsdp_devices: int = 4

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp-name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()


@struct.dataclass
class FrozenModelState:
    """Parameters and graph definition for the frozen teacher."""

    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]


@dataclasses.dataclass(frozen=True)
class AcpdCheckpointWeightLoader:
    """Loads base weights and keeps freshly initialized ACPD heads."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _weight_loaders.CheckpointWeightLoader(self.params_path).load(params)
        flat_loaded = traverse_util.flatten_dict(loaded_params, sep="/")
        flat_ref = traverse_util.flatten_dict(params, sep="/")
        for key, value in flat_ref.items():
            if key.startswith("acpd_aux_heads/") and key not in flat_loaded:
                flat_loaded[key] = value
        return traverse_util.unflatten_dict(flat_loaded, sep="/")


@dataclasses.dataclass(frozen=True)
class PairedLiberoDistillTransform:
    """Builds synchronized teacher and student inputs from one sample."""

    student_view_key: str
    teacher_view_key: str
    model_type: _model.ModelType
    norm_stats: dict[str, _transforms.NormStats]
    use_quantile_norm: bool
    model_transforms: Sequence[_transforms.DataTransformFn]
    student_use_wrist_image: bool = False
    teacher_use_wrist_image: bool = True
    wrist_view_key: str = "wrist_image"

    def __post_init__(self):
        object.__setattr__(
            self,
            "_student_transform",
            self._make_transform(use_wrist_image=self.student_use_wrist_image),
        )
        object.__setattr__(
            self,
            "_teacher_transform",
            self._make_transform(use_wrist_image=self.teacher_use_wrist_image),
        )

    def _make_transform(self, *, use_wrist_image: bool):
        return _transforms.compose(
            [
                libero_policy.LiberoInputs(
                    model_type=self.model_type,
                    use_wrist_image=use_wrist_image,
                ),
                _transforms.Normalize(self.norm_stats, use_quantiles=self.use_quantile_norm),
                *self.model_transforms,
            ]
        )

    def __call__(self, data: dict) -> dict:
        student, actions = self._make_view(
            data,
            self.student_view_key,
            self._student_transform,
            use_wrist_image=self.student_use_wrist_image,
        )
        teacher, _ = self._make_view(
            data,
            self.teacher_view_key,
            self._teacher_transform,
            use_wrist_image=self.teacher_use_wrist_image,
        )
        return {"student": student, "teacher": teacher, "actions": actions}

    def _make_view(
        self,
        data: dict,
        view_key: str,
        transform: _transforms.DataTransformFn,
        *,
        use_wrist_image: bool,
    ) -> tuple[dict, np.ndarray]:
        view_data = {
            "observation/image": data[view_key],
            "observation/state": data["state"],
            "actions": data["actions"],
            "prompt": data["prompt"],
        }
        if use_wrist_image:
            view_data["observation/wrist_image"] = data[self.wrist_view_key]
        model_input = transform(view_data)
        actions = model_input.pop("actions")
        return model_input, actions


class PairedDataLoader:
    """Converts transformed dictionaries to OpenPI model inputs."""

    def __init__(self, data_config: _config.DataConfig, torch_loader: _data_loader.TorchDataLoader):
        self._data_config = data_config
        self._torch_loader = torch_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._torch_loader:
            yield (
                _model.Observation.from_dict(batch["teacher"]),
                _model.Observation.from_dict(batch["student"]),
                batch["actions"],
            )


def _make_distill_model_config(
    model_config: _model.BaseModelConfig,
    config: DistillTrainConfig,
    *,
    create_acpd_heads: bool,
) -> pi0_distill_acpd.AcpdPi0Config:
    if not isinstance(model_config, pi0_config.Pi0Config):
        raise ValueError(f"ACPD only supports Pi0Config, got {type(model_config).__name__}.")
    base_fields = {field.name: getattr(model_config, field.name) for field in dataclasses.fields(pi0_config.Pi0Config)}
    return pi0_distill_acpd.AcpdPi0Config(
        **base_fields,
        align_layers=config.align_layers,
        acpd_memory_dim=config.acpd_memory_dim,
        acpd_projector_hidden_dim=config.acpd_projector_hidden_dim,
        create_acpd_heads=create_acpd_heads,
    )


def _make_student_train_config(config: DistillTrainConfig) -> _config.TrainConfig:
    base = _config.get_config(config.student_config_name)
    data = dataclasses.replace(
        base.data,
        repo_id=config.dataset_repo_id,
        assets=_config.AssetsConfig(assets_dir=config.assets_dir, asset_id=config.asset_id),
    )
    return dataclasses.replace(
        base,
        name=config.name,
        project_name=config.project_name,
        exp_name=config.exp_name,
        model=_make_distill_model_config(base.model, config, create_acpd_heads=True),
        weight_loader=AcpdCheckpointWeightLoader(config.student_init_params),
        data=data,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=config.warmup_steps,
            peak_lr=config.peak_lr,
            decay_steps=config.decay_steps,
            decay_lr=config.decay_lr,
        ),
        checkpoint_base_dir=config.checkpoint_base_dir,
        seed=config.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_train_steps=config.num_train_steps,
        log_interval=config.log_interval,
        save_interval=config.save_interval,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        wandb_enabled=config.wandb_enabled,
        fsdp_devices=config.fsdp_devices,
        ema_decay=None,
    )


def _make_teacher_train_config(
    config: DistillTrainConfig,
    student_config: _config.TrainConfig,
) -> _config.TrainConfig:
    model_config = dataclasses.replace(
        student_config.model,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    return dataclasses.replace(
        student_config,
        model=_make_distill_model_config(model_config, config, create_acpd_heads=False),
        weight_loader=_weight_loaders.CheckpointWeightLoader(config.teacher_params),
    )


def _create_paired_data_loader(
    config: DistillTrainConfig,
    train_config: _config.TrainConfig,
    *,
    sharding_: jax.sharding.Sharding,
    shuffle: bool,
) -> PairedDataLoader:
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    logging.info("data_config: %s", data_config)
    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("ACPD only supports LeRobot datasets.")
    if data_config.repo_id != "fake" and data_config.norm_stats is None:
        raise ValueError("Normalization stats are required for ACPD training.")

    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            PairedLiberoDistillTransform(
                student_view_key=config.student_view_key,
                teacher_view_key=config.teacher_view_key,
                model_type=train_config.model.model_type,
                norm_stats=data_config.norm_stats or {},
                use_quantile_norm=data_config.use_quantile_norm,
                model_transforms=data_config.model_transforms.inputs,
                student_use_wrist_image=config.student_use_wrist_image,
                teacher_use_wrist_image=config.teacher_use_wrist_image,
                wrist_view_key=config.wrist_view_key,
            )
        ],
    )
    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=train_config.batch_size // jax.process_count(),
        sharding=sharding_,
        shuffle=shuffle,
        num_batches=None,
        num_workers=train_config.num_workers,
        seed=train_config.seed,
        framework="jax",
    )
    return PairedDataLoader(data_config, torch_loader)


def _init_frozen_model_state(
    train_config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    params_path: str,
) -> tuple[FrozenModelState, Any]:
    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> FrozenModelState:
        _, model_rng = jax.random.split(rng)
        model = train_config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx_utils.state_map(
            nnx.state(model),
            nnx.Param,
            lambda param: param.replace(param.value.astype(jnp.bfloat16)),
        )
        return FrozenModelState(params=params, model_def=nnx.graphdef(model))

    state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(state_shape, mesh, log=True)
    partial_params = _load_weights_and_validate(
        _weight_loaders.CheckpointWeightLoader(params_path),
        state_shape.params.to_pure_dict(),
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    frozen_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=(replicated_sharding, replicated_sharding),
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return frozen_state, state_sharding


def _normalize_last_dim(x: at.Array, eps: float = 1e-6) -> at.Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def _action_corr_loss(student_v_t: at.Array, teacher_v_t: at.Array, *, task_action_dim: int = 7) -> at.Array:
    student = student_v_t[..., :task_action_dim].astype(jnp.float32).reshape((student_v_t.shape[0], -1))
    teacher = jax.lax.stop_gradient(teacher_v_t[..., :task_action_dim].astype(jnp.float32)).reshape(
        (teacher_v_t.shape[0], -1)
    )
    student = _normalize_last_dim(student - jnp.mean(student, axis=-1, keepdims=True))
    teacher = _normalize_last_dim(teacher - jnp.mean(teacher, axis=-1, keepdims=True))
    return 1.0 - jnp.mean(jnp.sum(student * teacher, axis=-1))


def _per_sample_prediction_error(prediction: at.Array, target: at.Array) -> at.Array:
    """Computes mean squared prediction error for each batch item."""
    return jnp.mean(
        jnp.square(prediction.astype(jnp.float32) - target.astype(jnp.float32)),
        axis=tuple(range(1, prediction.ndim)),
    )


def _acpd_prediction_loss(
    predicted_cue: at.Array,
    privileged_cue: at.Array,
) -> tuple[at.Array, at.Array]:
    predicted_cue = _normalize_last_dim(predicted_cue.astype(jnp.float32))
    privileged_cue = _normalize_last_dim(privileged_cue.astype(jnp.float32))
    cosine = jnp.sum(predicted_cue * privileged_cue, axis=-1)
    return jnp.mean(1.0 - cosine), jnp.mean(cosine)


def _acpd_variance_loss(privileged_cue: at.Array, eps: float = 1e-4) -> tuple[at.Array, at.Array]:
    cue = _normalize_last_dim(privileged_cue.astype(jnp.float32))
    cue = cue * jnp.sqrt(jnp.asarray(cue.shape[-1], dtype=jnp.float32))
    cue = cue.reshape((-1, cue.shape[-1]))
    feature_std = jnp.sqrt(jnp.var(cue, axis=0) + eps)
    return jnp.mean(jax.nn.relu(1.0 - feature_std)), jnp.mean(feature_std)


def _attention_diagnostics(attention: at.Array, num_visual_tokens: int) -> dict[str, at.Array]:
    attention = attention.astype(jnp.float32)
    entropy = -jnp.sum(attention * jnp.log(attention + 1e-8), axis=-1)
    return {
        "attention_entropy": jnp.mean(entropy),
        "attention_visual_mass": jnp.mean(jnp.sum(attention[:, :, :num_visual_tokens], axis=-1)),
        "attention_action_mass": jnp.mean(jnp.sum(attention[:, :, num_visual_tokens:], axis=-1)),
        "attention_top1_mass": jnp.mean(jnp.max(attention, axis=-1)),
    }


def _micro_step_train_rng(
    rng: at.KeyArrayLike,
    optimizer_step: at.Int[at.Array, ""],
    micro_step: at.Int[at.Array, ""],
    *,
    gradient_accumulation_steps: int,
) -> at.KeyArrayLike:
    train_rng = jax.random.fold_in(rng, optimizer_step)
    if gradient_accumulation_steps > 1:
        train_rng = jax.random.fold_in(train_rng, micro_step)
    return train_rng


def compute_gradients(
    config: DistillTrainConfig,
    train_config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    student_state: training_utils.TrainState,
    teacher_state: FrozenModelState,
    batch: tuple[_model.Observation, _model.Observation, _model.Actions],
    micro_step: at.Int[at.Array, ""],
) -> tuple[nnx.State, dict[str, at.Array]]:
    student_model = nnx.merge(student_state.model_def, student_state.params)
    teacher_model = nnx.merge(teacher_state.model_def, teacher_state.params)
    student_model.train()
    teacher_model.eval()
    train_rng = _micro_step_train_rng(
        rng,
        student_state.step,
        micro_step,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    preprocess_rng, noise_rng, time_rng = jax.random.split(train_rng, 3)
    teacher_observation, student_observation, actions = batch
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    target_v_t = noise - actions

    def loss_fn(model):
        student_v_t, _, student_hiddens = model.compute_train_outputs(
            preprocess_rng, student_observation, x_t, time, train=True
        )
        teacher_v_t, teacher_visual_tokens, teacher_hiddens = teacher_model.compute_train_outputs(
            preprocess_rng, teacher_observation, x_t, time, train=True
        )
        teacher_v_t = jax.lax.stop_gradient(teacher_v_t)
        supervised_loss = jnp.mean(jnp.square(student_v_t - target_v_t))
        action_corr_loss = _action_corr_loss(student_v_t, teacher_v_t)
        student_task_error = _per_sample_prediction_error(student_v_t[..., :7], target_v_t[..., :7])
        teacher_task_error = _per_sample_prediction_error(teacher_v_t[..., :7], target_v_t[..., :7])

        layer_losses = []
        prediction_losses = []
        variance_losses = []
        per_layer = {}
        for layer, student_hidden, teacher_hidden in zip(
            model.align_layers,
            student_hiddens,
            teacher_hiddens,
            strict=True,
        ):
            head = model.acpd_aux_heads[pi0_distill_acpd.layer_key(layer)]
            predicted_cue, privileged_cue, attention = head(
                student_hidden,
                teacher_visual_tokens,
                teacher_hidden,
                detach_query=config.acpd_detach_query,
            )
            prediction_loss, cosine = _acpd_prediction_loss(predicted_cue, privileged_cue)
            variance_loss, feature_std = _acpd_variance_loss(privileged_cue)
            layer_loss = prediction_loss + config.acpd_variance_loss_weight * variance_loss
            diagnostics = _attention_diagnostics(attention, teacher_visual_tokens.shape[1])
            key = pi0_distill_acpd.layer_key(layer)
            per_layer[f"acpd_loss_{key}"] = layer_loss
            per_layer[f"acpd_prediction_loss_{key}"] = prediction_loss
            per_layer[f"acpd_variance_loss_{key}"] = variance_loss
            per_layer[f"acpd_cosine_{key}"] = cosine
            per_layer[f"acpd_feature_std_{key}"] = feature_std
            for metric, value in diagnostics.items():
                per_layer[f"acpd_{metric}_{key}"] = value
            layer_losses.append(layer_loss)
            prediction_losses.append(prediction_loss)
            variance_losses.append(variance_loss)

        acpd_loss = jnp.mean(jnp.stack(layer_losses))
        acpd_prediction_loss = jnp.mean(jnp.stack(prediction_losses))
        acpd_variance_loss = jnp.mean(jnp.stack(variance_losses))
        weighted_acpd_loss = config.acpd_loss_weight * acpd_loss
        loss = (
            config.supervised_loss_weight * supervised_loss
            + weighted_acpd_loss
            + config.action_corr_loss_weight * action_corr_loss
        )
        return loss, {
            "loss": loss,
            "supervised_loss": supervised_loss,
            "acpd_loss": acpd_loss,
            "acpd_prediction_loss": acpd_prediction_loss,
            "acpd_variance_loss": acpd_variance_loss,
            "weighted_acpd_loss": weighted_acpd_loss,
            "action_corr_loss": action_corr_loss,
            "student_task_loss": jnp.mean(student_task_error),
            "teacher_task_loss": jnp.mean(teacher_task_error),
            "teacher_better_ratio": jnp.mean((teacher_task_error < student_task_error).astype(jnp.float32)),
            "acpd_to_supervised_ratio": weighted_acpd_loss / jnp.maximum(supervised_loss, 1e-6),
            **per_layer,
        }

    diff_state = nnx.DiffState(0, train_config.trainable_filter)
    (_, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(student_model)
    return grads, info


def _scale_gradients(grads: nnx.State, *, scale: float) -> nnx.State:
    return jax.tree.map(lambda grad: grad * scale, grads)


def _add_scaled_gradients(accumulated_grads: nnx.State, grads: nnx.State, *, scale: float) -> nnx.State:
    return jax.tree.map(lambda accumulated, grad: accumulated + scale * grad, accumulated_grads, grads)


def _reduce_microbatch_infos(infos: list[dict[str, at.Array]]) -> dict[str, at.Array]:
    return jax.tree.map(jnp.mean, common_utils.stack_forest(infos))


def apply_gradients(
    config: DistillTrainConfig,
    train_config: _config.TrainConfig,
    student_state: training_utils.TrainState,
    grads: nnx.State,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    student_model = nnx.merge(student_state.model_def, student_state.params)
    params = student_state.params.filter(train_config.trainable_filter)
    updates, new_opt_state = student_state.tx.update(grads, student_state.opt_state, params)
    nnx.update(student_model, optax.apply_updates(params, updates))
    new_state = dataclasses.replace(
        student_state,
        step=student_state.step + 1,
        params=nnx.state(student_model),
        opt_state=new_opt_state,
    )

    selector_grads = grads.filter(
        nnx_utils.PathRegex(
            ".*acpd_aux_heads.*(visual_memory_projector|action_memory_projector|query_projector|key_projector|value_projector).*"
        )
    )
    predictor_grads = grads.filter(nnx_utils.PathRegex(".*acpd_aux_heads.*privileged_predictor.*"))
    lora_grads = grads.filter(nnx_utils.PathRegex(".*lora.*"))
    return new_state, {
        "grad_norm": optax.global_norm(grads),
        "selector_grad_norm": optax.global_norm(selector_grads),
        "predictor_grad_norm": optax.global_norm(predictor_grads),
        "lora_grad_norm": optax.global_norm(lora_grads),
        "learning_rate": train_config.lr_schedule.create()(student_state.step),
        "gradient_accumulation_steps": jnp.asarray(config.gradient_accumulation_steps, dtype=jnp.float32),
        "global_micro_batch_size": jnp.asarray(config.batch_size, dtype=jnp.float32),
        "effective_batch_size": jnp.asarray(
            config.batch_size * config.gradient_accumulation_steps,
            dtype=jnp.float32,
        ),
    }


def main(config: DistillTrainConfig):
    """Runs ACPD distillation."""
    init_logging()
    logging.info("Running on: %s", platform.node())
    if config.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    logging.info(
        "Training scale: global micro-batch=%d, accumulation=%d, effective batch=%d, optimizer steps=%d",
        config.batch_size,
        config.gradient_accumulation_steps,
        config.batch_size * config.gradient_accumulation_steps,
        config.num_train_steps,
    )
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, student_init_rng, teacher_init_rng = jax.random.split(rng, 3)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    student_config = _make_student_train_config(config)
    teacher_config = _make_teacher_train_config(config, student_config)

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        student_config.checkpoint_dir,
        keep_period=student_config.keep_period,
        overwrite=student_config.overwrite,
        resume=student_config.resume,
    )
    init_wandb(student_config, resuming=resuming, enabled=config.wandb_enabled)
    if config.wandb_enabled:
        wandb.config.update(dataclasses.asdict(config), allow_val_change=True)

    data_loader = _create_paired_data_loader(config, student_config, sharding_=data_sharding, shuffle=True)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info("Initialized paired data loader:\n%s", training_utils.array_tree_to_info(batch))

    student_state, student_state_sharding = init_train_state(
        student_config,
        student_init_rng,
        mesh,
        resume=resuming,
    )
    jax.block_until_ready(student_state)
    if resuming:
        student_state = _checkpoints.restore_state(checkpoint_manager, student_state, data_loader)

    teacher_state, teacher_state_sharding = _init_frozen_model_state(
        teacher_config,
        teacher_init_rng,
        mesh,
        config.teacher_params,
    )
    jax.block_until_ready(teacher_state)

    pcompute_gradients = jax.jit(
        functools.partial(compute_gradients, config, student_config),
        in_shardings=(
            replicated_sharding,
            student_state_sharding,
            teacher_state_sharding,
            data_sharding,
            replicated_sharding,
        ),
    )
    papply_gradients = jax.jit(
        functools.partial(apply_gradients, config, student_config),
        out_shardings=(student_state_sharding, replicated_sharding),
        donate_argnums=(0, 1),
    )
    gradient_scale = 1.0 / config.gradient_accumulation_steps
    pscale_gradients = jax.jit(
        functools.partial(_scale_gradients, scale=gradient_scale),
        donate_argnums=(0,),
    )
    paccumulate_gradients = jax.jit(
        functools.partial(_add_scaled_gradients, scale=gradient_scale),
        donate_argnums=(0, 1),
    )

    start_step = int(student_state.step)
    progress = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )
    infos = []
    for step in progress:
        microbatch_infos = []
        accumulated_grads = None
        with sharding.set_mesh(mesh):
            for micro_step in range(config.gradient_accumulation_steps):
                grads, microbatch_info = pcompute_gradients(
                    train_rng,
                    student_state,
                    teacher_state,
                    batch,
                    jnp.asarray(micro_step, dtype=jnp.uint32),
                )
                if accumulated_grads is None:
                    accumulated_grads = pscale_gradients(grads)
                else:
                    accumulated_grads = paccumulate_gradients(accumulated_grads, grads)
                microbatch_infos.append(microbatch_info)
                if micro_step < config.gradient_accumulation_steps - 1:
                    batch = next(data_iter)
            student_state, update_info = papply_gradients(student_state, accumulated_grads)

        info = {**_reduce_microbatch_infos(microbatch_infos), **update_info}
        infos.append(info)
        if step % config.log_interval == 0:
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            progress.write(f"Step {step}: " + ", ".join(f"{key}={value:.4f}" for key, value in reduced_info.items()))
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        periodic_save = step % config.save_interval == 0 and step > start_step
        final_save = config.save_final_checkpoint and step == config.num_train_steps - 1
        if periodic_save or final_save:
            _checkpoints.save_state(checkpoint_manager, student_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(tyro.cli(DistillTrainConfig))
