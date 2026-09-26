"""Pi0.5 model exposing training-only features for ACPD distillation."""

import dataclasses
from typing import Literal

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import gemma
from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at


def _prenorm(x: at.Array, eps: float = 1e-6) -> at.Array:
    """Applies parameter-free LayerNorm to pre-final-norm hidden states."""
    x = x.astype(jnp.float32)
    x = x - jnp.mean(x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)


def _exact_contribution_kernel_init(key, shape, dtype=jnp.float32):
    return jax.random.normal(key, shape, dtype) * (0.01 * shape[0] ** -0.5)


def layer_key(layer: int) -> str:
    """Returns a stable dictionary key for an action-expert layer."""
    return f"l{layer}".replace("-", "m")


class Projector(nnx.Module):
    """Three-layer MLP used by the ACPD selector and predictor."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc3 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, x: at.Array) -> at.Array:
        x = nnx.silu(self.fc1(x))
        x = nnx.silu(self.fc2(x))
        return self.fc3(x)


class AcpdHead(nnx.Module):
    """Selects a teacher cue and predicts it from the student hidden state."""

    def __init__(
        self,
        visual_dim: int,
        action_hidden_dim: int,
        memory_dim: int,
        projector_hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.memory_dim = memory_dim
        self.visual_memory_projector = Projector(visual_dim, projector_hidden_dim, memory_dim, rngs=rngs)
        self.action_memory_projector = Projector(action_hidden_dim, projector_hidden_dim, memory_dim, rngs=rngs)
        self.query_projector = Projector(action_hidden_dim, projector_hidden_dim, memory_dim, rngs=rngs)
        self.key_projector = nnx.Linear(memory_dim, memory_dim, rngs=rngs)
        self.value_projector = nnx.Linear(memory_dim, memory_dim, rngs=rngs)
        self.privileged_predictor = Projector(action_hidden_dim, projector_hidden_dim, memory_dim, rngs=rngs)

    def __call__(
        self,
        student_hidden: at.Array,
        teacher_visual_tokens: at.Array,
        teacher_hidden: at.Array,
        *,
        detach_query: bool = True,
    ) -> tuple[at.Array, at.Array, at.Array]:
        teacher_visual_tokens = jax.lax.stop_gradient(teacher_visual_tokens.astype(jnp.float32))
        teacher_hidden = jax.lax.stop_gradient(teacher_hidden.astype(jnp.float32))
        student_hidden = student_hidden.astype(jnp.float32)

        visual_memory = self.visual_memory_projector(teacher_visual_tokens)
        action_memory = self.action_memory_projector(teacher_hidden)
        memory = jnp.concatenate([visual_memory, action_memory], axis=1)

        query_source = jax.lax.stop_gradient(student_hidden) if detach_query else student_hidden
        query = self.query_projector(query_source)
        key = self.key_projector(memory)
        value = self.value_projector(memory)
        logits = jnp.einsum("bhd,bnd->bhn", query, key) * (self.memory_dim**-0.5)
        attention = jax.nn.softmax(logits, axis=-1)
        privileged_cue = jnp.einsum("bhn,bnd->bhd", attention, value)
        predicted_cue = self.privileged_predictor(student_hidden)
        return predicted_cue, privileged_cue, attention


class ExactContributionHead(nnx.Module):
    """Predicts two teacher-view contributions and fuses them at inference."""

    def __init__(self, hidden_dim: int, *, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        self.predictor = nnx.Linear(
            hidden_dim,
            2 * hidden_dim,
            kernel_init=_exact_contribution_kernel_init,
            rngs=rngs,
        )
        self.gate = nnx.Param(jnp.zeros((), dtype=jnp.float32))

    def predict(self, student_hidden: at.Array) -> at.Array:
        prediction = self.predictor(student_hidden.astype(jnp.float32))
        return einops.rearrange(prediction, "b h (v d) -> b v h d", v=2, d=self.hidden_dim)

    def fuse(
        self,
        final_hidden: at.Array,
        student_hidden: at.Array,
        *,
        detach_prediction: bool = True,
    ) -> at.Array:
        predicted = self.predict(student_hidden)
        if detach_prediction:
            predicted = jax.lax.stop_gradient(predicted)
        predicted_residual = jnp.sum(predicted, axis=1).astype(final_hidden.dtype)
        gate = jnp.tanh(self.gate.value).astype(final_hidden.dtype)
        return final_hidden + gate * predicted_residual


@dataclasses.dataclass(frozen=True)
class AcpdPi0Config(pi0_config.Pi0Config):
    align_layers: int | tuple[int, ...] = (6, 12)
    acpd_memory_dim: int = 1024
    acpd_projector_hidden_dim: int = 2048
    create_acpd_heads: bool = True
    exact_contribution_fusion: bool = False
    exact_contribution_injection: bool = True
    exact_contribution_task_gradient: bool = False
    exact_contribution_fusion_location: Literal["final", "aligned_attention"] = "final"

    @override
    def create(self, rng: at.KeyArrayLike) -> "AcpdPi0":
        return AcpdPi0(self, rngs=nnx.Rngs(rng))


class AcpdPi0(pi0.Pi0):
    """Pi0.5 exposing ACPD features and optional deployed cue fusion."""

    def __init__(self, config: AcpdPi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)

        paligemma_config = gemma.get_config(config.paligemma_variant)
        action_expert_config = gemma.get_config(config.action_expert_variant)
        layers = config.align_layers
        self.align_layers = tuple(layers) if isinstance(layers, list | tuple) else (layers,)
        self.action_expert_depth = action_expert_config.depth
        self.exact_contribution_fusion = config.exact_contribution_fusion
        self.exact_contribution_injection = config.exact_contribution_injection
        self.exact_contribution_task_gradient = config.exact_contribution_task_gradient
        self.exact_contribution_fusion_location = config.exact_contribution_fusion_location
        if config.create_acpd_heads:
            self.acpd_aux_heads = nnx.Dict(
                {
                    layer_key(layer): AcpdHead(
                        paligemma_config.width,
                        action_expert_config.width,
                        config.acpd_memory_dim,
                        config.acpd_projector_hidden_dim,
                        rngs=rngs,
                    )
                    for layer in self.align_layers
                }
            )
        if self.exact_contribution_fusion:
            if len(self.align_layers) != 1:
                raise ValueError("Exact contribution fusion requires exactly one alignment layer.")
            self.exact_contribution_head = ExactContributionHead(
                action_expert_config.width,
                rngs=rngs,
            )

    @staticmethod
    def _layer_index(layer: int, depth: int) -> int:
        if layer == -1:
            return depth - 1
        if layer > 0:
            return layer - 1
        return depth + layer

    @at.typecheck
    def compute_train_outputs(
        self,
        rng: at.KeyArrayLike | None,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        train: bool = False,
        include_final_hidden: bool = False,
    ) -> tuple[
        at.Float[at.Array, "b ah ad"],
        at.Float[at.Array, "b s ve"],
        list[at.Float[at.Array, "b ah ae"]],
    ]:
        observation = _model.preprocess_observation(rng, observation, train=train)
        prefix_tokens, prefix_mask, prefix_ar_mask, image_tokens, _ = self._embed_prefix_with_image_tokens(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, noisy_actions, timestep
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        attention_intermediates = None
        if (
            self.exact_contribution_fusion
            and self.exact_contribution_injection
            and self.exact_contribution_fusion_location == "aligned_attention"
        ):
            (_, suffix_out), _, action_intermediates, attention_intermediates = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                positions,
                attn_mask,
                [None, adarms_cond],
                self.exact_contribution_head.predictor.kernel.value,
                self.exact_contribution_head.predictor.bias.value,
                self.exact_contribution_head.gate.value,
                self._layer_index(self.align_layers[0], self.action_expert_depth),
                self.action_horizon,
                method="forward_with_exact_contribution_fusion",
            )
        else:
            (_, suffix_out), _, action_intermediates = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
                method="forward_with_intermediates",
            )
        depth = action_intermediates.shape[0]
        hidden_intermediates = attention_intermediates if attention_intermediates is not None else action_intermediates
        hiddens = []
        for layer in self.align_layers:
            if layer == -1:
                hiddens.append(suffix_out[:, -self.action_horizon :])
            else:
                hidden = hidden_intermediates[self._layer_index(layer, depth)][:, -self.action_horizon :]
                hiddens.append(hidden if attention_intermediates is not None else _prenorm(hidden))
        final_hidden = suffix_out[:, -self.action_horizon :]
        if (
            self.exact_contribution_fusion
            and self.exact_contribution_injection
            and self.exact_contribution_fusion_location == "final"
        ):
            final_hidden = self.exact_contribution_head.fuse(
                final_hidden,
                hiddens[0],
                detach_prediction=not (train and self.exact_contribution_task_gradient),
            )
        v_t = self.action_out_proj(final_hidden)
        privileged_visual_tokens = jnp.concatenate(
            [image_tokens["base_0_rgb"], image_tokens["left_wrist_0_rgb"]], axis=1
        )
        if include_final_hidden:
            hiddens.append(suffix_out[:, -self.action_horizon :])
        return v_t, privileged_visual_tokens, hiddens

    @override
    def _decode_action_velocity(
        self,
        suffix_tokens: at.Array,
        full_attn_mask: at.Array,
        positions: at.Array,
        kv_cache: gemma.KVCache,
        adarms_cond: at.Array | None,
    ) -> at.Array:
        if not self.exact_contribution_fusion or not self.exact_contribution_injection:
            return super()._decode_action_velocity(
                suffix_tokens,
                full_attn_mask,
                positions,
                kv_cache,
                adarms_cond,
            )
        if self.exact_contribution_fusion_location == "aligned_attention":
            (prefix_out, suffix_out), _, _, _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                positions,
                full_attn_mask,
                [None, adarms_cond],
                self.exact_contribution_head.predictor.kernel.value,
                self.exact_contribution_head.predictor.bias.value,
                self.exact_contribution_head.gate.value,
                self._layer_index(self.align_layers[0], self.action_expert_depth),
                self.action_horizon,
                kv_cache=kv_cache,
                method="forward_with_exact_contribution_fusion",
            )
            assert prefix_out is None
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        (prefix_out, suffix_out), _, action_intermediates = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
            method="forward_with_intermediates",
        )
        assert prefix_out is None
        layer_index = self._layer_index(self.align_layers[0], action_intermediates.shape[0])
        student_hidden = _prenorm(action_intermediates[layer_index][:, -self.action_horizon :])
        final_hidden = self.exact_contribution_head.fuse(
            suffix_out[:, -self.action_horizon :],
            student_hidden,
        )
        return self.action_out_proj(final_hidden)

    @at.typecheck
    def compute_attention_contributions(
        self,
        rng: at.KeyArrayLike | None,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        train: bool = False,
    ):
        """Returns exact agentview and wrist contributions to action attention."""
        observation = _model.preprocess_observation(rng, observation, train=train)
        prefix_tokens, prefix_mask, prefix_ar_mask, _, image_slices = self._embed_prefix_with_image_tokens(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, noisy_actions, timestep
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        source_masks = jnp.zeros((2, input_mask.shape[1]), dtype=bool)
        for source_index, name in enumerate(("base_0_rgb", "left_wrist_0_rgb")):
            start, end = image_slices[name]
            source_masks = source_masks.at[source_index, start:end].set(True)

        (_, suffix_out), _, _, diagnostics = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            source_masks=source_masks,
            adarms_cond=[None, adarms_cond],
            method="forward_with_attention_contributions",
        )
        contributions, shuffled_contributions, total_attention = diagnostics
        layer_indices = [self._layer_index(layer, contributions.shape[0]) for layer in self.align_layers]
        contributions = jnp.stack([contributions[index] for index in layer_indices], axis=0)
        shuffled_contributions = jnp.stack([shuffled_contributions[index] for index in layer_indices], axis=0)
        total_attention = jnp.stack([total_attention[index] for index in layer_indices], axis=0)

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        contributions = jnp.transpose(contributions[:, :, :, -self.action_horizon :], (2, 0, 1, 3, 4))
        shuffled_contributions = jnp.transpose(shuffled_contributions[:, :, :, -self.action_horizon :], (2, 0, 1, 3, 4))
        total_attention = jnp.transpose(total_attention[:, :, -self.action_horizon :], (1, 0, 2, 3))
        return v_t, contributions, shuffled_contributions, total_attention

    def _embed_prefix_with_image_tokens(self, observation: _model.Observation):
        input_mask = []
        ar_mask = []
        tokens = []
        image_tokens_by_name = {}
        image_slices = {}
        token_offset = 0

        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
            image_tokens_by_name[name] = image_tokens
            image_slices[name] = (token_offset, token_offset + image_tokens.shape[1])
            token_offset += image_tokens.shape[1]
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(observation.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        if observation.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(observation.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        return (
            jnp.concatenate(tokens, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.asarray(ar_mask),
            image_tokens_by_name,
            image_slices,
        )
