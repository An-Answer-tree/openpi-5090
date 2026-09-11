"""Pi0.5 model exposing training-only features for ACPD distillation."""

import dataclasses

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


@dataclasses.dataclass(frozen=True)
class AcpdPi0Config(pi0_config.Pi0Config):
    align_layers: int | tuple[int, ...] = (6, 12)
    acpd_memory_dim: int = 1024
    acpd_projector_hidden_dim: int = 2048
    create_acpd_heads: bool = True

    @override
    def create(self, rng: at.KeyArrayLike) -> "AcpdPi0":
        return AcpdPi0(self, rngs=nnx.Rngs(rng))


class AcpdPi0(pi0.Pi0):
    """Pi0.5 with ACPD features and training-only auxiliary heads."""

    def __init__(self, config: AcpdPi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)

        paligemma_config = gemma.get_config(config.paligemma_variant)
        action_expert_config = gemma.get_config(config.action_expert_variant)
        layers = config.align_layers
        self.align_layers = tuple(layers) if isinstance(layers, list | tuple) else (layers,)
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
    ) -> tuple[
        at.Float[at.Array, "b ah ad"],
        at.Float[at.Array, "b s ve"],
        list[at.Float[at.Array, "b ah ae"]],
    ]:
        observation = _model.preprocess_observation(rng, observation, train=train)
        prefix_tokens, prefix_mask, prefix_ar_mask, image_tokens = self._embed_prefix_with_image_tokens(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, noisy_actions, timestep
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _, action_intermediates = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            method="forward_with_intermediates",
        )
        depth = action_intermediates.shape[0]
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        hiddens = []
        for layer in self.align_layers:
            if layer == -1:
                hiddens.append(suffix_out[:, -self.action_horizon :])
            else:
                hidden = action_intermediates[self._layer_index(layer, depth)][:, -self.action_horizon :]
                hiddens.append(_prenorm(hidden))
        privileged_visual_tokens = jnp.concatenate(
            [image_tokens["base_0_rgb"], image_tokens["left_wrist_0_rgb"]], axis=1
        )
        return v_t, privileged_visual_tokens, hiddens

    def _embed_prefix_with_image_tokens(self, observation: _model.Observation):
        input_mask = []
        ar_mask = []
        tokens = []
        image_tokens_by_name = {}

        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
            image_tokens_by_name[name] = image_tokens
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
        )
