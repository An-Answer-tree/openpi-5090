import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as model_lib
from openpi.models.pi0_distill_acpd import AcpdPi0
from openpi.models.pi0_distill_acpd import ExactContributionHead
from scripts.distillation_acpd.action_readout_probe import _analyze
from scripts.distillation_acpd.action_readout_probe import _corrections
from scripts.distillation_acpd.action_readout_probe import _extract_features
from scripts.distillation_acpd.action_readout_probe import _init_heads
from scripts.distillation_acpd.action_readout_probe import _make_inputs
from scripts.distillation_acpd.action_readout_probe import _sample_errors
from scripts.distillation_acpd.action_readout_probe import _train_step
from scripts.distillation_acpd.train_distill import FrozenModelState


def test_zero_initialization_and_capacity_control():
    final = jax.random.normal(jax.random.key(0), (3, 2, 4))
    layer = jax.random.normal(jax.random.key(1), (3, 2, 4))
    contribution = jax.random.normal(jax.random.key(2), (3, 2, 2, 4))
    inputs = _make_inputs(final, layer, contribution)
    params = _init_heads(jax.random.key(3), 12, 8)
    assert inputs.shape == (3, 3, 2, 12)
    np.testing.assert_array_equal(inputs[0, ..., 4:], 0)
    np.testing.assert_allclose(inputs[1, ..., 4:8], inputs[1, ..., 8:])
    for index in (1, 2):
        np.testing.assert_array_equal(params["input_kernel"][0], params["input_kernel"][index])
    np.testing.assert_array_equal(_corrections(params, inputs), np.zeros((3, 3, 2, 7)))
    velocity = jnp.ones((3, 2, 7))
    errors = _sample_errors(params, inputs, velocity, jnp.zeros_like(velocity))
    np.testing.assert_allclose(errors, 1.0)


def test_train_step_updates_only_head_and_reduces_simple_target_error():
    inputs = jax.random.normal(jax.random.key(0), (3, 8, 2, 12))
    velocity = jnp.zeros((8, 2, 7))
    target = jnp.ones_like(velocity) * 0.2
    params = _init_heads(jax.random.key(1), 12, 8)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)
    initial = float(jnp.mean(_sample_errors(params, inputs, velocity, target)))
    step = jax.jit(lambda p, o: _train_step(p, o, inputs, velocity, target, optimizer=optimizer))
    for _ in range(10):
        params, opt_state, losses = step(params, opt_state)
    assert jnp.isfinite(losses).all()
    assert float(jnp.mean(_sample_errors(params, inputs, velocity, target))) < initial
    assert float(jnp.linalg.norm(params["output_kernel"])) > 0


def test_extraction_detaches_backbone_and_uses_only_seven_real_actions():
    class TinyModel(nnx.Module):
        def __init__(self):
            self.weight = nnx.Param(jnp.asarray(0.5))
            self.exact_contribution_head = ExactContributionHead(4, rngs=nnx.Rngs(0))

        def compute_train_outputs(self, rng, observation, noisy_actions, timestep, *, train, include_final_hidden):
            assert not train
            assert include_final_hidden
            hidden = noisy_actions[..., :4] * self.weight.value
            return noisy_actions * self.weight.value, hidden, [hidden, hidden + 1]

    model = TinyModel()
    graphdef, params = nnx.split(model)
    state = FrozenModelState(model_def=graphdef, params=params)
    actions = jnp.ones((2, 3, 32))
    batch = (None, None, actions)
    inputs, velocity, target, flow_time = _extract_features(jax.random.key(42), state, batch)
    assert inputs.shape == (3, 2, 3, 12)
    assert velocity.shape == target.shape == (2, 3, 7)
    assert flow_time.shape == (2,)

    def loss(frozen_state):
        values = _extract_features(jax.random.key(42), frozen_state, batch)
        return sum(jnp.sum(value) for value in values)

    gradients = jax.grad(loss)(state)
    np.testing.assert_allclose(optax.global_norm(gradients.params), 0.0)


def test_paired_analysis_uses_episode_means_not_frame_frequency():
    episodes = np.asarray([0, 0, 0, 1])
    baseline = np.asarray([1.0, 1.0, 1.0, 3.0])
    errors = np.stack([baseline, baseline * 0.9, baseline * 0.8, baseline * 0.7], axis=1)
    result = _analyze(errors, episodes, bootstrap_samples=100, seed=42)
    assert result["episode_mean_mse"]["frozen_h9"] == 2.0
    assert result["frame_mean_mse"]["frozen_h9"] == 1.5
    assert result["passes_offline_screen"]
    assert not result["benchmark_success_available"]
    errors[:, -1] = errors[:, -2]
    result = _analyze(errors, episodes, bootstrap_samples=100, seed=42)
    assert not result["passes_offline_screen"]


def test_final_hidden_option_preserves_original_velocity(monkeypatch):
    class TinyLlm:
        def __call__(self, tokens, **kwargs):
            prefix, suffix = tokens
            final = suffix + 0.5
            return (prefix, final), None, jnp.stack([suffix] * 10)

    class TinyAcpd(AcpdPi0):
        def __init__(self):
            self.action_horizon = 2
            self.align_layers = (10,)
            self.action_expert_depth = 10
            self.exact_contribution_fusion = True
            self.exact_contribution_injection = True
            self.exact_contribution_task_gradient = False
            self.exact_contribution_fusion_location = "final"
            self.exact_contribution_head = ExactContributionHead(4, rngs=nnx.Rngs(0))
            self.exact_contribution_head.gate.value = jnp.asarray(0.1)
            self.PaliGemma = nnx.Dict(llm=TinyLlm())
            self.action_out_proj = nnx.Linear(4, 32, rngs=nnx.Rngs(1))

        def _embed_prefix_with_image_tokens(self, observation):
            tokens = jnp.ones((2, 2, 4))
            images = {"base_0_rgb": tokens, "left_wrist_0_rgb": tokens}
            return tokens, jnp.ones((2, 2), dtype=bool), jnp.zeros(2, dtype=bool), images, {}

        def embed_suffix(self, observation, noisy_actions, timestep):
            tokens = jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 4)
            return tokens, jnp.ones((2, 2), dtype=bool), jnp.zeros(2, dtype=bool), None

    monkeypatch.setattr(model_lib, "preprocess_observation", lambda rng, obs, **kwargs: obs)
    model = TinyAcpd()
    observation = model_lib.Observation(images={}, image_masks={}, state=jnp.zeros((2, 32)))
    args = (jax.random.key(0), observation, jnp.zeros((2, 2, 32)), jnp.ones(2))
    original = model.compute_train_outputs(*args)
    extended = model.compute_train_outputs(*args, include_final_hidden=True)
    np.testing.assert_array_equal(original[0], extended[0])
    assert len(original[2]) == 1
    assert len(extended[2]) == 2
    final_hidden = jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 4) + 0.5
    np.testing.assert_array_equal(extended[2][-1], final_hidden)
