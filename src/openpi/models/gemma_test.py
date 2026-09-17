import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma


def test_module_optionally_returns_action_intermediates():
    config = gemma.get_config("dummy")
    module = gemma.Module(configs=[config, config], embed_dtype="bfloat16")
    embedded = [jnp.zeros((2, 3, config.width)), jnp.zeros((2, 2, config.width))]
    positions = jnp.broadcast_to(jnp.arange(5), (2, 5))
    mask = jnp.ones((2, 5, 5), dtype=bool)
    llm = nnx_bridge.ToNNX(module)
    llm.lazy_init(rngs=nnx.Rngs(0), method="init", use_adarms=[False, False])

    outputs, _ = llm(embedded, positions, mask)
    outputs_with_hidden, _, action_hiddens = llm(
        embedded,
        positions,
        mask,
        method="forward_with_intermediates",
    )

    assert action_hiddens.shape == (config.depth, 2, 2, config.width)
    np.testing.assert_allclose(
        np.asarray(outputs[1], dtype=np.float32),
        np.asarray(outputs_with_hidden[1], dtype=np.float32),
    )

    prefix = [embedded[0], None]
    prefix_positions = positions[:, :3]
    prefix_mask = mask[:, :3, :3]
    prefix_outputs, _ = llm(prefix, prefix_positions, prefix_mask)
    assert prefix_outputs[1] is None


def test_attention_source_contributions_reconstruct_action_attention():
    config = gemma.get_config("dummy")
    module = gemma.Module(configs=[config, config], embed_dtype="bfloat16")
    embedded = [jnp.ones((2, 3, config.width)), jnp.ones((2, 2, config.width))]
    positions = jnp.broadcast_to(jnp.arange(5), (2, 5))
    mask = jnp.ones((2, 5, 5), dtype=bool)
    source_masks = jnp.asarray(
        [
            [True, True, False, False, False],
            [False, False, True, True, True],
        ]
    )
    llm = nnx_bridge.ToNNX(module)
    llm.lazy_init(rngs=nnx.Rngs(0), method="init", use_adarms=[False, False])

    outputs, _ = llm(embedded, positions, mask)
    diagnostic_outputs, _, _, diagnostics = llm(
        embedded,
        positions,
        mask,
        source_masks,
        method="forward_with_attention_contributions",
    )
    contributions, shuffled_contributions, total_attention = diagnostics

    assert contributions.shape == (config.depth, 2, 2, 2, config.width)
    assert shuffled_contributions.shape == contributions.shape
    reconstruction = np.asarray(contributions.sum(axis=1))
    total_attention = np.asarray(total_attention)
    relative_error = np.linalg.norm(reconstruction - total_attention) / np.linalg.norm(total_attention)
    assert relative_error < 1e-3
    np.testing.assert_allclose(
        np.asarray(outputs[1], dtype=np.float32),
        np.asarray(diagnostic_outputs[1], dtype=np.float32),
    )


def test_exact_contribution_fusion_masks_tokens_and_detaches_predictor():
    hidden = jnp.arange(24, dtype=jnp.float32).reshape(1, 3, 8)
    normalized = gemma._parameter_free_layer_norm(hidden)  # noqa: SLF001
    token_mask = jnp.asarray([False, True, True])
    kernel = jnp.ones((8, 16), dtype=jnp.float32)
    bias = jnp.ones((16,), dtype=jnp.float32)

    zero_gate = gemma._fuse_exact_contribution(  # noqa: SLF001
        hidden, normalized, token_mask, kernel, bias, jnp.asarray(0.0)
    )
    np.testing.assert_allclose(zero_gate, hidden)

    def flow_loss(kernel_, gate_):
        fused = gemma._fuse_exact_contribution(  # noqa: SLF001
            hidden, normalized, token_mask, kernel_, bias, gate_
        )
        return jnp.mean(fused)

    kernel_grad, gate_grad = jax.grad(flow_loss, argnums=(0, 1))(kernel, jnp.asarray(0.5))
    np.testing.assert_allclose(kernel_grad, 0.0)
    assert float(jnp.abs(gate_grad)) > 0.0

    fused = gemma._fuse_exact_contribution(  # noqa: SLF001
        hidden, normalized, token_mask, kernel, bias, jnp.asarray(0.5)
    )
    np.testing.assert_allclose(fused[:, 0], hidden[:, 0])
    assert not np.allclose(fused[:, 1:], hidden[:, 1:])


def test_module_fuses_exact_contribution_inside_selected_layer():
    config = gemma.get_config("dummy")
    module = gemma.Module(configs=[config, config], embed_dtype="bfloat16")
    embedded = [jnp.zeros((2, 3, config.width)), jnp.ones((2, 2, config.width))]
    positions = jnp.broadcast_to(jnp.arange(5), (2, 5))
    mask = jnp.ones((2, 5, 5), dtype=bool)
    adarms_cond = [None, None]
    kernel = jnp.ones((config.width, 2 * config.width), dtype=jnp.float32)
    bias = jnp.ones((2 * config.width,), dtype=jnp.float32)
    llm = nnx_bridge.ToNNX(module)
    llm.lazy_init(rngs=nnx.Rngs(0), method="init", use_adarms=[False, False])

    baseline, _ = llm(embedded, positions, mask)
    zero_gate, _, _, attention_hiddens = llm(
        embedded,
        positions,
        mask,
        adarms_cond,
        kernel,
        bias,
        jnp.asarray(0.0),
        1,
        2,
        method="forward_with_exact_contribution_fusion",
    )
    fused, _, _, _ = llm(
        embedded,
        positions,
        mask,
        adarms_cond,
        kernel,
        bias,
        jnp.asarray(0.5),
        1,
        2,
        method="forward_with_exact_contribution_fusion",
    )

    assert attention_hiddens.shape == (config.depth, 2, 2, config.width)
    np.testing.assert_allclose(
        np.asarray(zero_gate[1], dtype=np.float32),
        np.asarray(baseline[1], dtype=np.float32),
    )
    assert not np.allclose(np.asarray(fused[1], dtype=np.float32), np.asarray(baseline[1], dtype=np.float32))
