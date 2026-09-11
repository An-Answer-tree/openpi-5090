import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
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
