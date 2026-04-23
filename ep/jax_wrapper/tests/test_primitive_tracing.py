"""Tracing-only tests for the XLA custom-call primitives.

These tests require only ``jax`` (no GPU, no ``uccl.ep`` build) — we
stub the ``uccl.ep`` module and skip the FFI-target registration step,
then verify that the primitive layer lowers to a ``stablehlo.custom_call``
with the expected target name and output shapes.
"""

from __future__ import annotations

import sys
import types

import pytest


def _stub_uccl_ep():
    """Install a minimal ``uccl.ep`` stub so ``import uccl_ep_jax`` works."""
    if "uccl.ep" in sys.modules:
        return
    pkg = types.ModuleType("uccl")
    ep = types.ModuleType("uccl.ep")
    pkg.ep = ep
    sys.modules["uccl"] = pkg
    sys.modules["uccl.ep"] = ep


@pytest.fixture(autouse=True)
def _skip_ffi_registration():
    """Prevent the lax layer from trying to register real FFI targets."""
    _stub_uccl_ep()
    pytest.importorskip("jax")
    import uccl_ep_jax.primitive.registry as registry
    registry._REGISTERED = True
    yield


def test_low_latency_dispatch_lowers_to_custom_call():
    import jax
    import jax.numpy as jnp
    import uccl_ep_jax as ux

    @jax.jit
    def fwd(x, topk_idx):
        return ux.low_latency_dispatch(
            x, topk_idx, num_max_dispatch_tokens_per_rank=64,
            num_experts=16, num_ranks=2, use_fp8=False,
        )

    x = jnp.zeros((64, 2048), jnp.bfloat16)
    topk = jnp.zeros((64, 4), jnp.int32)
    hlo = str(jax.jit(fwd).lower(x, topk).compiler_ir(dialect="stablehlo"))
    assert "stablehlo.custom_call" in hlo
    assert "uccl_ll_dispatch" in hlo


def test_moe_dispatch_lowers_to_custom_call():
    import jax
    import jax.numpy as jnp
    import uccl_ep_jax as ux

    @jax.jit
    def fwd(x, topk_idx, topk_weights):
        return ux.moe_dispatch(
            x, topk_idx, topk_weights, num_experts=32, num_ranks=4,
            config=ux.get_dispatch_config(4),
        )[0]

    x = jnp.zeros((128, 4096), jnp.bfloat16)
    topk = jnp.zeros((128, 8), jnp.int32)
    topkw = jnp.zeros((128, 8), jnp.float32)
    hlo = str(jax.jit(fwd).lower(x, topk, topkw).compiler_ir(dialect="stablehlo"))
    assert "stablehlo.custom_call" in hlo
    assert "uccl_moe_dispatch" in hlo


def test_moe_dispatch_output_shapes_static_upper_bound():
    import jax
    import jax.numpy as jnp
    import uccl_ep_jax as ux

    @jax.jit
    def fwd(x, topk_idx, topk_weights):
        recv_x, *_ = ux.moe_dispatch(
            x, topk_idx, topk_weights, num_experts=32, num_ranks=4,
        )
        return recv_x

    x = jnp.zeros((128, 4096), jnp.bfloat16)
    topk = jnp.zeros((128, 8), jnp.int32)
    topkw = jnp.zeros((128, 8), jnp.float32)
    shape_dtype = jax.eval_shape(fwd, x, topk, topkw)
    # Upper-bound convention: num_worst_tokens = num_tokens * num_ranks.
    assert shape_dtype.shape == (512, 4096)
    assert shape_dtype.dtype == jnp.bfloat16


def test_low_latency_combine_lowers_to_custom_call():
    import jax
    import jax.numpy as jnp
    import uccl_ep_jax as ux

    num_local_experts = 16 // 2
    num_recv = 2 * 64

    @jax.jit
    def fwd(x, topk_idx, topk_weights, src_info, layout_range):
        handle = (src_info, layout_range, 64, 2048, 16)
        return ux.low_latency_combine(x, topk_idx, topk_weights, handle)

    x = jnp.zeros((num_local_experts, num_recv, 2048), jnp.bfloat16)
    topk = jnp.zeros((64, 4), jnp.int32)
    topkw = jnp.zeros((64, 4), jnp.float32)
    src = jnp.zeros((num_local_experts, num_recv), jnp.int32)
    lay = jnp.zeros((num_local_experts, 2), jnp.int64)
    hlo = str(jax.jit(fwd).lower(x, topk, topkw, src, lay)
              .compiler_ir(dialect="stablehlo"))
    assert "stablehlo.custom_call" in hlo
    assert "uccl_ll_combine" in hlo
