"""Multi-thread tests for the FFI primitive layer.

These tests focus on the *Python-side* properties that single-process
multi-thread mode relies on:

* Each thread's primitive lowering uses its own thread-local
  ``Buffer`` to auto-detect ``num_rdma_ranks`` /
  ``num_max_nvl_peers`` / ``source_meta_bytes``.
* The FFI target registry is idempotent across threads.

The actual custom-call handlers need a GPU + ``uccl.ep`` build to
execute, so these tests only exercise the Python plumbing (tracing +
shape propagation) in both execution modes. They run on a CPU-only box
with a stubbed ``uccl.ep``.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest


def _stub_uccl_ep():
    if "uccl.ep" in sys.modules:
        return
    pkg = types.ModuleType("uccl")
    ep = types.ModuleType("uccl.ep")
    pkg.ep = ep
    sys.modules["uccl"] = pkg
    sys.modules["uccl.ep"] = ep


@pytest.fixture(autouse=True)
def _skip_ffi_registration():
    _stub_uccl_ep()
    pytest.importorskip("jax")
    import uccl_ep_jax.primitive.registry as registry
    registry._REGISTERED = True
    yield


class _FakeRuntime:
    """Minimal stand-in for ``uccl.ep.Buffer`` used by the auto-detect path."""

    def __init__(self, num_rdma_ranks: int, num_max_nvl_peers: int,
                 source_meta_bytes: int):
        self._num_rdma_ranks = int(num_rdma_ranks)
        self._num_max_nvl_peers = int(num_max_nvl_peers)
        self._source_meta_bytes = int(source_meta_bytes)

    def get_num_rdma_ranks(self) -> int:
        return self._num_rdma_ranks

    def get_num_max_nvl_peers(self) -> int:
        return self._num_max_nvl_peers

    def get_source_meta_bytes(self) -> int:
        return self._source_meta_bytes

    def is_available(self) -> bool:
        return True


def _install_fake_buffer(num_rdma_ranks: int, *, cuda_device_index: int = 0):
    """Install a ``Buffer`` in the thread-local registry without the
    full ``initialize()`` path (no proxies, no rendezvous)."""
    import uccl_ep_jax
    from uccl_ep_jax.bootstrap import Buffer, _register_thread_buffer

    buf = Buffer(
        runtime=_FakeRuntime(num_rdma_ranks, 8, 8),
        scratch_ptr=0, scratch_nbytes=0, proxies=[],
        rank=cuda_device_index, world_size=max(num_rdma_ranks, 1) * 8,
        local_rank=cuda_device_index, local_world_size=8,
        low_latency_mode=False, rdma_buffer_is_host_allocated=False,
        num_experts=32, kv_client=None,
        cuda_device_index=cuda_device_index,
    )
    _register_thread_buffer(buf)
    return buf


def _clear_fake_buffer():
    from uccl_ep_jax.bootstrap import _unregister_thread_buffer
    _unregister_thread_buffer()


def test_primitive_auto_detects_per_thread_topology():
    """Different threads, each with a different ``Buffer``, must route
    to the correct intranode/internode primitive independently."""
    import jax
    import jax.numpy as jnp
    import uccl_ep_jax as ux

    x = jnp.zeros((128, 4096), jnp.bfloat16)
    topk = jnp.zeros((128, 8), jnp.int32)
    topkw = jnp.zeros((128, 8), jnp.float32)

    results: dict = {}
    barrier = threading.Barrier(2)

    def worker(name: str, num_rdma_ranks: int, num_ranks: int,
               expected_target: str, unexpected_target: str):
        _install_fake_buffer(num_rdma_ranks, cuda_device_index=hash(name) & 7)
        try:
            # Wait for every worker to install its own Buffer before any
            # thread starts tracing, so we would catch an accidental
            # global dependency.
            barrier.wait()

            def fwd(x, topk_idx, topk_weights):
                return ux.moe_dispatch(
                    x, topk_idx, topk_weights,
                    num_experts=32, num_ranks=num_ranks,
                )[0]

            hlo = str(
                jax.jit(fwd).lower(x, topk, topkw)
                .compiler_ir(dialect="stablehlo")
            )
            results[name] = {
                "expected_present": expected_target in hlo,
                "unexpected_absent": unexpected_target not in hlo,
            }
        finally:
            _clear_fake_buffer()

    t1 = threading.Thread(
        target=worker,
        args=("intranode", 1, 4, "uccl_moe_dispatch", "uccl_moe_internode_dispatch"),
    )
    t2 = threading.Thread(
        target=worker,
        args=("internode", 2, 16, "uccl_moe_internode_dispatch", "uccl_moe_cached_dispatch"),
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["intranode"]["expected_present"]
    assert results["intranode"]["unexpected_absent"]
    assert results["internode"]["expected_present"]
    assert results["internode"]["unexpected_absent"]


def test_primitive_topology_override_works_without_buffer():
    """No ``Buffer`` bound: passing the topology explicitly must still
    let tracing/lowering succeed."""
    import jax
    import jax.numpy as jnp
    import uccl_ep_jax as ux

    x = jnp.zeros((128, 4096), jnp.bfloat16)
    topk = jnp.zeros((128, 8), jnp.int32)
    topkw = jnp.zeros((128, 8), jnp.float32)

    def fwd_intranode(x, topk_idx, topk_weights):
        return ux.moe_dispatch(
            x, topk_idx, topk_weights, num_experts=32, num_ranks=4,
            num_rdma_ranks=1, num_max_nvl_peers=8, source_meta_bytes=8,
        )[0]

    def fwd_internode(x, topk_idx, topk_weights):
        return ux.moe_dispatch(
            x, topk_idx, topk_weights, num_experts=32, num_ranks=16,
            num_rdma_ranks=2, num_max_nvl_peers=8, source_meta_bytes=8,
        )[0]

    hlo_a = str(jax.jit(fwd_intranode).lower(x, topk, topkw)
                .compiler_ir(dialect="stablehlo"))
    hlo_b = str(jax.jit(fwd_internode).lower(x, topk, topkw)
                .compiler_ir(dialect="stablehlo"))
    assert "uccl_moe_dispatch" in hlo_a
    assert "uccl_moe_internode_dispatch" not in hlo_a
    assert "uccl_moe_internode_dispatch" in hlo_b


def test_primitive_single_process_single_gpu_lowers_to_custom_call():
    """Primitive tracing must work in the simplest single-process,
    single-GPU setup (no thread spawning, no explicit topology)."""
    import jax
    import jax.numpy as jnp
    import uccl_ep_jax as ux

    # Low-latency is the only meaningful path for a single GPU; moe_*
    # needs num_ranks >= 2. We trace the low-latency dispatch/combine
    # and check the custom_call targets are emitted.
    num_ranks = 1
    num_experts = 8
    num_max = 64
    hidden = 2048

    @jax.jit
    def fwd(x, topk_idx, topk_weights):
        recv_x, _count, handle = ux.moe_low_latency_dispatch(
            x, topk_idx,
            num_max_dispatch_tokens_per_rank=num_max,
            num_experts=num_experts, num_ranks=num_ranks, use_fp8=False,
        )
        return ux.moe_low_latency_combine(recv_x, topk_idx, topk_weights, handle)

    x = jnp.zeros((num_max, hidden), jnp.bfloat16)
    topk = jnp.zeros((num_max, 4), jnp.int32)
    topkw = jnp.zeros((num_max, 4), jnp.float32)
    hlo = str(jax.jit(fwd).lower(x, topk, topkw)
              .compiler_ir(dialect="stablehlo"))
    assert "uccl_moe_low_latency_dispatch" in hlo
    assert "uccl_moe_low_latency_combine" in hlo


def test_primitive_single_process_multi_thread_end_to_end_tracing():
    """Two worker threads, each driving its own 'GPU', each tracing an
    independent jit graph — verifies no thread-local state leaks
    between them and each lowering uses its own topology."""
    import jax
    import jax.numpy as jnp
    import uccl_ep_jax as ux

    barrier = threading.Barrier(2)
    hlos: dict = {}

    def worker(name: str, num_ranks: int, num_rdma_ranks: int):
        # Each worker owns its own Buffer; CUDA ordinal uses the thread
        # identity so the two buffers land on different keys in the
        # per-device registry.
        _install_fake_buffer(
            num_rdma_ranks, cuda_device_index=hash(name) & 7
        )
        try:
            barrier.wait()

            @jax.jit
            def fwd(x, topk_idx, topk_weights):
                recv_x, _ti, recv_tw, handle = ux.moe_dispatch(
                    x, topk_idx, topk_weights,
                    num_experts=32, num_ranks=num_ranks,
                )
                return ux.moe_combine(
                    recv_x, handle, topk_weights=recv_tw,
                    num_ranks=num_ranks,
                )

            x = jnp.zeros((128, 4096), jnp.bfloat16)
            topk = jnp.zeros((128, 8), jnp.int32)
            topkw = jnp.zeros((128, 8), jnp.float32)
            hlos[name] = str(
                jax.jit(fwd).lower(x, topk, topkw)
                .compiler_ir(dialect="stablehlo")
            )
        finally:
            _clear_fake_buffer()

    threads = [
        threading.Thread(
            target=worker, args=("A_intranode", 4, 1), name="worker-A"
        ),
        threading.Thread(
            target=worker, args=("B_internode", 16, 2), name="worker-B"
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Thread A: intranode custom calls only, no internode.
    assert "uccl_moe_dispatch" in hlos["A_intranode"]
    assert "uccl_moe_combine" in hlos["A_intranode"]
    assert "uccl_moe_internode_dispatch" not in hlos["A_intranode"]
    # Thread B: internode custom calls only, no bare intranode.
    assert "uccl_moe_internode_dispatch" in hlos["B_internode"]
    assert "uccl_moe_internode_combine" in hlos["B_internode"]
    # The intranode target must not leak into the internode thread.
    # We check for the exact target name (intranode moe_dispatch is a
    # proper prefix of moe_internode_dispatch, so we search for the
    # quoted attribute form XLA emits).
    assert 'call_target_name = "uccl_moe_dispatch"' not in hlos["B_internode"]


def test_register_ffi_targets_idempotent_across_threads():
    """``register_ffi_targets`` must be safe to call concurrently."""
    import uccl_ep_jax
    # Reset the flag so the concurrent calls actually contend.
    import uccl_ep_jax.primitive.registry as registry
    registry._REGISTERED = False

    errors: list = []

    def worker():
        try:
            uccl_ep_jax.register_ffi_targets()
        except RuntimeError as exc:
            # Our stubbed ``uccl.ep`` will correctly fail (no
            # ``get_jax_ffi_targets`` attr). We only want to make sure
            # the call doesn't race into an internal invariant error.
            if "does not expose `get_jax_ffi_targets`" not in str(exc):
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    # Restore the short-circuit flag so the other tests keep skipping
    # the registration path.
    registry._REGISTERED = True
