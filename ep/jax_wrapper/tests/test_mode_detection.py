"""Unit-ish tests for execution-mode detection that do *not* require GPUs.

These tests focus on the pure Python logic (mode detection, KV store) so
they can run on any CI node even when jaxlib is not present. They are
guarded with ``pytest.importorskip`` where the dependency is needed.
"""

from __future__ import annotations

import threading

import pytest


def test_in_process_kv_store_roundtrip():
    from uccl_ep_jax.kv_store import InProcessKeyValueStore, KVClient

    store = InProcessKeyValueStore()
    client = KVClient(store)

    client.set("foo", {"hello": "world"})
    assert client.get("foo") == {"hello": "world"}


def test_kv_store_all_gather_threads():
    from uccl_ep_jax.kv_store import InProcessKeyValueStore, KVClient

    store = InProcessKeyValueStore()
    world_size = 4
    results = [None] * world_size

    def worker(rank: int):
        c = KVClient(store)
        results[rank] = c.all_gather("ns", rank, world_size, rank * 10)

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    expected = [r * 10 for r in range(world_size)]
    assert all(r == expected for r in results)


def test_execution_mode_detection():
    pytest.importorskip("jax")
    from uccl_ep_jax import (
        JaxExecutionMode,
        detect_execution_mode,
        is_multi_process_mode,
        is_single_process_mode,
    )

    mode = detect_execution_mode()
    assert isinstance(mode, JaxExecutionMode)
    assert is_single_process_mode() != is_multi_process_mode()


def test_config_table_contains_common_sizes():
    from uccl_ep_jax import get_combine_config, get_dispatch_config

    for num_ranks in (2, 4, 8):
        d = get_dispatch_config(num_ranks)
        c = get_combine_config(num_ranks)
        assert d.num_sms > 0
        assert c.num_sms > 0
