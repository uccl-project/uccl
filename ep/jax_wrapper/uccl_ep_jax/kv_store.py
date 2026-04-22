"""Key-value rendezvous store used for OOB exchange.

The JAX distributed client (``DistributedRuntimeClient``) exposes
``key_value_set_bytes`` / ``blocking_key_value_get_bytes`` which works
across processes. For the single-process-multi-thread mode we implement
a lightweight in-process equivalent so callers can use the same API for
both modes.
"""

from __future__ import annotations

import pickle
import threading
import time
from typing import Any, Optional


class InProcessKeyValueStore:
    """Thread-safe in-process KV store used as a rendezvous primitive.

    Values are opaque ``bytes`` buffers, mirroring the JAX distributed
    client API.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._store: dict[str, bytes] = {}

    def key_value_set_bytes(self, key: str, value: bytes) -> None:
        with self._cond:
            self._store[key] = bytes(value)
            self._cond.notify_all()

    def blocking_key_value_get_bytes(self, key: str, timeout_ms: int) -> bytes:
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        with self._cond:
            while key not in self._store:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Key '{key}' not available after {timeout_ms} ms"
                    )
                self._cond.wait(timeout=remaining)
            return self._store[key]

    def clear(self) -> None:
        with self._cond:
            self._store.clear()
            self._cond.notify_all()


# Singleton store shared by all threads in the single-process mode.
_SHARED_STORE: Optional[InProcessKeyValueStore] = None
_SHARED_STORE_LOCK = threading.Lock()


def get_shared_store() -> InProcessKeyValueStore:
    global _SHARED_STORE
    if _SHARED_STORE is None:
        with _SHARED_STORE_LOCK:
            if _SHARED_STORE is None:
                _SHARED_STORE = InProcessKeyValueStore()
    return _SHARED_STORE


def reset_shared_store() -> None:
    global _SHARED_STORE
    with _SHARED_STORE_LOCK:
        if _SHARED_STORE is not None:
            _SHARED_STORE.clear()
        _SHARED_STORE = None


class KVClient:
    """Unified wrapper around either the JAX distributed client or the
    in-process store."""

    def __init__(self, backing: Any) -> None:
        self._backing = backing

    def set(self, key: str, value: Any) -> None:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        self._backing.key_value_set_bytes(key, payload)

    def get(self, key: str, timeout_ms: int = 60_000) -> Any:
        payload = self._backing.blocking_key_value_get_bytes(key, timeout_ms)
        return pickle.loads(payload)

    def barrier(self, namespace: str, rank: int, world_size: int, timeout_ms: int = 60_000) -> None:
        """Simple counting barrier on top of the KV store."""
        self.set(f"{namespace}/rank/{rank}", b"1")
        for r in range(world_size):
            self.get(f"{namespace}/rank/{r}", timeout_ms=timeout_ms)

    def all_gather(
        self, namespace: str, rank: int, world_size: int, value: Any, timeout_ms: int = 60_000
    ) -> list:
        self.set(f"{namespace}/{rank}", value)
        return [
            self.get(f"{namespace}/{r}", timeout_ms=timeout_ms) for r in range(world_size)
        ]
