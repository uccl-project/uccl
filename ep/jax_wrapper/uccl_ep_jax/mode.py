"""JAX execution mode detection and global rank bookkeeping.

UCCL-EP was originally written for ``torchrun``-style multi-process
execution. JAX however supports two different modes:

* **Multi-process / multi-controller** -- every GPU is owned by a
  dedicated Python process (``jax.process_count() > 1``). The JAX
  distributed client (``jax.distributed.initialize``) performs the OOB
  rendezvous, identically to ``torchrun``.

* **Single-process / single-controller multi-thread** -- one Python
  process owns every local GPU (``jax.process_count() == 1`` and
  ``jax.local_device_count() > 1``). Each thread typically drives one
  GPU.

This module exposes helpers to detect the active mode, mirroring the
conventions used in ``transformer_engine/jax`` and
``mori.jax``.
"""

from __future__ import annotations

import enum
import os
import threading
from typing import Optional


class JaxExecutionMode(enum.Enum):
    """Detected JAX execution mode.

    ``SINGLE_PROCESS`` -- one Python process drives one or more GPUs via
    one thread per device.

    ``MULTI_PROCESS`` -- one Python process drives exactly one GPU and
    the processes rendezvous through ``jax.distributed``.
    """

    SINGLE_PROCESS = "single_process"
    MULTI_PROCESS = "multi_process"


# Thread-local storage for rank assignment inside a single process. Each
# thread that owns one GPU installs its local rank / global rank here.
_thread_local = threading.local()


def detect_execution_mode() -> JaxExecutionMode:
    """Return the active :class:`JaxExecutionMode`.

    The decision mirrors NVIDIA/TransformerEngine: if JAX reports more
    than one process we are in multi-controller mode; otherwise we are in
    the single-process-multi-thread (or purely local) mode.
    """
    import jax

    if jax.process_count() > 1:
        return JaxExecutionMode.MULTI_PROCESS
    return JaxExecutionMode.SINGLE_PROCESS


def is_multi_process_mode() -> bool:
    return detect_execution_mode() is JaxExecutionMode.MULTI_PROCESS


def is_single_process_mode() -> bool:
    return detect_execution_mode() is JaxExecutionMode.SINGLE_PROCESS


def _set_thread_rank(
    local_rank: int, global_rank: int, global_world_size: int, local_world_size: int
) -> None:
    _thread_local.local_rank = int(local_rank)
    _thread_local.global_rank = int(global_rank)
    _thread_local.global_world_size = int(global_world_size)
    _thread_local.local_world_size = int(local_world_size)


def _clear_thread_rank() -> None:
    for attr in (
        "local_rank",
        "global_rank",
        "global_world_size",
        "local_world_size",
    ):
        if hasattr(_thread_local, attr):
            delattr(_thread_local, attr)


def get_local_rank() -> int:
    """Return the local rank of the calling thread / process.

    In multi-process mode this is ``jax.process_index()`` truncated to
    the local node (``LOCAL_RANK`` env var when set, otherwise
    ``jax.process_index() % jax.local_device_count()``). In
    single-process mode the value is installed by :func:`initialize` per
    thread.
    """
    if hasattr(_thread_local, "local_rank"):
        return _thread_local.local_rank

    import jax

    if is_multi_process_mode():
        if "LOCAL_RANK" in os.environ:
            return int(os.environ["LOCAL_RANK"])
        return jax.process_index() % max(jax.local_device_count(), 1)

    # No per-thread assignment has happened yet -- fall back to the
    # single device case.
    return 0


def get_global_rank() -> int:
    """Return the global rank (0 .. world_size-1) of the caller."""
    if hasattr(_thread_local, "global_rank"):
        return _thread_local.global_rank

    import jax

    if is_multi_process_mode():
        return jax.process_index()
    return 0


def get_global_world_size() -> int:
    """Return the global EP world size."""
    if hasattr(_thread_local, "global_world_size"):
        return _thread_local.global_world_size

    import jax

    if is_multi_process_mode():
        return jax.process_count() * max(jax.local_device_count(), 1)
    return max(jax.local_device_count(), 1)


def get_local_world_size() -> int:
    """Return the number of ranks (threads or processes) on the local node."""
    if hasattr(_thread_local, "local_world_size"):
        return _thread_local.local_world_size

    import jax

    return max(jax.local_device_count(), 1)


# ---------------------------------------------------------------------------
# JAX distributed client helpers (multi-process mode)
# ---------------------------------------------------------------------------


def get_distributed_client():
    """Return the JAX ``DistributedRuntimeClient`` if one is active.

    Matches the helper in :mod:`mori.jax.ops`. Raises ``RuntimeError`` when
    the client has not been initialized (e.g., user forgot to call
    ``jax.distributed.initialize``).
    """
    from jax._src.distributed import global_state  # type: ignore

    client = getattr(global_state, "client", None)
    if client is None:
        raise RuntimeError(
            "jax.distributed is not initialized. Call "
            "`jax.distributed.initialize(...)` before initializing uccl_ep_jax."
        )
    return client


def _try_get_distributed_client() -> Optional[object]:
    try:
        return get_distributed_client()
    except Exception:
        return None
