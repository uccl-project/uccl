"""Bootstrap / teardown of UCCL-EP state for JAX.

This module mirrors the PyTorch path in ``ep/deep_ep_wrapper/deep_ep``:

1. Allocate an RDMA scratch buffer (GPU or pinned host memory).
2. Start the UCCL CPU proxy threads.
3. Rendezvous with peers to exchange IPC handles / listen ports.
4. Construct the ``uccl.ep.Buffer`` C++ runtime and call ``sync``.

Both JAX execution modes are handled:

* **Multi-process** -- :func:`get_distributed_client` is used for the
  key/value rendezvous.
* **Single-process multi-thread** -- a shared in-process KV store is
  used (see :mod:`.kv_store`). Each Python thread owning a GPU calls
  :func:`initialize` with its own ``local_rank``.
"""

from __future__ import annotations

import glob
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

from .config import Config
from .kv_store import (
    InProcessKeyValueStore,
    KVClient,
    get_shared_store,
    reset_shared_store,
)
from .mode import (
    JaxExecutionMode,
    _clear_thread_rank,
    _set_thread_rank,
    detect_execution_mode,
    get_global_rank,
    get_global_world_size,
    get_local_rank,
    get_local_world_size,
    _try_get_distributed_client,
)

try:
    from uccl import ep as _uccl_ep
except ImportError as exc:  # pragma: no cover
    _uccl_ep = None
    _UCCL_IMPORT_ERROR = exc
else:
    _UCCL_IMPORT_ERROR = None


def _require_uccl_ep():
    if _uccl_ep is None:
        raise RuntimeError(
            "Failed to import `uccl.ep`. Install the uccl.ep wheel before using "
            f"uccl_ep_jax. Original import error: {_UCCL_IMPORT_ERROR}"
        )
    return _uccl_ep


def get_low_latency_rdma_size_hint(
    num_max_dispatch_tokens_per_rank: int,
    hidden: int,
    num_ranks: int,
    num_experts: int,
) -> int:
    """Return the RDMA scratch buffer size to pass to :func:`initialize`.

    Mirror of ``uccl.ep.get_low_latency_rdma_size_hint``.
    """
    return int(
        _require_uccl_ep().get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts
        )
    )


# ---------------------------------------------------------------------------
# Per-thread/process Buffer registry
# ---------------------------------------------------------------------------


class Buffer:
    """JAX-friendly wrapper around ``uccl.ep.Buffer``.

    Keeps a reference to the RDMA scratch buffer, the list of proxies and
    the C++ runtime, so everything lives as long as the user keeps this
    handle alive.
    """

    def __init__(
        self,
        *,
        runtime,
        scratch_ptr: int,
        scratch_nbytes: int,
        proxies: list,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        low_latency_mode: bool,
        rdma_buffer_is_host_allocated: bool,
        num_experts: int,
        kv_client: KVClient,
        cuda_device_index: int,
        _owned_buffer=None,
    ) -> None:
        self.runtime = runtime
        self.scratch_ptr = scratch_ptr
        self.scratch_nbytes = scratch_nbytes
        self.proxies = proxies
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.local_world_size = local_world_size
        self.low_latency_mode = low_latency_mode
        self.rdma_buffer_is_host_allocated = rdma_buffer_is_host_allocated
        self.num_experts = num_experts
        self.kv_client = kv_client
        # CUDA ordinal the runtime actually ended up on. This is what the
        # FFI handler's ``cudaGetDevice()`` will return on an XLA worker
        # thread, so we use it as the key in the per-device registry.
        self.cuda_device_index = int(cuda_device_index)
        # When the scratch buffer was allocated as a raw CUDA alloc via
        # `uccl.ep.get_rdma_buffer` we keep the capsule here so it is
        # freed at the right moment.
        self._owned_buffer = _owned_buffer
        self._destroyed = False
        self.num_sms = 20

    # ----- accessors ------------------------------------------------------

    def is_available(self) -> bool:
        return (not self._destroyed) and bool(self.runtime.is_available())

    @property
    def num_rdma_ranks(self) -> int:
        return int(self.runtime.get_num_rdma_ranks())

    @property
    def num_max_nvl_peers(self) -> int:
        return int(self.runtime.get_num_max_nvl_peers())

    @property
    def source_meta_bytes(self) -> int:
        return int(self.runtime.get_source_meta_bytes())

    # ----- lifecycle ------------------------------------------------------

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        ep = _require_uccl_ep()
        # Remove the FFI registration before tearing down the runtime so
        # any residual XLA call that might race the destructor sees an
        # absent entry and early-returns instead of dereferencing freed
        # memory. We unregister under the *CUDA ordinal* the runtime
        # was registered under (not the local_rank) so this works
        # whether the calling thread is the one that originally owned
        # this ``Buffer`` or another worker doing teardown.
        if hasattr(ep, "unregister_jax_ffi_buffer"):
            try:
                ep.unregister_jax_ffi_buffer(self.cuda_device_index)
            except Exception:
                pass
        try:
            self.runtime.destroy()
        except Exception:
            pass
        for p in self.proxies:
            try:
                p.stop()
            except Exception:
                pass
        try:
            ep.unregister_proxy(self.local_rank)
        except Exception:
            pass
        _unregister_thread_buffer()
        _clear_thread_rank()


# Per-thread registry so ops can look up "the" buffer for the current
# thread without the user having to thread it through every call site.
_thread_buffer_tls = threading.local()
_buffers_by_global_rank: Dict[int, Buffer] = {}
_registry_lock = threading.Lock()


def _register_thread_buffer(buf: Buffer) -> None:
    _thread_buffer_tls.buffer = buf
    with _registry_lock:
        _buffers_by_global_rank[buf.rank] = buf


def _unregister_thread_buffer() -> None:
    buf = getattr(_thread_buffer_tls, "buffer", None)
    if buf is not None:
        with _registry_lock:
            _buffers_by_global_rank.pop(buf.rank, None)
        _thread_buffer_tls.buffer = None


def get_buffer() -> Buffer:
    """Return the :class:`Buffer` bound to the current Python thread."""
    buf = getattr(_thread_buffer_tls, "buffer", None)
    if buf is None:
        raise RuntimeError(
            "No uccl_ep_jax Buffer is bound to the current thread. "
            "Call `uccl_ep_jax.initialize(...)` first."
        )
    return buf


# ---------------------------------------------------------------------------
# RDMA scratch buffer allocation
# ---------------------------------------------------------------------------


def _allocate_rdma_scratch(num_rdma_bytes: int, device_index: int):
    """Allocate the scratch buffer used by the UCCL runtime.

    Returns ``(scratch_ptr, scratch_nbytes, is_host_allocated, owner)``
    where ``owner`` is whatever needs to stay alive to keep the memory
    around (a DLPack capsule, a ``numpy`` array, or ``None`` for raw
    allocations freed by the C++ side).
    """
    ep = _require_uccl_ep()

    if hasattr(ep, "get_rdma_buffer"):
        # Raw cudaMalloc / cudaMallocHost allocation that survives DLPack
        # conversion. This is the preferred path.
        dlpack_capsule, is_host = ep.get_rdma_buffer(num_rdma_bytes, device_index)
        # Convert to a JAX-owned array so it is tracked by Python's GC.
        # We keep the capsule handle as the "owner" so the underlying
        # memory stays alive even if the array is reassigned.
        try:
            import jax

            arr = jax.dlpack.from_dlpack(dlpack_capsule)
            scratch_ptr = int(arr.unsafe_buffer_pointer()) if not is_host else None
        except Exception:
            arr = None
            scratch_ptr = None

        # Fall back: ask the C++ binding for the raw pointer directly via
        # the capsule's ``byte_offset``/``data`` fields through a helper.
        # We implement a small helper on the C++ side (see setup notes)
        # but for host-allocated buffers we can rely on a numpy view.
        if scratch_ptr is None:
            # The DLPack capsule encodes the pointer; we read it out via a
            # small ctypes dance.
            scratch_ptr = _raw_ptr_from_dlpack(dlpack_capsule)

        return scratch_ptr, num_rdma_bytes, bool(is_host), (dlpack_capsule, arr)

    # Fallback for old bindings that do not ship ``get_rdma_buffer``.
    if num_rdma_bytes <= 0:
        return 0, 0, False, None

    should_use_host = False
    if hasattr(ep, "can_register_rdma_gpu_buffer"):
        should_use_host = not bool(
            ep.can_register_rdma_gpu_buffer(device_index, num_rdma_bytes)
        )
    elif hasattr(ep, "rdma_buffer_should_use_host_alloc"):
        should_use_host = bool(
            ep.rdma_buffer_should_use_host_alloc(device_index, num_rdma_bytes)
        )

    # If we have neither of the helpers above we must defer to a raw
    # cudaMalloc via ``cudart``; that is best-effort and out of scope for
    # this wrapper.
    raise RuntimeError(
        "uccl.ep.get_rdma_buffer is not available in this build of uccl.ep; "
        "please rebuild uccl.ep to use it with JAX."
    )


def _raw_ptr_from_dlpack(capsule) -> int:
    """Extract the ``data`` pointer from a DLPack capsule.

    This is a narrow helper used when :mod:`jax` refuses to wrap an FP8
    / int8 device tensor but we still need the raw pointer.
    """
    import ctypes

    class _DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

    class _DLDataType(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_uint8),
            ("bits", ctypes.c_uint8),
            ("lanes", ctypes.c_uint16),
        ]

    class _DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device", _DLDevice),
            ("ndim", ctypes.c_int32),
            ("dtype", _DLDataType),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_uint64),
        ]

    class _DLManagedTensor(ctypes.Structure):
        _fields_ = [
            ("dl_tensor", _DLTensor),
            ("manager_ctx", ctypes.c_void_p),
            ("deleter", ctypes.c_void_p),
        ]

    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.restype = ctypes.c_void_p
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

    for name in (b"dltensor", b"used_dltensor"):
        try:
            ptr = PyCapsule_GetPointer(capsule, name)
            if ptr:
                mt = ctypes.cast(ptr, ctypes.POINTER(_DLManagedTensor)).contents
                return int(mt.dl_tensor.data) + int(mt.dl_tensor.byte_offset)
        except Exception:
            continue
    raise RuntimeError("Could not extract raw pointer from DLPack capsule")


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------


def _cleanup_shm_files() -> None:
    try:
        for shm_file in glob.glob("/dev/shm/uccl_barrier_*"):
            try:
                os.remove(shm_file)
            except OSError:
                pass
    except Exception:
        pass


def _get_device_index(local_rank: Optional[int]) -> int:
    import jax

    if local_rank is None:
        local_rank = get_local_rank()

    local_devices = jax.local_devices()
    if not local_devices:
        raise RuntimeError("jax.local_devices() returned no devices")

    dev = local_devices[local_rank]
    # Device.id is the global device index under JAX; the CUDA ordinal
    # maps to ``id % local_device_count()`` when every device is a GPU.
    local_count = max(len(local_devices), 1)
    return int(getattr(dev, "id", local_rank)) % local_count


def _build_kv_client(
    mode: JaxExecutionMode,
    shared_store: Optional[InProcessKeyValueStore] = None,
) -> KVClient:
    if mode is JaxExecutionMode.MULTI_PROCESS:
        client = _try_get_distributed_client()
        if client is None:
            raise RuntimeError(
                "JAX distributed client is not initialized. Call "
                "`jax.distributed.initialize(...)` before initialize()."
            )
        return KVClient(client)
    # Single-process mode
    return KVClient(shared_store or get_shared_store())


def initialize(
    *,
    num_rdma_bytes: int,
    low_latency_mode: bool = True,
    num_experts: int = 0,
    is_intranode: bool = False,
    local_rank: Optional[int] = None,
    global_rank: Optional[int] = None,
    global_world_size: Optional[int] = None,
    local_world_size: Optional[int] = None,
    num_qps_per_rank: int = 24,
    explicitly_destroy: bool = True,
    shared_store: Optional[InProcessKeyValueStore] = None,
    ns: str = "uccl_ep_jax/default",
    rendezvous_timeout_ms: int = 120_000,
) -> Buffer:
    """Initialize the UCCL-EP runtime for the calling thread/process.

    Arguments:
        num_rdma_bytes: size of the RDMA scratch buffer in bytes. Use
            :func:`uccl_ep_jax.get_low_latency_rdma_size_hint` to pick a
            reasonable value in low-latency mode.
        low_latency_mode: whether the low-latency dispatch/combine
            kernels will be used. Matches ``uccl.ep.Buffer``.
        num_experts: total number of experts (required for the proxy).
        is_intranode: whether the group is purely intranode (skips the
            RDMA bring-up).
        local_rank / global_rank / global_world_size / local_world_size:
            override the values inferred from JAX. Required in
            single-process-multi-thread mode (each thread must pass its
            own ``local_rank``).
        num_qps_per_rank: forwarded to the proxies (must equal the number
            of local experts in low-latency mode).
        explicitly_destroy: if True (default for JAX), the user is
            responsible for calling :meth:`Buffer.destroy`.
        shared_store: optional :class:`InProcessKeyValueStore` to use in
            single-process mode. Defaults to the module-level singleton.
        ns: rendezvous namespace; override this to run several EP groups
            independently in the same process.
    """
    ep = _require_uccl_ep()
    mode = detect_execution_mode()

    # --- rank bookkeeping ----------------------------------------------
    if mode is JaxExecutionMode.MULTI_PROCESS:
        import jax

        if global_rank is None:
            global_rank = jax.process_index()
        if global_world_size is None:
            global_world_size = jax.process_count() * max(jax.local_device_count(), 1)
        if local_rank is None:
            local_rank = get_local_rank()
        if local_world_size is None:
            local_world_size = max(jax.local_device_count(), 1)
    else:
        import jax

        if global_world_size is None:
            global_world_size = max(jax.local_device_count(), 1)
        if local_world_size is None:
            local_world_size = global_world_size
        if local_rank is None:
            # Single-process mode covers both:
            #   * single-GPU (one thread, one device): local_rank defaults to 0.
            #   * single-process multi-thread multi-GPU (one thread per GPU):
            #     each worker thread MUST pass its own local_rank explicitly.
            if max(jax.local_device_count(), 1) == 1:
                local_rank = 0
            else:
                raise RuntimeError(
                    "In single-process multi-thread JAX mode each thread must "
                    "pass its own `local_rank` (0 .. local_device_count-1). "
                    "Single-GPU callers can omit `local_rank`."
                )
        if global_rank is None:
            global_rank = local_rank

    _set_thread_rank(local_rank, global_rank, global_world_size, local_world_size)

    device_index = _get_device_index(local_rank)
    ep.set_device(device_index)
    # Pin this Python thread to the CUDA context that the runtime is
    # about to use, and cache the ordinal that ``cudaGetDevice()`` will
    # report inside the FFI handler. In single-process multi-thread
    # mode, each worker thread has its own CUDA context set by
    # ``ep.set_device``; in multi-process mode this is the one and only
    # ordinal for the process.
    try:
        cuda_device_index = int(ep.get_device())
    except Exception:
        cuda_device_index = int(device_index)

    _cleanup_shm_files()

    # --- scratch buffer -------------------------------------------------
    scratch_ptr, scratch_nbytes, is_host_allocated, owner = _allocate_rdma_scratch(
        num_rdma_bytes, device_index
    )

    # --- proxies --------------------------------------------------------
    nproc_per_node = local_world_size
    num_nodes = global_world_size // max(nproc_per_node, 1)
    node_idx = global_rank // max(nproc_per_node, 1)

    proxies = []
    for i in range(ep.get_num_proxy_threads()):
        proxy = ep.Proxy(
            thread_idx=i,
            gpu_buffer_addr=scratch_ptr,
            total_size=scratch_nbytes,
            rank=global_rank,
            node_idx=node_idx,
            local_rank=local_rank,
            num_experts=num_experts,
            num_ranks=global_world_size,
            num_nodes=num_nodes,
            use_normal_mode=not low_latency_mode,
            is_intranode=is_intranode,
            gpu_buffer_is_host_allocated=is_host_allocated,
        )
        proxies.append(proxy)

    # --- rendezvous: exchange OOB metadata ------------------------------
    kv = _build_kv_client(mode, shared_store=shared_store)

    my_ip = ep.get_oob_ip()
    meta = {
        "rank": global_rank,
        "ptr": int(scratch_ptr or 0),
        "nbytes": int(scratch_nbytes),
        "ip": my_ip,
        "listen_ports": [p.get_listen_port() for p in proxies],
    }
    peers_meta = kv.all_gather(
        f"{ns}/meta", global_rank, global_world_size, meta, timeout_ms=rendezvous_timeout_ms
    )
    peers_meta_list = sorted(peers_meta, key=lambda m: m["rank"])

    if not is_intranode:
        for p in proxies:
            p.set_peers_meta(peers_meta_list)

    ep.register_proxies(local_rank, proxies)

    if not is_intranode and proxies:
        atomic_ptr = proxies[0].get_atomic_buffer_ptr()
        if atomic_ptr:
            for p in proxies:
                p.set_atomic_buffer_ptr(atomic_ptr)

    # Barrier before starting the proxies.
    kv.barrier(
        f"{ns}/proxy_pre_start", global_rank, global_world_size,
        timeout_ms=rendezvous_timeout_ms,
    )

    if not is_intranode:
        for p in proxies:
            p.start_dual()

    # Give proxies a moment to come up (mirrors deep_ep_wrapper).
    time.sleep(3)

    # --- C++ Buffer runtime --------------------------------------------
    runtime = ep.Buffer(
        global_rank,
        global_world_size,
        0,  # num_nvl_bytes - this wrapper assumes low-latency path by default
        scratch_nbytes,
        low_latency_mode,
        explicitly_destroy,
        local_world_size,
    )
    if scratch_nbytes:
        runtime.set_rdma_buffer(scratch_ptr, is_host_allocated)

    # --- IPC handle exchange --------------------------------------------
    local_device_id = runtime.get_local_device_id()
    local_ipc_handle = runtime.get_local_ipc_handle()
    local_rdma_ipc_handle = (
        runtime.get_local_rdma_ipc_handle()
        if scratch_nbytes > 0 and not is_host_allocated
        else None
    )

    device_ids = kv.all_gather(
        f"{ns}/device_ids", global_rank, global_world_size, local_device_id,
        timeout_ms=rendezvous_timeout_ms,
    )
    ipc_handles = kv.all_gather(
        f"{ns}/ipc_handles", global_rank, global_world_size, local_ipc_handle,
        timeout_ms=rendezvous_timeout_ms,
    )
    rdma_ipc_handles = kv.all_gather(
        f"{ns}/rdma_ipc_handles", global_rank, global_world_size, local_rdma_ipc_handle,
        timeout_ms=rendezvous_timeout_ms,
    )

    runtime.sync(device_ids, ipc_handles, None, rdma_ipc_handles)
    assert runtime.is_available(), "uccl.ep runtime failed to come up"

    if proxies:
        ep.connect_atomic_buffer(proxies[0], runtime)
        atomic_ptr = proxies[0].get_atomic_buffer_ptr()
        for p in proxies:
            p.set_atomic_buffer_ptr(atomic_ptr)

    # Register the C++ runtime for the JAX FFI handlers so XLA can
    # resolve "which Buffer?" for the active CUDA device when it
    # invokes the custom-call targets. The key is the *CUDA ordinal*
    # (what ``cudaGetDevice()`` returns inside the handler); this makes
    # the registry work transparently across:
    #   * single-process multi-thread mode: every worker thread has
    #     its own CUDA context set by ``ep.set_device`` and owns a
    #     distinct ``Buffer`` in ``g_jax_ffi_buffers``;
    #   * multi-process mode: each process has one CUDA context and
    #     one registered ``Buffer``.
    # Older uccl.ep builds may not have this hook; in that case we
    # silently fall back to the eager code path.
    if hasattr(ep, "register_jax_ffi_buffer"):
        try:
            ep.register_jax_ffi_buffer(cuda_device_index, runtime)
        except Exception:
            pass

    buf = Buffer(
        runtime=runtime,
        scratch_ptr=scratch_ptr,
        scratch_nbytes=scratch_nbytes,
        proxies=proxies,
        rank=global_rank,
        world_size=global_world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        low_latency_mode=low_latency_mode,
        rdma_buffer_is_host_allocated=is_host_allocated,
        num_experts=num_experts,
        kv_client=kv,
        cuda_device_index=cuda_device_index,
        _owned_buffer=owner,
    )
    _register_thread_buffer(buf)
    return buf


def shutdown(buf: Optional[Buffer] = None) -> None:
    """Destroy the per-thread ``Buffer`` and free proxies."""
    if buf is None:
        buf = getattr(_thread_buffer_tls, "buffer", None)
    if buf is None:
        return
    buf.destroy()

    # When the last thread in single-process mode is gone, reset the
    # shared store so a subsequent ``initialize`` starts fresh.
    if detect_execution_mode() is JaxExecutionMode.SINGLE_PROCESS:
        with _registry_lock:
            empty = not _buffers_by_global_rank
        if empty:
            reset_shared_store()
