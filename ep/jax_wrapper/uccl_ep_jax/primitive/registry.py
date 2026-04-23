"""Register the UCCL-EP XLA custom-call targets with JAX.

:func:`register_ffi_targets` is idempotent and safe to call from any
thread; it fetches the ``PyCapsule`` handles exposed by
``uccl.ep.get_jax_ffi_targets()`` and hands them to
``jax.ffi.register_ffi_target`` under the platform matching the active
JAX backend (``cuda`` or ``rocm``).
"""

from __future__ import annotations

import threading

_REGISTERED = False
_REGISTER_LOCK = threading.Lock()

TARGET_NAMES = (
    "uccl_moe_low_latency_dispatch",
    "uccl_moe_low_latency_combine",
    "uccl_moe_dispatch",
    "uccl_moe_combine",
    "uccl_moe_internode_dispatch",
    "uccl_moe_internode_combine",
    "uccl_moe_cached_dispatch",
    "uccl_moe_internode_cached_dispatch",
)


def _active_platform() -> str:
    """Return the platform string to pass to ``jax.ffi.register_ffi_target``.

    jaxlib's :func:`jaxlib.xla_client.register_custom_call_target` looks up
    the per-platform registration callback via an internal dict keyed by
    the *XLA* platform name (``"CUDA"`` / ``"ROCM"`` / ``"Host"``), and it
    aliases ``"gpu"`` to ``"CUDA"`` via ``xla_platform_names``. On AMD/ROCm
    builds, ``jax.Device.platform`` still returns ``"gpu"`` but the XLA
    backend is actually ``"ROCM"`` - so blindly passing ``"gpu"`` or
    ``"cuda"`` silently registers the handler under the wrong key and
    results in ``NOT_FOUND: No FFI handler registered ... platform ROCM``
    at execution time.

    We therefore:

    1. Prefer the XLA backend names that are already present in jaxlib's
       ``_custom_callback_handler`` dict (the source of truth), returning
       ``"ROCM"`` or ``"CUDA"`` directly when available.
    2. Fall back to ``jax.default_backend()`` / device detection to pick
       between ROCm and CUDA otherwise.
    """

    import jax

    try:
        from jaxlib.xla_client import _custom_callback_handler
        handler_keys = set(_custom_callback_handler.keys())
    except Exception:
        handler_keys = set()

    rocm_available = False
    cuda_available = False
    for dev in jax.local_devices():
        plat = getattr(dev, "platform", "").lower()
        dev_kind = getattr(dev, "device_kind", "").lower()
        client_plat = getattr(dev, "client", None)
        if client_plat is not None:
            client_plat = getattr(client_plat, "platform", "").lower()
        rocm_markers = ("rocm", "amd", "hip")
        if (
            plat == "rocm"
            or any(m in dev_kind for m in rocm_markers)
            or (client_plat and any(m in client_plat for m in rocm_markers))
            or str(dev).startswith("rocm")
        ):
            rocm_available = True
        elif plat == "cuda" or "nvidia" in dev_kind or str(dev).startswith("cuda"):
            cuda_available = True
        elif plat == "gpu":
            try:
                backend = jax.default_backend().lower()
            except Exception:
                backend = ""
            if backend == "rocm":
                rocm_available = True
            elif backend == "cuda":
                cuda_available = True
            elif "ROCM" in handler_keys and "CUDA" not in handler_keys:
                rocm_available = True
            else:
                cuda_available = True

    if rocm_available and "ROCM" in handler_keys:
        return "ROCM"
    if cuda_available and "CUDA" in handler_keys:
        return "CUDA"
    if rocm_available:
        return "ROCM"
    if cuda_available:
        return "CUDA"
    # Fallback; custom calls will not actually run on CPU but registering
    # them there is harmless.
    return "CUDA"


def register_ffi_targets() -> None:
    """Register the UCCL EP custom-call targets exactly once per process.

    Raises ``RuntimeError`` if ``uccl.ep.get_jax_ffi_targets`` is absent,
    which happens on older builds of ``uccl.ep`` without the JAX
    bindings.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    with _REGISTER_LOCK:
        if _REGISTERED:
            return
        import jax

        try:
            from uccl import ep as _uccl_ep
        except ImportError as exc:
            raise RuntimeError(
                "uccl.ep is not installed; install it before registering "
                "UCCL-EP JAX FFI targets."
            ) from exc

        if not hasattr(_uccl_ep, "get_jax_ffi_targets"):
            raise RuntimeError(
                "This build of uccl.ep does not expose `get_jax_ffi_targets`. "
                "Rebuild uccl.ep from a version that includes the JAX FFI "
                "bridge (see ep/src/uccl_ep.cc)."
            )

        targets = _uccl_ep.get_jax_ffi_targets()
        platform = _active_platform()
        for name in TARGET_NAMES:
            capsule = targets[name]
            # C++ side uses the XLA Typed FFI
            # (``XLA_FFI_DEFINE_HANDLER_SYMBOL``), where each capsule
            # wraps an ``XLA_FFI_Handler*`` function. The matching
            # ``jax.ffi.register_ffi_target`` value is ``api_version=1``
            # (XLA ``API_VERSION_TYPED_FFI``), and on the call site
            # ``jax.ffi.ffi_call`` lowers to MLIR with
            # ``custom_call_api_version=4`` (the default, which we rely
            # on in :mod:`uccl_ep_jax.primitive._calls`).
            #
            # Legacy custom-call ABIs (ORIGINAL / STATUS_RETURNING) had
            # their XLA:GPU execution paths removed in late-2025 and can
            # no longer be used - attempting to register or call them
            # results in a ``CustomCallThunk`` SIGSEGV at runtime.
            jax.ffi.register_ffi_target(
                name, capsule, platform=platform, api_version=1
            )
        _REGISTERED = True
