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
    "uccl_ll_dispatch",
    "uccl_ll_combine",
    "uccl_moe_dispatch",
    "uccl_moe_combine",
)


def _active_platform() -> str:
    import jax

    # On ROCm, jax uses the platform string "rocm"; on CUDA it uses "cuda".
    for dev in jax.local_devices():
        plat = getattr(dev, "platform", "").lower()
        if plat in ("cuda", "rocm"):
            return plat
    # Fallback; custom calls will not actually run on CPU but registering
    # them there is harmless.
    return "cuda"


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
            # api_version=0 means "legacy custom call" (stream + buffers + opaque).
            jax.ffi.register_ffi_target(
                name, capsule, platform=platform, api_version=0
            )
        _REGISTERED = True
