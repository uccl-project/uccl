"""Lite-EP user-facing environment resolver.

Lite-EP exposes only three runtime environment variables to users:

- ``LITE_EP_TRANSPORT`` -- one of ``uccl-no-gdr`` (default), ``uccl-gdr``,
  ``nccl-gdr``, ``nccl-no-gdr``. Selects the EP datapath (backend +
  GDR/no-GDR).
- ``LITE_EP_NVLINK`` -- ``0`` (default, NVLink off) or ``1`` (NVLink on).
  Must be ``0`` when ``LITE_EP_TRANSPORT`` is a UCCL mode.
- ``LITE_EP_VERBOSE`` -- ``0`` (default), ``1``, ``2``. Reserved for the
  unified debug/trace knob (consumed elsewhere).

The legacy names ``EP_USE_UCCL_PROXY`` / ``UCCL_FORCE_NO_GDR`` /
``EP_FORCE_HOST_WINDOW`` / ``EP_FORCE_NO_NVLINK`` / ``EP_DISABLE_GIN`` are
**no longer user-facing**. They are still used as the internal wire format
between the Python frontend and the C++ / JIT layers; this module sets them
based on the two new vars and rejects any direct user setting with a clear
error.
"""

from __future__ import annotations

import os

_TRANSPORTS = ('uccl-no-gdr', 'uccl-gdr', 'nccl-gdr', 'nccl-no-gdr')

# Legacy env names that are no longer user-facing. Setting any of these
# directly is an error -- it bypasses the resolver and leaves the
# Python/C++ sides inconsistent.
_REMOVED = (
    'EP_USE_UCCL_PROXY',
    'UCCL_FORCE_NO_GDR',
    'EP_FORCE_HOST_WINDOW',
    'EP_FORCE_NO_NVLINK',
    'EP_DISABLE_GIN',
)

_RESOLVED: dict[str, object] | None = None

# Sentinel env var that marks "this process tree has already gone through
# the resolver in some ancestor". When set, we skip the legacy-name guard
# (legacy names are set by the resolver itself as internal wire format).
_SENTINEL = '_LITE_EP_RESOLVED'


def _fail(msg: str) -> None:
    raise RuntimeError(f'[lite-ep env] {msg}')


def resolve() -> dict[str, object]:
    """Resolve LITE_EP_* into the internal env vars consumed by C++/Python.

    Safe to call more than once; the first call wins and subsequent calls
    return the cached result.
    """
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED

    if os.environ.get(_SENTINEL) != '1':
        for name in _REMOVED:
            if name in os.environ:
                _fail(
                    f'{name} is no longer user-facing. '
                    f'Use LITE_EP_TRANSPORT={{{"|".join(_TRANSPORTS)}}} '
                    f'and LITE_EP_NVLINK={{0,1}} instead.'
                )

    transport = os.environ.get('LITE_EP_TRANSPORT', 'uccl-no-gdr')
    if transport not in _TRANSPORTS:
        _fail(
            f'LITE_EP_TRANSPORT={transport!r} is invalid; '
            f'choose one of {list(_TRANSPORTS)}.'
        )

    nvlink_raw = os.environ.get('LITE_EP_NVLINK', '0')
    try:
        nvlink = int(nvlink_raw)
    except ValueError:
        _fail(f'LITE_EP_NVLINK={nvlink_raw!r} is not an integer (expected 0 or 1).')
        raise  # unreachable, for type checkers
    if nvlink not in (0, 1):
        _fail(f'LITE_EP_NVLINK={nvlink} is invalid; choose 0 (off) or 1 (on).')

    uccl = transport.startswith('uccl-')
    no_gdr = transport == 'uccl-no-gdr'

    # UCCL proxy mode asserts no-NVLink in buffer.hpp; surface that as a
    # config error here instead of letting the C++ assert fire later.
    if uccl and nvlink == 1:
        _fail(
            f'LITE_EP_TRANSPORT={transport!r} requires LITE_EP_NVLINK=0 '
            f'(UCCL proxy mode runs each rank with a singleton scaleup domain).'
        )

    # Translate to the internal env vars. These names are intentionally the
    # same as the historical knobs so that existing C++/JIT/Python read sites
    # keep working unchanged.
    internal = {
        'EP_USE_UCCL_PROXY': '1' if uccl else '0',
        'UCCL_FORCE_NO_GDR': '1' if no_gdr else '0',
        'EP_FORCE_HOST_WINDOW': '1' if no_gdr else '0',
        'EP_FORCE_NO_NVLINK': '0' if nvlink else '1',
        # GIN is the NCCL-only fast path; disable whenever UCCL proxy owns
        # the EP datapath. We do not expose a separate knob for "NCCL no
        # GIN" -- if that path is needed, drive it from source.
        'EP_DISABLE_GIN': '1' if uccl else '0',
        # Pre-set the JIT-visible gate so the first compile picks up
        # -DEP_USE_UCCL_PROXY even before Buffer construction sets it.
        'EP_UCCL_PROXY_ACTIVE': '1' if uccl else '0',
    }
    for k, v in internal.items():
        os.environ[k] = v

    # NCCL GDR level is a NCCL knob but we manage it here so users do not
    # have to remember to set it alongside the transport.
    os.environ.setdefault('NCCL_NET_GDR_LEVEL', '0' if no_gdr else '5')

    # Mark this process tree as resolved so child processes that inherit
    # the translated legacy env vars do not trip the legacy-name guard.
    os.environ[_SENTINEL] = '1'

    _RESOLVED = {
        'transport': transport,
        'nvlink': nvlink,
        'internal': internal,
    }
    return _RESOLVED
