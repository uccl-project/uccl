"""Compatibility layer exposing :mod:`uccl.ep` through the historical ``deep_ep`` API."""

from __future__ import annotations

import sys

try:  # Preserve DeepEP's implicit torch import (best-effort only).
    import torch  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - torch is optional.
    pass

from uccl import __version__ as __uccl_version__
from uccl.ep import (  # type: ignore F401 - symbols are re-exported.
    Buffer,
    Config,
    EventOverlap,
    destroy_uccl,
    ep_cpp,
    initialize_uccl,
    test_internode,
    buffer as _buffer_module,
    utils as _utils_module,
)

# Expose module attributes for attribute access (e.g. ``deep_ep.buffer``).
buffer = _buffer_module
utils = _utils_module

# Ensure ``import deep_ep.buffer`` and peers succeed.
sys.modules.setdefault(__name__ + ".buffer", buffer)
sys.modules.setdefault(__name__ + ".utils", utils)
sys.modules.setdefault(__name__ + ".ep_cpp", ep_cpp)

__all__ = [
    "Buffer",
    "Config",
    "EventOverlap",
    "initialize_uccl",
    "destroy_uccl",
    "test_internode",
    "buffer",
    "utils",
    "ep_cpp",
    "__version__",
]

# Align version string with the bundled uccl package.
__version__ = __uccl_version__
