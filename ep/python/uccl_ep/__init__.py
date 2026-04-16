"""
uccl.ep — Expert-Parallel communication for Mixture-of-Experts models.

This package provides both the native C++/CUDA extension (compiled as
``_ep_native``) and Python-level helpers (``Buffer``, ``EventOverlap``, etc.).

Public API
----------
* Everything exported by the native extension (``Config``, ``EventHandle``,
  ``Buffer`` (native), ``Proxy``, helper functions, …) is available directly
  as ``uccl.ep.<name>``.
* High-level Python wrappers live in submodules:
  - ``uccl.ep.buffer.Buffer``   — the main user-facing ``Buffer`` class
  - ``uccl.ep.utils``           — ``EventOverlap``, ``initialize_uccl``, etc.

For backward compatibility, ``from uccl.ep import Config, EventHandle`` still
works (they come from the native extension), and
``from uccl.ep import Buffer`` returns the **Python** wrapper class.
"""

from uccl.ep._ep_native import *  # noqa: F401,F403 — re-export native symbols

# Keep a reference so users can do ``from uccl.ep import _ep_native`` when
# they need the raw C++ module (e.g. ``_ep_native.Buffer`` vs the Python
# wrapper ``Buffer``).
from uccl.ep import _ep_native  # noqa: F401

# Import the Python wrapper ``Buffer`` *after* the wildcard import so it
# shadows the native ``Buffer`` class with the richer Python version.
from uccl.ep.buffer import Buffer  # noqa: F401
from uccl.ep.utils import (  # noqa: F401
    EventOverlap,
    check_nvlink_connections,
    initialize_uccl,
    destroy_uccl,
)
