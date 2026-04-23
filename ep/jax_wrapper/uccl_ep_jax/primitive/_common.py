"""Shared helpers for the UCCL-EP primitives."""

from __future__ import annotations

import struct

import numpy as np


def pack_ints32(*values: int) -> bytes:
    """Pack an arbitrary number of int32 values into a bytes blob.

    The resulting blob is used as the ``backend_config`` opaque attribute
    of the XLA custom call.  The native (host) endianness is used, which
    matches how the C++ side ``reinterpret_cast``s the bytes back into a
    struct.
    """
    fmt = "i" * len(values)
    return struct.pack(fmt, *[int(v) for v in values])


def np_dtype_to_code(dtype) -> int:
    """Map a numpy/JAX dtype to the integer codes used by ``uccl.ep``."""
    import jax.numpy as jnp

    dt = np.dtype(dtype)
    table = [
        (jnp.uint8, 0),
        (jnp.int8, 1),
        (jnp.int16, 2),
        (jnp.int32, 3),
        (jnp.int64, 4),
        (jnp.float16, 5),
        (jnp.bfloat16, 6),
        (jnp.float32, 7),
        (jnp.float64, 8),
        (jnp.bool_, 9),
    ]
    if hasattr(jnp, "float8_e4m3fn") and dt == np.dtype(jnp.float8_e4m3fn):
        return 10
    if hasattr(jnp, "float8_e4m3fnuz") and dt == np.dtype(jnp.float8_e4m3fnuz):
        return 10
    for t, code in table:
        if dt == np.dtype(t):
            return code
    raise ValueError(f"Unsupported dtype for uccl.ep: {dtype}")


def default_row_major_layout(ndim: int):
    """Return the ``stablehlo.custom_call`` layout tuple for a row-major
    array of the given rank (major-to-minor: ``[n-1, n-2, ..., 0]``)."""
    return tuple(range(ndim - 1, -1, -1))


def itemsize(dtype) -> int:
    return int(np.dtype(dtype).itemsize)
