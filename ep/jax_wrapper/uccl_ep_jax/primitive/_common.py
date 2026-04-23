"""Shared helpers for the UCCL-EP primitives."""

from __future__ import annotations

import numpy as np


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


def itemsize(dtype) -> int:
    return int(np.dtype(dtype).itemsize)
