"""Thin compatibility shim — all implementation lives in uccl.ep.buffer.

Bench scripts traditionally do ``from buffer import Buffer``. Keep that
working without duplicating the source file: re-export the canonical class
so ``buffer.Buffer is uccl.ep.buffer.Buffer``.
"""

from uccl.ep.buffer import Buffer  # noqa: F401

__all__ = ["Buffer"]
