"""Thin compatibility shim — all implementation lives in uccl.ep.test_internode.

Bench scripts and ``torchrun bench/test_internode.py`` continue to work by
delegating to the canonical module, avoiding the dual-loading pitfalls of a
filesystem symlink.
"""

from uccl.ep.test_internode import *  # noqa: F401,F403


if __name__ == "__main__":
    # The canonical module's ``__main__`` block (argparse + torchrun entry)
    # only fires when *that* file is executed directly. Re-execute it here so
    # ``torchrun bench/test_internode.py`` keeps working.
    import runpy

    runpy.run_module("uccl.ep.test_internode", run_name="__main__")
