"""Install the opt-in UCCL Lite SGLang hook in spawned model workers."""

import importlib.abc
import importlib.machinery
import os
import sys


if (
    os.environ.get("UCCL_LITE_SGLANG_RS_SUM2") == "1"
    or os.environ.get("UCCL_LITE_SGLANG_AG_GATE") == "1"
    or os.environ.get("UCCL_LITE_SGLANG_READY_COMBINE") == "1"
):
    _TARGET = "sglang.srt.models.glm4_moe"

    class _PatchLoader(importlib.abc.Loader):
        def __init__(self, base):
            self.base = base

        def create_module(self, spec):
            if hasattr(self.base, "create_module"):
                return self.base.create_module(spec)
            return None

        def exec_module(self, module):
            self.base.exec_module(module)
            if os.environ.get("UCCL_LITE_SGLANG_RS_SUM2") == "1":
                from rs_sum2 import install as install_rs

                install_rs(module)
            if os.environ.get("UCCL_LITE_SGLANG_READY_COMBINE") == "1":
                from ready_combine import install as install_ready

                # AG+gate must wrap this forward_normal hook to preserve ready RS.
                install_ready(module)
            if os.environ.get("UCCL_LITE_SGLANG_AG_GATE") == "1":
                from ag_gate import install as install_ag

                install_ag(module)

    class _PatchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = _PatchLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, _PatchFinder())
