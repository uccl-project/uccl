import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def isolated_deep_ep_import(monkeypatch):
    def is_deep_ep_module(name):
        return name == "deep_ep" or name.startswith("deep_ep.")

    previous_modules = {
        name: module for name, module in sys.modules.items() if is_deep_ep_module(name)
    }
    for name in previous_modules:
        sys.modules.pop(name)

    yield monkeypatch

    for name in list(sys.modules):
        if is_deep_ep_module(name):
            sys.modules.pop(name)
    sys.modules.update(previous_modules)


class _StubTorch(ModuleType):
    def __getattr__(self, name):
        placeholder = type(name, (), {})
        setattr(self, name, placeholder)
        return placeholder


def _load_deep_ep_wrapper(monkeypatch, current_device):
    torch = _StubTorch("torch")
    torch.cuda = ModuleType("torch.cuda")
    torch.cuda.current_device = current_device

    torch_dist = ModuleType("torch.distributed")
    torch_dist.ProcessGroup = type("ProcessGroup", (), {})
    torch.distributed = torch_dist

    uccl = ModuleType("uccl")
    uccl_ep = ModuleType("uccl.ep")
    uccl_ep.Config = type("Config", (), {})
    uccl_ep.EventHandle = type("EventHandle", (), {})
    uccl.ep = uccl_ep

    wrapper_utils = ModuleType("deep_ep.utils")
    wrapper_utils.EventOverlap = type("EventOverlap", (), {})
    wrapper_utils.check_nvlink_connections = lambda *args, **kwargs: None
    wrapper_utils.initialize_uccl = lambda *args, **kwargs: None
    wrapper_utils.destroy_uccl = lambda *args, **kwargs: None
    wrapper_utils._fp8_e4m3_dtype = object()

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", torch_dist)
    monkeypatch.setitem(sys.modules, "uccl", uccl)
    monkeypatch.setitem(sys.modules, "uccl.ep", uccl_ep)
    monkeypatch.setitem(sys.modules, "deep_ep.utils", wrapper_utils)
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))

    return importlib.import_module("deep_ep"), torch.cuda


def test_enable_shrink_compatibility(isolated_deep_ep_import):
    monkeypatch = isolated_deep_ep_import

    def fail_if_cuda_is_queried():
        raise AssertionError("enable_shrink=True reached CUDA initialization")

    deep_ep, cuda = _load_deep_ep_wrapper(monkeypatch, fail_if_cuda_is_queried)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    with pytest.raises(
        NotImplementedError,
        match="UCCL EP does not currently support enable_shrink=True",
    ):
        deep_ep.Buffer(group=object(), enable_shrink=True)

    class ReachedNormalInitialization(Exception):
        pass

    def mark_normal_initialization():
        raise ReachedNormalInitialization

    cuda.current_device = mark_normal_initialization
    with pytest.raises(ReachedNormalInitialization):
        deep_ep.Buffer(group=object(), enable_shrink=False)
