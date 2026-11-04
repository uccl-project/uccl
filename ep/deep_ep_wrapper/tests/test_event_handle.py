"""deep_ep.EventHandle() must record on the current stream, as DeepEP's does (#1121)."""

import ast
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

WRAPPER_DIR = Path(__file__).parents[1]
UTILS = WRAPPER_DIR.parent / "bench" / "utils.py"


def _load_event_handle(current_stream_ptr):
    # Execute only EventHandle from utils.py, without importing torch or uccl.ep.
    tree = ast.parse(UTILS.read_text())
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "EventHandle"
        ),
        None,
    )
    assert function is not None, "deep_ep/utils.py does not define EventHandle"
    recorded_on = []

    class UcclEventHandle:
        def __init__(self, stream_ptr):
            recorded_on.append(stream_ptr)

    current_stream = SimpleNamespace(cuda_stream=current_stream_ptr)
    torch = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: current_stream))
    namespace = {
        "Optional": Optional,
        "torch": torch,
        "_UcclEventHandle": UcclEventHandle,
    }
    exec(  # noqa: S102 -- execute only a trusted function from this repository
        compile(ast.Module(body=[function], type_ignores=[]), str(UTILS), "exec"),
        namespace,
    )
    return namespace["EventHandle"], recorded_on


def test_event_handle_records_on_current_stream_by_default():
    event_handle, recorded_on = _load_event_handle(current_stream_ptr=0x1234)

    event_handle()
    event_handle(0x5678)

    assert recorded_on == [0x1234, 0x5678]


def test_deep_ep_exports_stream_aware_event_handle():
    init = ast.parse((WRAPPER_DIR / "deep_ep" / "__init__.py").read_text())
    imported_from = {
        alias.name: (node.module, node.level)
        for node in init.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert imported_from["EventHandle"] == ("utils", 1)


@pytest.fixture
def cuda_deep_ep(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1":
        pytest.skip("one hardware queue hides a missing event dependency")
    pytest.importorskip("uccl.ep")

    def is_deep_ep_module(name):
        return name == "deep_ep" or name.startswith("deep_ep.")

    previous_modules = {
        name: module for name, module in sys.modules.items() if is_deep_ep_module(name)
    }
    for name in previous_modules:
        sys.modules.pop(name)
    monkeypatch.syspath_prepend(str(WRAPPER_DIR))

    yield torch, importlib.import_module("deep_ep")

    for name in list(sys.modules):
        if is_deep_ep_module(name):
            sys.modules.pop(name)
    sys.modules.update(previous_modules)


def test_event_handle_waits_for_work_on_current_stream(cuda_deep_ep):
    """Single-GPU reproducer of #1121: the consumer must see what the producer stream wrote."""
    torch, deep_ep = cuda_deep_ep
    x = torch.zeros(1 << 20, device="cuda")
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(200_000_000)
        x.fill_(1.0)
        event = deep_ep.EventOverlap(deep_ep.EventHandle())
    with torch.cuda.stream(consumer):
        event.current_stream_wait()
        y = x.clone()
    torch.cuda.synchronize()

    assert torch.all(
        y == 1.0
    ), "the consumer read the tensor before the producer wrote it"
