"""PCI identity checks independent of GPU vendor and domain formatting."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import test_engine_onesided_ipc as ipc


@pytest.mark.parametrize("address", ["0000:ca:00.0", "00000000:CA:00.0"])
def test_bdf_case_and_padding(address):
    assert ipc._normalize_bdf(address) == "0000:ca:00.0"


def test_distinct_pci_domains_stay_distinct():
    assert ipc._normalize_bdf("0008:01:00.0") != ipc._normalize_bdf("0009:01:00.0")


@pytest.mark.parametrize("hip", [None, "6.4"])
def test_active_runtime_query(monkeypatch, hip):
    def query(buffer, size, ordinal):
        assert size == 32 and ordinal == 7
        buffer.value = b"00000009:CA:00.0"
        return 0

    cuda_query = Mock(side_effect=query)
    hip_query = Mock(side_effect=query)
    runtime = SimpleNamespace(
        cudaDeviceGetPCIBusId=cuda_query, hipDeviceGetPCIBusId=hip_query
    )
    monkeypatch.setattr(ipc.ctypes, "CDLL", lambda path: runtime)
    monkeypatch.setattr(ipc.torch.version, "hip", hip)
    assert ipc._runtime_bdf(7) == "0009:ca:00.0"
    assert cuda_query.call_count == (0 if hip else 1)
    assert hip_query.call_count == (1 if hip else 0)


def test_binding_detects_domain_mismatch(monkeypatch):
    endpoint = SimpleNamespace(parse_metadata=lambda metadata: (0, 0, "0008:01:00.0"))
    monkeypatch.setattr(ipc, "p2p", SimpleNamespace(Endpoint=endpoint))
    monkeypatch.setattr(ipc, "_runtime_bdf", lambda ordinal: "0009:01:00.0")
    with pytest.raises(AssertionError, match="device binding mismatch"):
        ipc._assert_device_binding(SimpleNamespace(get_metadata=lambda: b"metadata"), 0)
