"""CPU regression for preserving caller ownership through CUDA inventory queries."""
import ctypes
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
import runtime_profile as profile

def integer(value):
    def call(pointer):
        pointer._obj.value = value
        return 0
    return call

def get_device(pointer, ordinal):
    pointer._obj.value = ordinal
    return 0

def get_uuid(buffer, device):
    for i in range(16):
        buffer[i] = i
    return 0

cuda = SimpleNamespace(cudaDriverGetVersion=integer(13000), cudaRuntimeGetVersion=integer(12090))
strict = SimpleNamespace(ncclGetVersion=integer(22807), uccl_lite_nccl_runtime_topology=lambda: b'topology')
driver = SimpleNamespace(cuInit=lambda _: 0, cuDeviceGetCount=integer(1), cuDeviceGet=get_device,
                         cuDeviceGetUuid=get_uuid, cuDeviceGetUuid_v2=get_uuid)
def library(path):
    return driver if str(path) == 'libcuda.so.1' else cuda if 'cudart' in str(path) else strict

with patch.object(profile.common, 'identity', return_value=('unused', {'hostname': 'host', 'packet': 'legacy'})), \
     patch.object(profile.common, 'sha256', return_value='artifact'), \
     patch.object(profile.subprocess, 'check_output', return_value='libcudart.so.12 => /fake/libcudart.so.12\n'), \
     patch.object(profile.ctypes, 'CDLL', side_effect=library), \
     patch.object(profile, 'catalog', return_value=[]), \
     patch.object(profile.platform, 'node', return_value='host'):
    record = profile.record(None, Path('/fake/strict.so'), rank=7, device=0)
    assert record['rank'] == 7 and record['device'] == 0
    assert record['gpu'] == bytes(range(16)).hex()
    assert record['details']['rank_binding'] == {'rank': 7, 'device': 0, 'gpu': record['gpu']}
    assert 'packet' not in record['details']
    for rank, device in [(7, 1), (32, 0), (True, 0), (7, None)]:
        try:
            profile.record(None, Path('/fake/strict.so'), rank=rank, device=device)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid rank/device ownership accepted')
print('PASS: rank ownership survives device enumeration; invalid bindings rejected')
