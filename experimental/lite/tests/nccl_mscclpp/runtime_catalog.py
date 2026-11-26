"""Check the unified ABI's catalog and capability-driven candidate matching."""
import ctypes
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
import runtime_profile as profile


def main():
    libraries = {
        'nccl_runtime/libuccl_lite_nccl_runtime.so': {'peer-packet', 'peer-staged', 'peer-staged-fp32', 'host-packet', 'prepared-local', 'prepared-local-fp32', 'network-hierarchy', 'prepared-distributed', 'prepared-distributed-fp32'},
        'nccl_local/libuccl_lite_nccl_local_strict.so': {'peer-packet'},
        'nccl_local8/libuccl_lite_nccl_local8.so': {'host-packet'},
        'nccl_lite/libuccl_lite_nccl_strict.so': {'network-hierarchy'},
    }
    all_backends = None
    for name, expected in libraries.items():
        library = ROOT / 'build' / name
        backends = profile.catalog(library)
        handle = ctypes.CDLL(str(library))
        handle.ncclGetUniqueId.argtypes = [ctypes.c_void_p]
        first, second = (ctypes.c_ubyte * 128)(), (ctypes.c_ubyte * 128)()
        assert handle.ncclGetUniqueId(first) == 0 and handle.ncclGetUniqueId(second) == 0
        assert bytes(first) != bytes(second), 'unique IDs must not depend on a selected backend/profile'

        assert {b['name'] for b in backends} == expected
        dynamic = subprocess.check_output(['readelf', '-d', str(library)], text=True)
        assert not any('NEEDED' in line and 'libnccl' in line for line in dynamic.splitlines())
        if name.startswith('nccl_runtime/'):
            all_backends = backends
    assert all_backends
    def host(name, n):
        return dict(hostname=name, identity='synthetic', versions=[1, 2, 3], uuids=[str(i) for i in range(n)])
    with tempfile.TemporaryDirectory(dir=ROOT / 'build') as tmp:
        path = Path(tmp) / 'profile'
        for dtype, records, expected in [('float32', [host('a', 4)], 'peer-packet'),
                                          ('bfloat16', [host('a', 8)], 'host-packet'),
                                          ('float32', [host('a', 4), host('b', 4)], 'network-hierarchy')]:
            assert profile.write_candidate(path, records, [(1, 128), (2, 128), (3, 128)], dtype, all_backends, expected) == expected
            assert path.read_text().splitlines()[0] == 'UCCL_LITE_RUNTIME_PROFILE_V1'
            assert path.read_text().splitlines()[1].split()[0] == expected
        for dtype, expected in [('float32', 'prepared-distributed-fp32'), ('bfloat16', 'prepared-distributed')]:
            for layout in ([1, 1], [3, 1], [3, 3], [4, 4], [1, 2, 3]):
                records = [host(str(i), n) for i, n in enumerate(layout)]
                shapes = [(op, count, [[False, 4 if dtype == 'float32' else 2]] * sum(layout))
                          for op in (1, 2, 3) for count in (1, 257)]
                assert profile.write_candidate(path, records, shapes, dtype, all_backends) == expected
                assert path.read_text().splitlines()[0] == 'UCCL_LITE_RUNTIME_PROFILE_V2'
            generic = next(b for b in all_backends if b['name'] == expected)
            assert generic['extent_multiple'] == 1
            for bad in ([1], [32, 1], [0, 2], [33]):
                assert not profile.supports_layout(generic, bad)
        try:
            profile.write_candidate(path, [host('a', 4), host('b', 4)], [(3, 257)],
                                    'float32', all_backends, 'network-hierarchy')
        except ValueError:
            pass
        else:
            raise AssertionError('hierarchy byte granularity was relaxed')
        def owned(rank, host, view, device):
            uuids = [f'{n:032x}' for n in view]
            binding = dict(rank=rank, device=device, gpu=uuids[device])
            details = dict(hostname=host, visible_cuda_uuids=uuids, rank_binding=binding)
            return dict(hostname=host, details=details, versions=[1, 2, 3], uuids=uuids,
                        identity=hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest(), **binding)
        records = [owned(0, 'a', [1], 0), owned(1, 'b', [3, 4], 1),
                   owned(2, 'a', [2, 1], 0), owned(3, 'b', [3], 0)]
        shapes = [(3, 257, [[False, 4]] * 4)]
        assert profile.write_rank_candidate(path, records, shapes, 'float32', all_backends) == 'prepared-distributed-fp32'
        lines = path.read_text().splitlines()
        assert lines[0] == 'UCCL_LITE_RUNTIME_PROFILE_V4'
        for record in records:
            profile.verify_rank_profile(lines, record, all_backends)
        def rejected_rank_records(values):
            try:
                profile.write_rank_candidate(path, values, shapes, 'float32', all_backends)
            except ValueError:
                return
            raise AssertionError('invalid rank record admitted')
        rejected_rank_records(list(reversed(records)))
        bad = copy.deepcopy(records); bad[2]['gpu'] = records[0]['gpu']; rejected_rank_records(bad)
        bad = copy.deepcopy(records); bad[0]['details']['visible_cuda_uuids'] = [f'{7:032x}']; rejected_rank_records(bad)
        bad = copy.deepcopy(records); bad[0]['device'] = 9; rejected_rank_records(bad)
        try:
            profile.verify_rank_profile(lines, owned(0, 'a', [2, 1], 1), all_backends)
        except RuntimeError:
            pass
        else:
            raise AssertionError('changed per-rank visibility reused identity')
        older = [dict(b, profile_versions=[v for v in b['profile_versions'] if v != 4]) for b in all_backends]
        try:
            profile.write_rank_candidate(path, records, shapes, 'float32', older)
        except ValueError:
            pass
        else:
            raise AssertionError('V4 profile emitted for unsupported provider')
        for provider in ('host-packet', 'peer-staged'):
            assert profile.write_candidate(path, [host('a', 3)], [(1, 128)], 'bfloat16', all_backends, provider) == provider
        host_provider = next(b for b in all_backends if b['name'] == 'host-packet')
        assert profile.supports_layout(host_provider, [3])
        assert not profile.supports_layout(host_provider, [4, 4])
        assert not profile.supports_layout(host_provider, [33])

        assert profile.write_candidate(path, [host('a', 3)], [(1, 128)], 'float32', all_backends) == 'prepared-local-fp32'
        assert path.read_text().splitlines()[0] == 'UCCL_LITE_RUNTIME_PROFILE_V2'
        fp32_auto = next(b for b in all_backends if b['name'] == 'prepared-local-fp32')
        assert profile.write_candidate(path, [host('a', 4)], [(1, 64, [[True, 256]] * 4)], 'float32', all_backends) == 'prepared-local-fp32'
        try:
            profile.write_candidate(path, [host('a', 4)], [(1, 64, [[False, 2]] * 4)], 'float32', all_backends)
        except ValueError:
            pass
        else:
            raise AssertionError('FP32 scalar alignment was ignored')
        rejected = False
        try:
            profile.write_candidate(path, [host('a', 4)], [(1, 128)], 'float32', all_backends + [fp32_auto])
        except ValueError:
            rejected = True
        assert rejected, 'ambiguous providers were selected without evidence'
        for dtype, provider in [('float32', 'peer-staged'),
                                ('bfloat16', 'unknown')]:
            try:
                profile.write_candidate(path, [host('a', 4)], [(1, 128)], dtype, all_backends, provider)
            except ValueError:
                pass
            else:
                raise AssertionError('ambiguous or incompatible provider was admitted')
        assert profile.write_candidate(path, [host('a', 3)], [(1, 128)], 'bfloat16', all_backends) == 'prepared-local'
        assert path.read_text().splitlines()[0] == 'UCCL_LITE_RUNTIME_PROFILE_V2'
        assert path.read_text().splitlines()[-1] == '1 128 512 512 512'
        mapping = [[rank % 2, rank // 2] for rank in range(8)]
        records = [host('a', 4), host('b', 4)]
        assert profile.write_candidate(path, records, [(1, 128)], 'float32', all_backends,
                                       'network-hierarchy', mapping) == 'network-hierarchy'
        assert path.read_text().splitlines()[0] == 'UCCL_LITE_RUNTIME_PROFILE_V3'
        assert path.read_text().splitlines()[2] == 'ranks 8 ' + ' '.join(str(v) for pair in mapping for v in pair)
        assert profile.write_candidate(path, [host('a', 3)], [(1, 128)], 'bfloat16', all_backends,
                                       'prepared-local', [[0, 2], [0, 0], [0, 1]]) == 'prepared-local'
        assert path.read_text().splitlines()[-1] == '1 128 512 512 512'
        for bad in [mapping[:-1], mapping[:-1] + [mapping[0]], mapping[:-1] + [[2, 0]],
                    mapping[:-1] + [[1, 4]], mapping[:-1] + [[True, 3]], 8]:
            try:
                profile.write_candidate(path, records, [(1, 128)], 'float32', all_backends, 'network-hierarchy', bad)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid rank binding admitted')
        older = [dict(b, profile_versions=[1]) for b in all_backends]
        try:
            profile.write_candidate(path, records, [(1, 128)], 'float32', older, 'network-hierarchy', mapping)
        except ValueError:
            pass
        else:
            raise AssertionError('rank-map profile emitted for an older frontend')
        assert profile.write_candidate(path, [host('a', 3)], [(2, 128, [[False, 2], [True, 16], [False, 256]])], 'bfloat16', all_backends) == 'prepared-local'
        assert path.read_text().splitlines()[-1] == '2 128 4 33 512'
        for shapes in [[(3, 64, [[False, 3]] * 3)], [(3, 64, [[False, 16]])],
                       [(3, 64), (3, 64, [[True, 16]] * 3)]]:
            try:
                profile.write_candidate(path, [host('a', 3)], shapes, 'bfloat16', all_backends)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid or ambiguous prepared layout admitted')
    print('PASS: shared ABI catalog, isolated legacy providers, dtype/layout matching and no native NCCL dependency')


if __name__ == '__main__':
    main()
