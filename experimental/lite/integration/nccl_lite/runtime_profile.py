"""Discover compiled runtime capabilities and launch explicit qualified-shape experiments.

This adapter does not turn historical timings into automatic admission. It
selects a uniquely matching registered implementation from dtype and observed
host layout, or an explicitly named eligible experiment provider; the ABI
validates identity, shapes and transport capabilities. Ambiguity never chooses
the first provider in the registry.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import dropin_profile as common


def catalog(library):
    lib = ctypes.CDLL(str(library))
    function = lib.uccl_lite_nccl_runtime_backends
    function.restype = ctypes.c_char_p
    return json.loads(function().decode())


def descriptor(record):
    return [record['hostname'], record['identity'], *map(str, record['versions']), *record['uuids']]


def record(real, library, rank=None, device=None):
    common.LIBRARY = library
    _, details = common.identity(real)
    # The common helper also serves a historical eight-GPU launcher. Its fixed
    # packet presets are not the live configuration of this runtime.
    details.pop('packet', None)
    network_profile = os.environ.get('UCCL_LITE_NETWORK_PROFILE')
    details['reference_kind'] = 'native-artifact-reference' if real else 'no-native-reference'
    details['network_profile_sha256'] = common.sha256(network_profile) if network_profile else None
    dependencies = subprocess.check_output(['ldd', str(library)], text=True)
    if re.search(r'libnccl\.so', dependencies) or 'not found' in dependencies:
        raise RuntimeError('strict runtime has a native NCCL or unresolved dependency')
    runtime = re.search(r'libcudart\.so[^ ]* => (\S+)', dependencies)
    if not runtime:
        raise RuntimeError('CUDA runtime dependency is unresolved')
    cuda = ctypes.CDLL(runtime[1])
    versions = []
    for lib, function in ((cuda, 'cudaDriverGetVersion'), (cuda, 'cudaRuntimeGetVersion'),
                          (ctypes.CDLL(str(library)), 'ncclGetVersion')):
        value = ctypes.c_int()
        if getattr(lib, function)(ctypes.byref(value)):
            raise RuntimeError(function + ' failed')
        versions.append(value.value)
    adapter = ctypes.CDLL(str(library))
    if hasattr(adapter, 'uccl_lite_nccl_runtime_topology'):
        adapter.uccl_lite_nccl_runtime_topology.restype = ctypes.c_char_p
        snapshot = adapter.uccl_lite_nccl_runtime_topology()
        if not snapshot:
            raise RuntimeError('runtime topology discovery failed')
        details['runtime_topology_identity'] = snapshot.decode()
    details.update(runtime_backends=catalog(library), launcher_sha256=common.sha256(__file__),
                   cudart_sha256=common.sha256(runtime[1]),
                   control_root=os.environ.get('UCCL_LITE_MSCCLPP_CONTROL_ROOT', '127.0.0.1'),
                   transport_environment={k: v for k, v in os.environ.items()
                                          if k.startswith(('MSCCLPP_', 'UCCL_LITE_')) and not k.startswith('UCCL_LITE_MSCCLPP_')})
    match = re.search(r'libmscclpp\.so[^ ]* => (\S+)', dependencies)
    details['mscclpp_sha256'] = common.sha256(match[1]) if match else None
    # NVML reports physical inventory even when CUDA_VISIBLE_DEVICES filters
    # or reorders CUDA ordinals. Admission must use the caller's CUDA view.
    driver = ctypes.CDLL('libcuda.so.1')
    count = ctypes.c_int()
    if driver.cuInit(0) or driver.cuDeviceGetCount(ctypes.byref(count)) or count.value < 1:
        raise RuntimeError('no visible CUDA devices')
    get_uuid = getattr(driver, 'cuDeviceGetUuid_v2', driver.cuDeviceGetUuid)
    uuids = []
    for ordinal in range(count.value):
        cuda_device = ctypes.c_int()
        uuid = (ctypes.c_ubyte * 16)()
        if driver.cuDeviceGet(ctypes.byref(cuda_device), ordinal) or get_uuid(uuid, cuda_device):
            raise RuntimeError('cannot query CUDA device identity')
        uuids.append(bytes(uuid).hex())
    details['visible_cuda_uuids'] = uuids
    binding = {}
    if rank is not None or device is not None:
        if type(rank) is not int or not 0 <= rank < 32 or type(device) is not int or not 0 <= device < len(uuids):
            raise ValueError('rank probing requires a valid logical rank and owned CUDA device')
        binding = dict(rank=rank, device=device, gpu=uuids[device])
        details['rank_binding'] = binding
    return dict(hostname=platform.node(), details=details, versions=versions,
                identity=hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest(),
                uuids=uuids, **binding)


def supports_layout(backend, layout):
    if not layout or any(n < 1 or n > 32 for n in layout) or sum(layout) > 32:
        return False
    if backend.get('max_hosts', 0):
        return (1 <= backend.get('min_hosts', 0) <= len(layout) <= backend['max_hosts'] <= 32
                and 1 <= backend.get('min_total_ranks', 0) <= sum(layout) <= backend.get('max_total_ranks', 0) <= 32
                and 1 <= backend.get('min_local_ranks', 0) <= backend.get('max_local_ranks', 0) <= 32
                and all(backend['min_local_ranks'] <= n <= backend['max_local_ranks'] for n in layout))
    return backend['ranks_per_host'] == layout or (len(layout) == 1 and
        backend.get('min_local_ranks', 0) > 0 and
        backend['min_local_ranks'] <= layout[0] <= backend['max_local_ranks'])


def validate_rank_map(layout, rank_map):
    if not isinstance(rank_map, (list, tuple)) or len(rank_map) != sum(layout):
        raise ValueError('one host/device binding per logical rank is required')
    seen = set()
    for pair in rank_map:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError('rank binding must be a host/device pair')
        host, device = pair
        if type(host) is not int or type(device) is not int or not 0 <= host < len(layout) or not 0 <= device < layout[host] or (host, device) in seen:
            raise ValueError('invalid or duplicate rank binding')
        seen.add((host, device))


def choose_backend(layout, dtype, backends, provider=None):
    eligible = [b for b in backends if b['dtype'] == dtype and supports_layout(b, layout)]
    if provider is not None:
        eligible = [b for b in eligible if b['name'] == provider]
    elif any(b.get('automatic') for b in eligible):
        eligible = [b for b in eligible if b.get('automatic')]
    if len(eligible) != 1:
        raise ValueError('expected one supported provider; ambiguous layouts require measured selection')
    backend = eligible[0]
    return backend


def shape_rows(shapes, dtype, layout, backend):
    rows = []
    element = 2 if dtype == 'bfloat16' else 4
    automatic = backend.get('automatic', False)
    for shape in shapes:
        if len(shape) not in (2, 3):
            raise ValueError('shape must contain op, count and optional rank layouts')
        op, count = shape[:2]
        scale = element * (1 if op == 3 else sum(layout))
        if op not in (1, 2, 3) or count <= 0 or count * scale > 1 << 30 or count * scale % backend.get('extent_multiple', 128):
            raise ValueError('unsupported exact candidate extent')
        row = f'{op} {count}'
        if automatic:
            placements = shape[2] if len(shape) == 3 else [[False, 256]] * sum(layout)
            if len(placements) != sum(layout):
                raise ValueError('one layout per logical rank is required')
            encoded = []
            for rank, (inplace, alignment) in enumerate(placements):
                if type(inplace) is not bool or type(alignment) is not int or alignment < element or alignment > 256 or alignment & (alignment - 1):
                    raise ValueError('invalid prepared rank layout')
                if inplace and op != 3 and rank * count * element % alignment:
                    raise ValueError('in-place shard offset violates alignment')
                encoded.append(str(2 * alignment + inplace))
            row += ' ' + ' '.join(encoded)
        elif len(shape) == 3:
            raise ValueError('rank layouts require an automatic prepared provider')
        rows.append(row)
    if not rows or len({tuple(row.split()[:2]) for row in rows}) != len(rows):
        raise ValueError('candidate requires unique exact shapes')
    return rows


def write_candidate(path, records, shapes, dtype, backends, provider=None, rank_map=None):
    if not records or len({r['hostname'] for r in records}) != len(records):
        raise ValueError('ordered, distinct host records are required')
    layout = [len(r['uuids']) for r in records]
    backend = choose_backend(layout, dtype, backends, provider)
    if rank_map is not None:
        if 3 not in backend.get('profile_versions', []):
            raise ValueError('compiled provider does not support rank-map profiles')
        validate_rank_map(layout, rank_map)
    automatic = backend.get('automatic', False)
    rows = shape_rows(shapes, dtype, layout, backend)
    magic = 'UCCL_LITE_RUNTIME_PROFILE_V3' if rank_map is not None else 'UCCL_LITE_RUNTIME_PROFILE_V2' if automatic else 'UCCL_LITE_RUNTIME_PROFILE_V1'
    header = magic + '\n' + backend['name'] + ' ' + str(len(layout)) + ' ' + ' '.join(map(str, layout))
    if rank_map is not None:
        header += '\nranks ' + str(len(rank_map)) + ' ' + ' '.join(str(v) for pair in rank_map for v in pair)
    Path(path).write_text(header + '\n' + '\n'.join(' '.join(descriptor(r)) for r in records)
                         + '\n' + '\n'.join(rows) + '\n')
    return backend['name']


def rank_descriptor(record, node):
    return [str(node), record['hostname'], record['identity'], *map(str, record['versions']), record['gpu']]


def write_rank_candidate(path, records, shapes, dtype, backends, provider=None):
    if not records or len(records) > 32:
        raise ValueError('one record per logical rank, at most 32, is required')
    hosts, layout, nodes, seen = [], [], [], set()
    for rank, r in enumerate(records):
        device, gpu, host = r.get('device'), r.get('gpu'), r.get('hostname')
        if (r.get('rank') != rank or type(r.get('rank')) is not int
                or type(device) is not int or not 0 <= device < len(r.get('uuids', []))
                or r['uuids'][device] != gpu or not isinstance(gpu, str)
                or not re.fullmatch('[0-9a-f]{32}', gpu) or not isinstance(host, str)
                or not host or any(c.isspace() for c in host)):
            raise ValueError('invalid rank/device/UUID record')
        details = r.get('details', {})
        if (details.get('rank_binding') != dict(rank=rank, device=device, gpu=gpu)
                or details.get('visible_cuda_uuids') != r['uuids']
                or details.get('hostname') != host
                or hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest() != r.get('identity')):
            raise ValueError('rank record identity does not bind its actual view/ownership')
        if len(r.get('versions', [])) != 3 or any(type(v) is not int or v <= 0 for v in r['versions']):
            raise ValueError('invalid rank version record')
        if (host, gpu) in seen:
            raise ValueError('duplicate owned GPU')
        seen.add((host, gpu))
        if host not in hosts:
            hosts.append(host); layout.append(0)
        node = hosts.index(host); nodes.append(node); layout[node] += 1
    backend = choose_backend(layout, dtype, backends, provider)
    if 4 not in backend.get('profile_versions', []) or not backend.get('automatic'):
        raise ValueError('provider does not support UUID rank profiles')
    rows = shape_rows(shapes, dtype, layout, backend)
    header = ('UCCL_LITE_RUNTIME_PROFILE_V4\n' + backend['name'] + ' ' + str(len(layout))
              + ' ' + ' '.join(map(str, layout)) + '\nparticipants ' + str(len(records)))
    Path(path).write_text(header + '\n' + '\n'.join(' '.join(rank_descriptor(r, node))
                         for r, node in zip(records, nodes)) + '\n' + '\n'.join(rows) + '\n')
    return backend['name']


def verify_rank_profile(lines, record, backends):
    fields = lines[1].split()
    backend = next((b for b in backends if b['name'] == fields[0]), None)
    layout = list(map(int, fields[2:])); hosts = int(fields[1])
    if not backend or 4 not in backend.get('profile_versions', []) or hosts != len(layout) or not supports_layout(backend, layout):
        raise RuntimeError('invalid UUID profile provider/layout')
    tag = lines[2].split()
    if tag != ['participants', str(sum(layout))] or len(lines) <= 3 + sum(layout):
        raise RuntimeError('invalid UUID participant extent')
    population = [0] * hosts; names = {}; seen = set()
    for row in lines[3:3 + sum(layout)]:
        values = row.split()
        if len(values) != 7:
            raise RuntimeError('invalid UUID participant row')
        node = int(values[0])
        if not 0 <= node < hosts or not re.fullmatch('[0-9a-f]{32}', values[6]) or (node, values[6]) in seen:
            raise RuntimeError('invalid/duplicate UUID participant')
        if node in names and names[node] != values[1]:
            raise RuntimeError('inconsistent host identity')
        names[node] = values[1]; population[node] += 1; seen.add((node, values[6]))
    if population != layout or len(set(names.values())) != hosts:
        raise RuntimeError('UUID host population mismatch')
    rank = record.get('rank')
    if type(rank) is not int or not 0 <= rank < sum(layout):
        raise RuntimeError('UUID launch requires --rank and --device')
    selected = lines[3 + rank].split()
    if selected != rank_descriptor(record, int(selected[0])):
        raise RuntimeError('stale UUID rank/view identity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('catalog', 'probe', 'candidate', 'launch'))
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--real-nccl', type=Path)
    parser.add_argument('--records', type=Path, nargs='+')
    parser.add_argument('--rank-records', type=Path, nargs='+', help='V4 records in logical rank order')
    parser.add_argument('--rank', type=int, help='logical rank for V4 probe/launch')
    parser.add_argument('--device', type=int, help='owned CUDA ordinal in this process view')
    parser.add_argument('--shapes', type=Path)
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'))
    parser.add_argument('--provider', help='explicit experiment provider; does not qualify performance')
    parser.add_argument('--rank-map', type=Path, help='JSON [host-index, CUDA-device] binding per logical rank')
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--output', type=Path)
    args, command = parser.parse_known_args()
    library = args.library.resolve(strict=True)
    backends = catalog(library)
    if args.mode == 'catalog':
        print(json.dumps(backends, indent=2))
        return
    if args.mode == 'candidate':
        if not (bool(args.records) != bool(args.rank_records) and args.shapes and args.dtype and args.profile):
            parser.error('candidate needs exactly one of --records/--rank-records, plus --shapes, --dtype and --profile')
        if args.rank_records:
            if args.rank_map:
                parser.error('UUID records already bind logical ranks; do not pass --rank-map')
            name = write_rank_candidate(args.profile, [json.loads(p.read_text()) for p in args.rank_records],
                                        json.loads(args.shapes.read_text()), args.dtype, backends, args.provider)
        else:
            name = write_candidate(args.profile, [json.loads(p.read_text()) for p in args.records],
                                   json.loads(args.shapes.read_text()), args.dtype, backends, args.provider,
                                   json.loads(args.rank_map.read_text()) if args.rank_map else None)
        print('Explicit candidate provider:', name)
        return
    r = record(args.real_nccl.resolve(strict=True) if args.real_nccl else None, library, args.rank, args.device)
    if args.mode == 'probe':
        text = json.dumps(r, indent=2) + '\n'
        if args.output:
            args.output.write_text(text)
        else:
            print(text, end='')
        return
    if not args.profile:
        parser.error('launch requires --profile')
    lines = args.profile.read_text().splitlines()
    start = 1
    uuid_profile = lines[0] == 'UCCL_LITE_RUNTIME_PROFILE_V4'
    if uuid_profile:
        verify_rank_profile(lines, r, backends)
        backend = True
    elif lines[0] in ('UCCL_LITE_RUNTIME_PROFILE_V1', 'UCCL_LITE_RUNTIME_PROFILE_V2', 'UCCL_LITE_RUNTIME_PROFILE_V3'):
        fields = lines[1].split()
        backend = next((b for b in backends if b['name'] == fields[0]), None)
        hosts = int(fields[1])
        layout = list(map(int, fields[2:]))
        if not backend or hosts != len(layout) or not supports_layout(backend, layout):
            raise RuntimeError('invalid runtime profile layout')
        start = 2
        if lines[0] == 'UCCL_LITE_RUNTIME_PROFILE_V3':
            if 3 not in backend.get('profile_versions', []):
                raise RuntimeError('compiled provider does not support rank-map profiles')
            fields = lines[2].split()
            if len(fields) != 2 + 2 * sum(layout) or fields[0] != 'ranks' or int(fields[1]) != sum(layout):
                raise RuntimeError('invalid runtime rank-map extent')
            values = list(map(int, fields[2:]))
            validate_rank_map(layout, list(zip(values[::2], values[1::2])))
            start = 3
    else:
        backend = next((b for b in backends if b['profile'] == lines[0]), None)
        hosts = len(backend['ranks_per_host']) if backend else 0
    if not uuid_profile and (not backend or descriptor(r) not in [line.split() for line in lines[start:start + hosts]]):
        raise RuntimeError('stale runtime profile identity')
    env = {k: v for k, v in os.environ.items() if not k.startswith('UCCL_LITE_MSCCLPP_')}
    if env.get('LD_PRELOAD'):
        parser.error('launch requires an unset LD_PRELOAD')
    replacement = library.parent / 'strict'
    if not (replacement / 'libnccl.so.2').is_file():
        raise RuntimeError('runtime loader alias is missing')
    env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT=os.environ.get('UCCL_LITE_MSCCLPP_CONTROL_ROOT', '127.0.0.1'),
               UCCL_LITE_MSCCLPP_PROFILE=str(args.profile.resolve()),
               UCCL_LITE_MSCCLPP_IDENTITY=r['identity'], UCCL_LITE_MSCCLPP_CANDIDATE='1',
               LD_PRELOAD=str(library), LD_LIBRARY_PATH=str(replacement) + ':' + env.get('LD_LIBRARY_PATH', ''))
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('launch requires a command after --')
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
