"""Identity-checked, explicit candidate launcher for the MSCCL++ adapter.

No deployment profiles are issued here: fresh nccl-tests measurements are
required before admitting a backend into an automatic routing policy.
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


def local_packet_environment():
    env = common.packet_environment()
    # The strict local target reserves these CPUs. Yielding while waiting for
    # GPU flags causes intermittent scheduler stalls in sparse per-call use.
    env['UCCL_LITE_MP_PACKET_SPIN'] = '1'
    return env


def record(real, library, strict=True, local8=False):
    if not strict:
        raise ValueError("native NCCL fallback mode has been removed")
    common.LIBRARY = library
    _, details = common.identity(real)
    dependencies = subprocess.check_output(['ldd', str(library)], text=True)
    if re.search(r'libnccl\.so', dependencies) or 'not found' in dependencies:
        raise RuntimeError('strict replacement has a native NCCL or unresolved dependency')
    match = re.search(r'libmscclpp\.so[^ ]* => (\S+)', dependencies)
    if not match and not local8:
        raise RuntimeError('MSCCL++ shared dependency is unresolved')
    runtime = re.search(r'libcudart\.so[^ ]* => (\S+)', dependencies)
    if not runtime:
        raise RuntimeError('CUDA runtime shared dependency is unresolved')
    # A host may install CUDA 12 and 13 side by side. Probe the actual DT_NEEDED
    # runtime of the adapter, not the toolkit's unversioned libcudart symlink.
    versions = []
    cuda = ctypes.CDLL(runtime[1])
    for lib, name in [(cuda, 'cudaDriverGetVersion'), (cuda, 'cudaRuntimeGetVersion'),
                      (ctypes.CDLL(str(library if strict else real)), 'ncclGetVersion')]:
        value = ctypes.c_int()
        if getattr(lib, name)(ctypes.byref(value)):
            raise RuntimeError(name + ' failed')
        versions.append(value.value)
    details['cudart_sha256'] = common.sha256(runtime[1])
    details.update(mscclpp_sha256=common.sha256(match[1]) if match else None,
                   adapter_launcher_sha256=common.sha256(__file__),
                   control_root=os.environ.get('UCCL_LITE_MSCCLPP_CONTROL_ROOT', ''),
                   network_environment={k: v for k, v in os.environ.items()
                                        if k.startswith('MSCCLPP_')})
    lines = details['gpus'].splitlines()
    if len(lines) != (8 if local8 else 4) or any('NVIDIA L4' not in line for line in lines):
        raise RuntimeError('candidate has the wrong NVIDIA L4 topology')
    if local8:
        details['packet'] = local_packet_environment()
        details['local_transport_environment'] = {k: v for k, v in os.environ.items()
                                                  if k.startswith('UCCL_LITE_MP_')}
        if details['local_transport_environment'] != local_packet_environment():
            raise RuntimeError('local8 requires the fixed packet environment')
    return dict(hostname=platform.node(), details=details,
                identity=hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest(),
                versions=versions,
                uuids=[line.split(',')[0].strip().removeprefix('GPU-').replace('-', '')
                       for line in lines])


def descriptor(r):
    return [r['hostname'], r['identity'], *map(str, r['versions']), *r['uuids']]


def write_candidate(path, records, shapes, strict=True, local8=False):
    if not strict:
        raise ValueError("native NCCL fallback mode has been removed")
    if len(records) != (1 if local8 else 2) or (not local8 and records[0]['hostname'] == records[1]['hostname']):
        raise ValueError('one eight-L4 host is required' if local8 else
                         'two ordered, distinct hosts are required')
    rows = []
    for op, count in shapes:
        scale = (2 if local8 else 4) * (1 if op == 3 else 8)
        if op not in (1, 2, 3) or count <= 0 or count * scale > ((1 << 30) if strict else (512 << 10)) or count * scale % 128:
            raise ValueError('unsupported candidate shape')
        rows.append(f'{op} {count}')
    if not rows or len(set(rows)) != len(rows):
        raise ValueError('candidate requires unique exact shapes')
    text = ('UCCL_LITE_LOCAL8_STRICT_V1' if local8 else 'UCCL_LITE_MSCCLPP_STRICT_V1') + '\n'
    text += '\n'.join(' '.join(descriptor(r)) for r in records) + '\n'
    Path(path).write_text(text + '\n'.join(rows) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['probe', 'candidate', 'launch'])
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--real-nccl', type=Path, required=True)
    p.add_argument('--profile', type=Path)
    p.add_argument('--records', type=Path, nargs='+')
    p.add_argument('--local8', action='store_true', help='single host, eight L4s, BF16 packet transport')
    p.add_argument('--shapes', type=Path, help='JSON list of [operation, NCCL count]')
    p.add_argument('--output', type=Path)
    p.add_argument('--strict', action='store_true', default=True, help='use the NCCL-free compatibility library')
    args, command = p.parse_known_args()
    if args.local8:
        # Configuration is fixed and is part of every rank's identity.
        os.environ.update(local_packet_environment())
    real = args.real_nccl.resolve(strict=True)
    library = args.library.resolve(strict=True)
    if args.mode == 'candidate':
        if not (args.profile and args.records and args.shapes):
            p.error('candidate requires --profile, --records and --shapes')
        write_candidate(args.profile, [json.loads(x.read_text()) for x in args.records],
                        json.loads(args.shapes.read_text()), strict=args.strict, local8=args.local8)
        return
    if args.mode == 'probe':
        text = json.dumps(record(real, library, strict=args.strict, local8=args.local8), indent=2)
        if args.output:
            args.output.write_text(text + '\n')
        else:
            print(text, flush=True)
        return
    if not args.profile:
        p.error('launch requires --profile (candidate experiments only)')
    identity = 'invalid-profile'
    try:
        r = record(real, library, strict=args.strict, local8=args.local8)
        lines = args.profile.read_text().splitlines()
        if lines[0] == ('UCCL_LITE_LOCAL8_STRICT_V1' if args.local8 else 'UCCL_LITE_MSCCLPP_STRICT_V1') and descriptor(r) in [x.split() for x in lines[1:2 if args.local8 else 3]]:
            identity = r['identity']
        else:
            print('UCCL Lite: stale identity; strict initialization will reject', file=sys.stderr)
    except (OSError, ValueError, IndexError, RuntimeError, subprocess.SubprocessError) as error:
        print('UCCL Lite: unavailable profile; strict initialization will reject:', error, file=sys.stderr)
    env = {k: v for k, v in os.environ.items() if not k.startswith('UCCL_LITE_MSCCLPP_')}
    env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT=os.environ.get('UCCL_LITE_MSCCLPP_CONTROL_ROOT', ''),
               UCCL_LITE_REAL_NCCL=str(real),
               UCCL_LITE_MSCCLPP_PROFILE=str(args.profile.resolve()),
               UCCL_LITE_MSCCLPP_IDENTITY=identity,
               UCCL_LITE_MSCCLPP_CANDIDATE='1')
    # Avoid chaining different collective interposers on the same communicator.
    if env.get('LD_PRELOAD'):
        p.error('launch requires an unset LD_PRELOAD')
    if args.strict:
        replacement = library.parent / 'strict'
        if not (replacement / 'libnccl.so.2').is_file():
            p.error('strict library directory is missing libnccl.so.2')
        env['LD_LIBRARY_PATH'] = str(replacement) + ':' + env.get('LD_LIBRARY_PATH', '')
        env['LD_PRELOAD'] = str(library)
        env.pop('UCCL_LITE_REAL_NCCL', None)
    else:
        env['LD_PRELOAD'] = f'{library}:{real}'
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        p.error('launch requires a command after --')
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
