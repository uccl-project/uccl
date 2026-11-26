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


def record(real, library):
    common.LIBRARY = library
    _, details = common.identity(real)
    dependencies = subprocess.check_output(['ldd', str(library)], text=True)
    match = re.search(r'libmscclpp\.so[^ ]* => (\S+)', dependencies)
    if not match:
        raise RuntimeError('MSCCL++ shared dependency is unresolved')
    runtime = re.search(r'libcudart\.so[^ ]* => (\S+)', dependencies)
    if not runtime:
        raise RuntimeError('CUDA runtime shared dependency is unresolved')
    # A host may install CUDA 12 and 13 side by side. Probe the actual DT_NEEDED
    # runtime of the adapter, not the toolkit's unversioned libcudart symlink.
    versions = []
    cuda = ctypes.CDLL(runtime[1])
    for lib, name in [(cuda, 'cudaDriverGetVersion'), (cuda, 'cudaRuntimeGetVersion'),
                      (ctypes.CDLL(str(real)), 'ncclGetVersion')]:
        value = ctypes.c_int()
        if getattr(lib, name)(ctypes.byref(value)):
            raise RuntimeError(name + ' failed')
        versions.append(value.value)
    details['cudart_sha256'] = common.sha256(runtime[1])
    details.update(mscclpp_sha256=common.sha256(match[1]),
                   adapter_launcher_sha256=common.sha256(__file__),
                   network_environment={k: v for k, v in os.environ.items()
                                        if k.startswith('MSCCLPP_')})
    lines = details['gpus'].splitlines()
    if len(lines) != 4 or any('NVIDIA L4' not in line for line in lines):
        raise RuntimeError('candidate requires four NVIDIA L4s per host')
    return dict(hostname=platform.node(), details=details,
                identity=hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest(),
                versions=versions,
                uuids=[line.split(',')[0].strip().removeprefix('GPU-').replace('-', '')
                       for line in lines])


def descriptor(r):
    return [r['hostname'], r['identity'], *map(str, r['versions']), *r['uuids']]


def write_candidate(path, records, shapes):
    if len(records) != 2 or records[0]['hostname'] == records[1]['hostname']:
        raise ValueError('two ordered, distinct hosts are required')
    rows = []
    for op, count in shapes:
        scale = 4 if op == 3 else 32
        if op not in (1, 2, 3) or count <= 0 or count * scale > 256 * 1024 or count * scale % 128:
            raise ValueError('unsupported FP32 candidate shape')
        rows.append(f'{op} {count}')
    if not rows or len(set(rows)) != len(rows):
        raise ValueError('candidate requires unique exact shapes')
    text = 'UCCL_LITE_MSCCLPP_CANDIDATE_V1\n'
    text += '\n'.join(' '.join(descriptor(r)) for r in records) + '\n'
    Path(path).write_text(text + '\n'.join(rows) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['probe', 'candidate', 'launch'])
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--real-nccl', type=Path, required=True)
    p.add_argument('--profile', type=Path)
    p.add_argument('--records', type=Path, nargs=2)
    p.add_argument('--shapes', type=Path, help='JSON list of [operation, NCCL count]')
    p.add_argument('--output', type=Path)
    args, command = p.parse_known_args()
    real = args.real_nccl.resolve(strict=True)
    library = args.library.resolve(strict=True)
    if args.mode == 'candidate':
        if not (args.profile and args.records and args.shapes):
            p.error('candidate requires --profile, --records and --shapes')
        write_candidate(args.profile, [json.loads(x.read_text()) for x in args.records],
                        json.loads(args.shapes.read_text()))
        return
    if args.mode == 'probe':
        text = json.dumps(record(real, library), indent=2)
        if args.output:
            args.output.write_text(text + '\n')
        else:
            print(text, flush=True)
        return
    if not args.profile:
        p.error('launch requires --profile (candidate experiments only)')
    identity = 'invalid-profile'
    try:
        r = record(real, library)
        lines = args.profile.read_text().splitlines()
        if lines[0] == 'UCCL_LITE_MSCCLPP_CANDIDATE_V1' and descriptor(r) in [x.split() for x in lines[1:3]]:
            identity = r['identity']
        else:
            print('UCCL Lite: stale candidate identity, using NCCL', file=sys.stderr)
    except (OSError, ValueError, IndexError, RuntimeError, subprocess.SubprocessError) as error:
        print('UCCL Lite: unavailable candidate, using NCCL:', error, file=sys.stderr)
    env = {k: v for k, v in os.environ.items() if not k.startswith('UCCL_LITE_MSCCLPP_')}
    env.update(UCCL_LITE_REAL_NCCL=str(real),
               UCCL_LITE_MSCCLPP_PROFILE=str(args.profile.resolve()),
               UCCL_LITE_MSCCLPP_IDENTITY=identity,
               UCCL_LITE_MSCCLPP_CANDIDATE='1')
    # Avoid chaining different collective interposers on the same communicator.
    if env.get('LD_PRELOAD'):
        p.error('launch requires an unset LD_PRELOAD')
    env['LD_PRELOAD'] = f'{library}:{real}'
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        p.error('launch requires a command after --')
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
