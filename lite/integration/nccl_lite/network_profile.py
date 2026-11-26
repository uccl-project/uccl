"""Identity-checked two-host profile launcher for the network NCCL interposer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import dropin_profile as local

ROOT = Path(__file__).resolve().parents[2]
local.LIBRARY = ROOT / 'build/nccl_lite/libuccl_lite_nccl_multinode.so'
NETWORK = ROOT / 'build/libuccl_lite_multinode.so'


def record(real):
    ident, details = local.identity(real)
    dependencies = subprocess.check_output(['ldd', str(NETWORK)], text=True)
    match = re.search(r'libmscclpp\.so[^ ]* => (\S+)', dependencies)
    if not match:
        raise RuntimeError('cannot resolve MSCCL++ dependency')
    details.update(network_sha256=local.sha256(NETWORK),
                   mscclpp_sha256=local.sha256(match.group(1)),
                   network_launcher_sha256=local.sha256(__file__),
                   network_environment={k: v for k, v in os.environ.items()
                                        if k.startswith('MSCCLPP_')})
    lines = details['gpus'].splitlines()
    if len(lines) != 4 or any('NVIDIA L4' not in line for line in lines):
        raise RuntimeError('network calibration requires four L4 GPUs per host')
    ident = hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
    return {'hostname': platform.node(), 'identity': ident, 'details': details,
            'probe_rank': int(os.environ.get('OMPI_COMM_WORLD_RANK', '-1')),
            'versions': local.versions(real),
            'uuids': [line.split(',')[0].strip().removeprefix('GPU-').replace('-', '') for line in lines]}


def descriptor(r):
    return [r['hostname'], r['identity'], *r['versions'], *r['uuids']]


def write_profile(path, records, rows, candidate=False, capacity=None):
    if len(records) != 2 or records[0]['hostname'] == records[1]['hostname']:
        raise ValueError('requires two distinct ordered hosts')
    descriptions = [descriptor(r) for r in records]
    digest = hashlib.sha256(json.dumps(descriptions).encode()).hexdigest()
    magic = 'UCCL_LITE_NETWORK_CANDIDATE_V1' if candidate else 'UCCL_LITE_NETWORK_V1'
    text = magic + ' ' + digest + '\n'
    for r in records:
        text += ' '.join(map(str, descriptor(r)[:5])) + '\n'
        text += '\n'.join(r['uuids']) + '\n'
    rows = list(rows)
    if capacity is None:
        capacity = max((int(row[1]) * (4 if int(row[0]) == 3 else 32) for row in rows), default=128)
    text += f'capacity {capacity}\n'
    text += ''.join(' '.join(map(str, row)) + '\n' for row in rows)
    Path(path).write_text(text)


def environment(real, profile, calibrating=False):
    env = local.launch_environment(real)
    expected = 'UCCL_LITE_NETWORK_CANDIDATE_V1' if calibrating else 'UCCL_LITE_NETWORK_V1'
    identity = 'invalid-profile'
    try:
        tokens = Path(profile).read_text().split()
        if len(tokens) < 20 or tokens[0] != expected:
            raise ValueError('profile mode/header mismatch')
        records = [tokens[2:11], tokens[11:20]]
        descriptions = [[r[0], r[1], *map(int, r[2:5]), *r[5:]] for r in records]
        digest = hashlib.sha256(json.dumps(descriptions).encode()).hexdigest()
        current = record(real)
        if tokens[1] == digest:
            identity = current['identity']
        if tokens[1] != digest or descriptor(current) not in descriptions:
            print('UCCL Lite network: stale identity; voting for NCCL fallback', file=sys.stderr)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print('UCCL Lite network: profile unavailable; voting for NCCL fallback:', error, file=sys.stderr)
    # Keep every rank in the native initialization vote, including a host
    # whose profile is missing or stale. No network resources are prepared.
    env['UCCL_LITE_DROPIN_PROFILE'] = str(Path(profile).resolve())
    env['UCCL_LITE_DROPIN_IDENTITY'] = identity
    if calibrating:
        env['UCCL_LITE_DROPIN_CALIBRATING'] = '1'
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['probe', 'launch'])
    parser.add_argument('--real-nccl', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--calibrating', action='store_true')
    args, command = parser.parse_known_args()
    if args.mode == 'probe':
        print(json.dumps(record(args.real_nccl)), flush=True)
        return
    if not args.profile:
        parser.error('--profile is required')
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('a command is required')
    env = environment(args.real_nccl, args.profile, args.calibrating)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
