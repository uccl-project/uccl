"""Explicit four-L4 NUMA-local launch placement; no automatic admission.

Reuse the shared strict profile format and hardware/software fingerprint.
The parent allowed CPU set, GPU NUMA nodes, exact rank masks and this launcher
are fingerprinted before binding; the selected mask is checked before exec.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import mscclpp_profile as base


def cpu_list(text):
    cpus = set()
    for item in text.strip().split(','):
        bounds = item.split('-')
        if len(bounds) == 1:
            first = last = int(bounds[0])
        elif len(bounds) == 2:
            first, last = map(int, bounds)
        else:
            raise ValueError('invalid CPU list')
        if first < 0 or last < first:
            raise ValueError('invalid CPU range')
        cpus.update(range(first, last + 1))
    return cpus


def placement(details, sysfs=Path('/sys')):
    allowed = set(details['launch_affinity'])
    nodes, masks = [], []
    for gpu in details['gpus'].splitlines():
        domain, bus, function = gpu.split(',')[2].strip().split(':')
        address = f'{int(domain, 16):04x}:{bus.lower()}:{function.lower()}'
        node = int((sysfs / 'bus/pci/devices' / address / 'numa_node').read_text())
        if node < 0:
            raise ValueError('GPU NUMA node is unknown')
        cpus = cpu_list((sysfs / 'devices/system/node' / f'node{node}' / 'cpulist').read_text())
        selected = sorted(allowed & cpus)
        if not selected:
            raise ValueError('GPU NUMA node has no allowed caller CPUs')
        nodes.append(node)
        masks.append(selected)
    if len(masks) != 4:
        raise ValueError('local placement requires four GPU records')
    return nodes, masks


def record(real, library):
    r = base.record(real, library)
    nodes, masks = placement(r['details'])
    r['details'].update(cpu_placement='numa-local', gpu_numa_nodes=nodes,
                        rank_cpu_affinity=masks,
                        numa_launcher_sha256=base.common.sha256(__file__))
    r['identity'] = hashlib.sha256(json.dumps(r['details'], sort_keys=True).encode()).hexdigest()
    return r


def launcher_path(mode):
    if mode not in ('unbound', 'numa'):
        raise ValueError('unsupported CPU placement')
    return Path(__file__ if mode == 'numa' else base.__file__).resolve()


def make_record(mode, real, library, output):
    launcher_path(mode)
    r = record(real, library) if mode == 'numa' else base.record(real, library)
    if mode == 'numa':
        # Preserve provenance without making historical audits depend on the
        # future contents of either launcher in the working tree.
        (output / 'numa_launcher.py').write_bytes(Path(__file__).read_bytes())
        (output / 'base_launcher.py').write_bytes(Path(base.__file__).read_bytes())
    return r


def validate_profile(r, path):
    lines = path.read_text().splitlines()
    if (len(lines) < 3 or lines[0] != 'UCCL_LITE_MSCCLPP_LOCAL_STRICT_V1'
            or lines[1].split() != base.descriptor(r)):
        raise ValueError('stale or non-local placement profile')


def validate_run(directory, mode, command, log):
    if Path(command[command.index('-B') + 1]).resolve() != launcher_path(mode):
        raise ValueError('placement launcher changed')
    if mode != 'numa':
        return
    r = json.loads((directory / 'identity.json').read_text())
    d = r['details']
    digest = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    if digest != r['identity'] or d.get('cpu_placement') != 'numa-local':
        raise ValueError('placement identity mismatch')
    validate_profile(r, directory / 'profile.tsv')
    if (base.common.sha256(directory / 'numa_launcher.py') != d['numa_launcher_sha256']
            or base.common.sha256(directory / 'base_launcher.py') != d['adapter_launcher_sha256']):
        raise ValueError('placement launcher snapshot mismatch')
    cpus = [row for row in csv.reader(d['cpus'].splitlines()) if row and not row[0].startswith('#')]
    allowed = set(d['launch_affinity'])
    if len(d['gpu_numa_nodes']) != 4 or len(d['rank_cpu_affinity']) != 4:
        raise ValueError('placement rank coverage')
    for rank, (node, mask) in enumerate(zip(d['gpu_numa_nodes'], d['rank_cpu_affinity'])):
        expected = sorted(int(row[0]) for row in cpus if int(row[3]) == node and int(row[0]) in allowed)
        marker = f'UCCL Lite CPU placement: rank={rank} cpus=' + ','.join(map(str, mask)) + '\n'
        if node < 0 or not expected or mask != expected or log.count(marker) != 1:
            raise ValueError('missing or inconsistent rank placement')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('probe', 'launch'))
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--real-nccl', type=Path, required=True)
    p.add_argument('--profile', type=Path)
    p.add_argument('--output', type=Path)
    args, command = p.parse_known_args()
    os.environ.setdefault('UCCL_LITE_MSCCLPP_CONTROL_ROOT', '127.0.0.1')
    real, library = args.real_nccl.resolve(strict=True), args.library.resolve(strict=True)
    r = record(real, library)
    if args.mode == 'probe':
        result = json.dumps(r, indent=2) + '\n'
        if args.output:
            args.output.write_text(result)
        else:
            print(result, end='', flush=True)
        return
    if not args.profile:
        p.error('launch requires --profile')
    validate_profile(r, args.profile)
    if os.environ.get('OMPI_COMM_WORLD_SIZE') != '4':
        p.error('NUMA-local launcher requires four Open MPI ranks on this host')
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    if not 0 <= rank < 4:
        p.error('invalid local rank')
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        p.error('launch requires a command after --')
    if os.environ.get('LD_PRELOAD'):
        p.error('launch requires an unset LD_PRELOAD')
    replacement = library.parent / 'strict'
    if not (replacement / 'libnccl.so.2').is_file():
        p.error('strict replacement directory is missing')
    selected = r['details']['rank_cpu_affinity'][rank]
    os.sched_setaffinity(0, selected)
    if sorted(os.sched_getaffinity(0)) != selected:
        raise RuntimeError('CPU placement could not be applied exactly')
    print(f'UCCL Lite CPU placement: rank={rank} cpus=' + ','.join(map(str, selected)), flush=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('UCCL_LITE_MSCCLPP_')}
    env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT=os.environ['UCCL_LITE_MSCCLPP_CONTROL_ROOT'],
               UCCL_LITE_MSCCLPP_PROFILE=str(args.profile.resolve()),
               UCCL_LITE_MSCCLPP_IDENTITY=r['identity'], UCCL_LITE_MSCCLPP_CANDIDATE='1',
               LD_PRELOAD=str(library),
               LD_LIBRARY_PATH=str(replacement) + ':' + env.get('LD_LIBRARY_PATH', ''))
    env.pop('UCCL_LITE_REAL_NCCL', None)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
