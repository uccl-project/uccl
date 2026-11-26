"""Explicit disjoint physical-core groups on each GPU's NUMA node.

The complete allowed SMT-sibling mask is used. Reserve the first two local
cores when capacity permits; never share a physical core between ranks.
This changes candidate placement only, not performance admission.
"""
import csv
import hashlib
import json
from pathlib import Path

import local_numa_profile as numa
import mscclpp_profile as base


def core_masks(details, nodes, numa_masks=None, cores_per_rank=1):
    if cores_per_rank not in (1, 2):
        raise ValueError('unsupported core-group width')
    rows = [list(map(int, row)) for row in csv.reader(details['cpus'].splitlines())
            if row and not row[0].startswith('#')]
    if any(len(row) != 4 or min(row) < 0 for row in rows):
        raise ValueError('unknown CPU core topology')
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError('duplicate CPU topology records')
    allowed = set(details['launch_affinity'])
    if numa_masks is None:
        numa_masks = [sorted(row[0] for row in rows if row[3] == node and row[0] in allowed)
                      for node in nodes]
    if len(nodes) != 4 or len(numa_masks) != 4:
        raise ValueError('core placement requires four rank records')
    result = [None] * 4
    for node in sorted(set(nodes)):
        ranks = [r for r, actual in enumerate(nodes) if actual == node]
        expected = {row[0] for row in rows if row[3] == node and row[0] in allowed}
        if node < 0 or not expected or any(set(numa_masks[r]) != expected for r in ranks):
            raise ValueError('inconsistent NUMA CPU topology')
        groups = {}
        for cpu, core, socket, cpu_node in rows:
            if cpu_node == node and cpu in expected:
                groups.setdefault((socket, core), []).append(cpu)
        ordered = sorted((sorted(mask) for mask in groups.values()), key=lambda mask: mask[0])
        required = len(ranks) * cores_per_rank
        if len(ordered) < required:
            raise ValueError('not enough distinct local CPU cores')
        offset = 2 if len(ordered) >= required + 2 else 0
        for index, rank in enumerate(ranks):
            first = offset + index * cores_per_rank
            result[rank] = sorted(cpu for mask in ordered[first:first + cores_per_rank] for cpu in mask)
    return result


def record(real, library, cores_per_rank=1, launcher=None):
    launcher = Path(launcher or __file__)
    r = base.record(real, library)
    nodes, available = numa.placement(r['details'])
    masks = core_masks(r['details'], nodes, available, cores_per_rank)
    r['details'].update(cpu_placement='numa-core', core_binding_version=1,
                        cores_per_rank=cores_per_rank,
                        gpu_numa_nodes=nodes, rank_cpu_affinity=masks,
                        numa_launcher_sha256=base.common.sha256(numa.__file__),
                        core_launcher_sha256=base.common.sha256(launcher),
                        core_helper_sha256=base.common.sha256(__file__))
    r['identity'] = hashlib.sha256(json.dumps(r['details'], sort_keys=True).encode()).hexdigest()
    return r


def make_record(real, library, output, cores_per_rank=1, launcher=None):
    launcher = Path(launcher or __file__)
    r = record(real, library, cores_per_rank, launcher)
    for name, source in [('core_launcher.py', launcher), ('core_helper.py', __file__),
                         ('numa_launcher.py', numa.__file__),
                         ('base_launcher.py', base.__file__)]:
        (output / name).write_bytes(Path(source).read_bytes())
    return r


def validate_run(directory, command, log, cores_per_rank=1, launcher=None):
    launcher = Path(launcher or __file__)
    if Path(command[command.index('-B') + 1]).resolve() != launcher.resolve():
        raise ValueError('core placement launcher changed')
    r = json.loads((directory / 'identity.json').read_text())
    d = r['details']
    digest = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    if (digest != r['identity'] or d.get('cpu_placement') != 'numa-core'
            or d.get('core_binding_version') != 1
            or d.get('cores_per_rank', 1) != cores_per_rank):
        raise ValueError('core placement identity mismatch')
    numa.validate_profile(r, directory / 'profile.tsv')
    for name, key in [('core_launcher.py', 'core_launcher_sha256'),
                      ('numa_launcher.py', 'numa_launcher_sha256'),
                      ('base_launcher.py', 'adapter_launcher_sha256')]:
        if base.common.sha256(directory / name) != d[key]:
            raise ValueError('core placement launcher snapshot mismatch')
    if ('core_helper_sha256' in d
            and base.common.sha256(directory / 'core_helper.py') != d['core_helper_sha256']):
        raise ValueError('core helper snapshot mismatch')
    expected = core_masks(d, d['gpu_numa_nodes'], cores_per_rank=cores_per_rank)
    if d['rank_cpu_affinity'] != expected:
        raise ValueError('core placement masks changed')
    for rank, mask in enumerate(expected):
        marker = f'UCCL Lite CPU placement: rank={rank} cpus=' + ','.join(map(str, mask)) + '\n'
        if log.count(marker) != 1:
            raise ValueError('missing core placement marker')


if __name__ == '__main__':
    # Reuse the established validation, binding and exec boundary unchanged.
    numa.record = record
    numa.main()
