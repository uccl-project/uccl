"""Calibrate exact two-host FP32 shapes using unchanged MPI nccl-tests."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import network_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hosts', required=True)
    p.add_argument('--real-nccl', type=Path, required=True)
    p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mpirun', required=True)
    p.add_argument('--iface', required=True)
    p.add_argument('--control-root')
    p.add_argument('--rounds', type=int, default=2)
    args = p.parse_args()
    if args.rounds < 2:
        p.error('at least two ABBA rounds are required (four independent jobs per backend/alignment)')
    args.out.mkdir(parents=True, exist_ok=False)
    cases = {}
    records = None
    lock = open("/tmp/uccl-lite-integration-wxz-l4.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    for alignment in (0, 1, 2, 4):
        for repeat in range(args.rounds):
            target = args.out / f'align{alignment}_round{repeat}'
            command = [sys.executable, str(network_profile.ROOT / 'tests/nccl_dropin/network.py'),
                       '--hosts', args.hosts, '--real-nccl', str(args.real_nccl), '--tests', str(args.tests),
                       '--out', str(target), '--mpirun', args.mpirun, '--iface', args.iface,
                       '--abba', '--unalign', str(alignment), '--warmup', '20', '--iterations', '100']
            if args.control_root:
                command += ['--control-root', args.control_root]
            subprocess.run(command, check=True)
            manifest = json.loads((target / 'manifest.json').read_text())
            descriptors = [network_profile.descriptor(r) for r in manifest['records']]
            if records is not None and descriptors != [network_profile.descriptor(r) for r in records]:
                raise RuntimeError('identity changed during calibration')
            records = manifest['records']
            for row in json.loads((target / 'summary.json').read_text()):
                if not row['candidate_selected']:
                    continue
                key = (row['operation'], row['count'])
                group = cases.setdefault(key, {}).setdefault((alignment, row['layout']), {'nccl': [], 'uccl': []})
                group['nccl'] += [row['jobs_us'][0], row['jobs_us'][3]]
                group['uccl'] += row['jobs_us'][1:3]
    summary, admitted = [], []
    operations = {'all_gather': 1, 'reduce_scatter': 2, 'all_reduce': 3}
    for (operation, count), combinations in sorted(cases.items()):
        if len(combinations) != 8:
            raise RuntimeError('missing layout/alignment combinations')
        rows = [{'unalign_elements': a, 'layout': layout, **values,
                 'minimum_speedup': min(values['nccl']) / max(values['uccl'])}
                for (a, layout), values in combinations.items()]
        limiting = min(rows, key=lambda row: row['minimum_speedup'])
        selected = limiting['minimum_speedup'] >= 1.05
        summary.append({'operation': operation, 'count': count, 'combinations': rows,
                        'minimum_speedup': limiting['minimum_speedup'], 'admitted': selected})
        if selected:
            admitted.append([operations[operation], count, min(limiting['nccl']) / 1000,
                             max(limiting['uccl']) / 1000, args.rounds * 2])
    capacity = max(count * (4 if operation == 'all_reduce' else 32) for operation, count in cases)
    network_profile.write_profile(args.out / 'profile.tsv', records, admitted, capacity=capacity)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Admitted', len(admitted), '/', len(summary), 'exact network shapes')


if __name__ == '__main__':
    main()
