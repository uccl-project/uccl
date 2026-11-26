"""Conservative four-L4 admission using actual nccl-tests host batch times."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import dropin_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('library', 'real-nccl', 'tests', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--mpirun', required=True)
    p.add_argument('--control-root')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    dropin_profile.LIBRARY = args.library.resolve(strict=True)
    lock = open('/tmp/uccl-lite-integration-wxz-l4.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    cases = {}
    identity = details = None
    for offset in (0, 1, 2, 4, 8):
        for repeat in range(2):
            target = args.out / f'offset{offset}_round{repeat}'
            command = [sys.executable, str(dropin_profile.HERE / '../../tests/nccl_dropin/local_profile.py'),
                '--library', str(args.library), '--real-nccl', str(args.real_nccl), '--tests', str(args.tests),
                '--mpirun', args.mpirun, '--out', str(target), '--unalign', str(offset)]
            if args.control_root:
                command += ['--control-root', args.control_root]
            subprocess.run(command, check=True)
            manifest = json.loads((target / 'manifest.json').read_text())
            if identity is not None and identity != manifest['identity']:
                raise RuntimeError('identity changed during calibration')
            identity, details = manifest['identity'], manifest['details']
            for row in json.loads((target / 'summary.json').read_text()):
                if not row['selected']:
                    continue
                combination = cases.setdefault((row['operation'], row['count']), {}).setdefault(
                    (offset, row['layout']), {'native_us': [], 'uccl_us': []})
                combination['native_us'] += [row['jobs_us'][0], row['jobs_us'][3]]
                combination['uccl_us'] += row['jobs_us'][1:3]
    summaries, admitted = [], []
    operations = {'all_gather': 1, 'reduce_scatter': 2, 'all_reduce': 3}
    for (op, count), combinations in sorted(cases.items()):
        if len(combinations) != 10:
            raise RuntimeError('missing alignment/layout combinations')
        rows = [{'offset_elements': offset, 'layout': layout, **values,
                 'minimum_speedup': min(values['native_us']) / max(values['uccl_us'])}
                for (offset, layout), values in combinations.items()]
        limiting = min(rows, key=lambda x: x['minimum_speedup'])
        accept = limiting['minimum_speedup'] >= 1.05
        summaries.append({'operation': op, 'count': count, 'combinations': rows,
                          'minimum_speedup': limiting['minimum_speedup'], 'admitted': accept})
        if accept:
            admitted.append([operations[op], count, min(limiting['native_us']) / 1000,
                             max(limiting['uccl_us']) / 1000, 4])
    capacity = max(count * (2 if op == 'all_reduce' else 8) for op, count in cases)
    dropin_profile.write_profile(args.out / 'profile.tsv', identity, details, args.real_nccl,
                                admitted, capacity=capacity)
    (args.out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    print('Local nccl-tests admitted', len(admitted), '/', len(summaries))


if __name__ == '__main__':
    main()
