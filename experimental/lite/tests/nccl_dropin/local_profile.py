"""Local four-L4 profile check against unchanged nccl-tests batch timing."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from run import checked_output
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
import dropin_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--real-nccl', type=Path, required=True)
    p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--mpirun', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--profile', type=Path)
    p.add_argument('--cases', type=Path, help='candidate [operation,count] pairs; not deployment admission')
    p.add_argument('--capacity', type=int, help='reserve candidate bytes while screening a subset')
    p.add_argument('--control-root', help='common rank-0 IPv4 address for CPU metadata')
    p.add_argument('--single-process', action='store_true')
    p.add_argument('--graph', type=int, default=0)
    p.add_argument('--unalign', type=int, default=0)
    p.add_argument('--dtype', default='bfloat16')
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--max-bytes', default='65536')
    args = p.parse_args()
    if args.profile and (args.cases or args.capacity is not None):
        p.error('--cases/--capacity apply only to calibration candidates')
    if args.control_root:
        os.environ['UCCL_LITE_DROPIN_CONTROL_ROOT'] = args.control_root
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    dropin_profile.LIBRARY = args.library.resolve(strict=True)
    real = args.real_nccl.resolve(strict=True)
    ident, details = dropin_profile.identity(real)
    if len(details['gpus'].splitlines()) != 4:
        p.error('requires four local GPUs')
    profile = args.out / ('profile.tsv' if args.profile else 'candidate.tsv')
    if args.profile:
        profile.write_text(args.profile.read_text())
    else:
        cases = json.loads(args.cases.read_text()) if args.cases else [[op, count] for op in (1, 2, 3) for count in (32, 256, 4096)]
        dropin_profile.write_profile(profile, ident, details, real,
            [[op, count, 1, 1, 3] for op, count in cases], candidate=True, capacity=args.capacity)
    selected = {(int(f[0]), int(f[1])) for line in profile.read_text().splitlines()
                if len(f := line.split()) == 5 and f[0] in ('1', '2', '3')}
    env = dropin_profile.launch_environment(real, profile, calibrating=not args.profile)
    env['LD_LIBRARY_PATH'] = str(real.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    export = [x for key in env if key.startswith(('NCCL_', 'CUDA_', 'UCCL_LITE_')) or key in ('LD_PRELOAD', 'LD_LIBRARY_PATH') for x in ('-x', key)]
    launcher = [] if args.single_process else [args.mpirun, '-np', '4', '--bind-to', 'none', *export]
    manifest = {'identity': ident, 'details': details, 'commands': [],
                'scope': 'deployment validation' if args.profile else 'candidate correctness/screen only',
                'timing': 'max-rank host enqueue/completion batch mean; 20 warmups; ABBA jobs; both layouts',
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    summary = []
    for op, slot in [('all_reduce', 3), ('all_gather', 1), ('reduce_scatter', 2)]:
        binary = args.tests.resolve() / (op + '_perf')
        if not args.single_process:
            dependencies = subprocess.check_output(['ldd', str(binary)], env=env, text=True)
            if 'libmpi.so' not in dependencies:
                raise RuntimeError('requires MPI-enabled nccl-tests')
        jobs = []
        for index, backend in enumerate(('nccl', 'uccl', 'uccl', 'nccl')):
            env['LD_PRELOAD'] = (str(dropin_profile.LIBRARY) + ':' if backend == 'uccl' else '') + str(real)
            command = launcher + [str(binary), '-b', '8', '-e', args.max_bytes, '-f', '2',
                '-g', '4' if args.single_process else '1', '-d', args.dtype, '-w', '20',
                '-n', str(args.iterations), '-c', '1', '-a', '3', '-G', str(args.graph), '-u', str(args.unalign)]
            log = args.out / f'{op}_{index}_{backend}.log'
            print(op, index, backend, flush=True)
            with log.open('w') as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=240)
            text = log.read_text()
            jobs.append(checked_output(text, 4))
            if backend == 'uccl':
                counters = re.findall(r'rank=(\d+) AG=(\d+) RS=(\d+) AR=(\d+)', text)
                # All sizes receive eager warmups before the first capture.
                # The first graph size is below every admitted candidate, so
                # only those warmups may have used our route in graph mode.
                observed = {int(size) // (2 if args.dtype == 'bfloat16' else 4) // (1 if slot == 3 else 4) for size in jobs[-1]}
                admitted = {n for o, n in selected if o == slot and n in observed}
                expected = not args.single_process and args.dtype == 'bfloat16' and bool(admitted)
                if expected and {int(r[0]) for r in counters if int(r[slot]) > 0} != set(range(4)):
                    raise RuntimeError('missing selected-route counts on all local ranks')
                if not expected and counters:
                    raise RuntimeError('fallback-only communicator prepared a route')
                if expected and args.graph:
                    first = min((n for n in observed if n), default=0)
                    eager = 20 * len(admitted) + int(first in admitted)
                    if len(counters) != 4 or any(int(r[slot]) != eager for r in counters):
                        raise RuntimeError('graph route counts differ from eager pre-capture calls')
            manifest['commands'].append(command)
            (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if any(set(j) != set(jobs[0]) for j in jobs):
            raise RuntimeError('paired size rows differ')
        for size in jobs[0]:
            count = int(size) // (2 if args.dtype == 'bfloat16' else 4) // (1 if slot == 3 else 4)
            for layout in range(2):
                times = [job[size][layout] for job in jobs]
                summary.append({'operation': op, 'bytes': int(size), 'count': count,
                    'layout': ('out', 'inplace')[layout], 'jobs_us': times,
                    'selected': args.dtype == 'bfloat16' and (slot, count) in selected and not args.graph and not args.single_process,
                    'worst_uccl_over_best_nccl': max(times[1:3]) / min(times[0], times[3])})
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.profile:
        failures = sum(row['worst_uccl_over_best_nccl'] > 1.05 for row in summary)
        print('Local deployment mixed-curve 5% failures:', failures, '/', len(summary))
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
