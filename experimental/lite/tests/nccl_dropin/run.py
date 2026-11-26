#!/usr/bin/env python3
"""Run unchanged nccl-tests against real NCCL and the interposer (ABBA)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_output(log, ranks):
    # Both layouts have separate time/algbw/busbw/error columns. Reject logs
    # without actual result rows or the final correctness summary.
    if not re.search(r'# Out of bounds values\s*:\s*0\s+OK', log):
        raise RuntimeError('missing successful nccl-tests correctness summary')
    observed = {int(rank) for rank in re.findall(r'#\s+Rank\s+(\d+)\s+Group', log)}
    if observed != set(range(ranks)):
        raise RuntimeError(f'expected {ranks} collective ranks, observed {observed}')
    rows = {}
    for line in log.splitlines():
        fields = line.split()
        if len(fields) != 13 or not fields[0].isdigit():
            continue
        if fields[8] != '0' or fields[12] != '0':
            raise RuntimeError('nccl-tests reported incorrect output: ' + line)
        rows[fields[0]] = [float(fields[5]), float(fields[9])]
    if not rows:
        raise RuntimeError('no validated in-place/out-of-place timing rows')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--real-nccl', type=Path, required=True)
    p.add_argument('--library', type=Path, default=Path('build/nccl_lite/libuccl_lite_nccl.so'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hosts', help='two hosts with four slots each, e.g. node0:4,node1:4')
    p.add_argument('--mpirun', default='mpirun')
    p.add_argument('--mpi-args', default='', help='additional shlex-parsed MPI options')
    p.add_argument('--mpi', action='store_true', help='one process per GPU, also on one host')
    p.add_argument('--suffix', default='_perf', help='use _perf_mpi for separately named MPI builds')
    p.add_argument('--operations', nargs='+', default=['all_reduce', 'all_gather', 'reduce_scatter', 'broadcast', 'reduce'])
    p.add_argument('--dtypes', nargs='+', default=['bfloat16', 'float'])
    p.add_argument('--min-bytes', default='8')
    p.add_argument('--max-bytes', default='64M')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--graph', type=int, default=0, help='nccl-tests graph replay count')
    p.add_argument('--timeout', type=int, default=240)
    args = p.parse_args()
    args.library = args.library.resolve(strict=True)
    args.real_nccl = args.real_nccl.resolve(strict=True)
    args.tests = args.tests.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    # This harness establishes the fallback baseline. Never inherit an unrelated
    # host's calibration profile or inject it into the native NCCL control.
    for key in list(env):
        if key.startswith('UCCL_LITE_DROPIN_'):
            del env[key]
    env['UCCL_LITE_REAL_NCCL'] = str(args.real_nccl)
    env['LD_LIBRARY_PATH'] = str(args.real_nccl.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['LD_PRELOAD'] = str(args.real_nccl)
    mpi = args.mpi or bool(args.hosts)
    launcher = []
    if mpi:
        launcher = [args.mpirun, '-np', '8' if args.hosts else '4', '--bind-to', 'none']
        if args.hosts:
            launcher += ['--host', args.hosts, '--map-by', 'ppr:4:node']
        launcher += shlex.split(args.mpi_args)
        for key in sorted(env):
            if key.startswith(('NCCL_', 'CUDA_', 'UCCL_LITE_')) or key in ('LD_PRELOAD', 'LD_LIBRARY_PATH'):
                launcher += ['-x', key]
    manifest = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'library_sha256': digest(args.library), 'nccl_sha256': digest(args.real_nccl),
                'environment': {k: v for k, v in env.items() if k.startswith(('NCCL_', 'CUDA_', 'UCCL_LITE_'))},
                'timing': 'nccl-tests host enqueue+completion batch mean, default rank average, microseconds, both layouts',
                'selection': 'all rows retained; no best variant selection',
                'scope': 'NCCL fallback only; no calibrated acceleration profile',
                'order': ['nccl', 'dropin', 'dropin', 'nccl']}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for label, probe in [('gpus', ['nvidia-smi', '-L']),
                         ('topology', ['nvidia-smi', 'topo', '-m'])]:
        command = launcher + probe
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=args.timeout)
        (args.out / (label + '.txt')).write_text(result.stdout)
        result.check_returncode()
    results = {}
    for op in args.operations:
        binary = args.tests / (op + args.suffix)
        if not re.fullmatch(r'[a-z_]+', op) or not binary.is_file():
            raise RuntimeError('missing nccl-tests binary: ' + str(binary))
        if mpi:
            dependencies = subprocess.check_output(['ldd', str(binary)], env=env, text=True)
            if not re.search(r'libmpi\.so', dependencies):
                raise RuntimeError('MPI mode requires an MPI-enabled nccl-tests build: ' + str(binary))
        for dtype in args.dtypes:
            runs = []
            for i, backend in enumerate(manifest['order']):
                env['LD_PRELOAD'] = (str(args.library) + ':' if backend == 'dropin' else '') + str(args.real_nccl)
                command = launcher + [str(binary), '-b', args.min_bytes, '-e', args.max_bytes,
                    '-f', '2', '-g', '1' if mpi else '4', '-d', dtype,
                    '-w', str(args.warmup), '-n', str(args.iterations), '-c', '1', '-G', str(args.graph)]
                prefix = args.out / f'{op}_{dtype}_{i}_{backend}'
                prefix.with_suffix('.command.json').write_text(json.dumps(command) + '\n')
                print(shlex.join(command), backend, flush=True)
                with prefix.with_suffix('.log').open('w') as log:
                    r = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                r.check_returncode()
                runs.append(checked_output(prefix.with_suffix('.log').read_text(), 8 if args.hosts else 4))
            if any(set(run) != set(runs[0]) for run in runs):
                raise RuntimeError('different sizes across paired jobs')
            results[f'{op}/{dtype}'] = {size: {
                'nccl_us': [statistics.median([runs[0][size][j], runs[3][size][j]]) for j in range(2)],
                'dropin_us': [statistics.median([runs[1][size][j], runs[2][size][j]]) for j in range(2)],
                'dropin_over_nccl': [statistics.median([runs[1][size][j], runs[2][size][j]]) /
                                    statistics.median([runs[0][size][j], runs[3][size][j]]) for j in range(2)]
            } for size in runs[0]}
            (args.out / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
