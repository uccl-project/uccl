#!/usr/bin/env python3
"""Run unchanged MPI nccl-tests on two four-L4 hosts, with route assertions.

This is a candidate evaluation, not automatic deployment admission. Artifacts
are staged into a new build directory at the same path on both hosts.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
import mscclpp_profile


def checked(log, operation, routed):
    if not re.search(r'# Out of bounds values\s*:\s*0\s+OK', log):
        raise RuntimeError('missing nccl-tests correctness summary')
    ranks = {int(x) for x in re.findall(r'#\s+Rank\s+(\d+)\s+Group', log)}
    if ranks != set(range(8)):
        raise RuntimeError('expected eight nccl-tests ranks')
    counters = re.findall(r'UCCL Lite MSCCL\+\+: rank=(\d+) AG=(\d+) RS=(\d+) AR=(\d+) fallback=(\d+)', log)
    if routed and (len(counters) != 8 or {int(x[0]) for x in counters} != set(range(8)) or
                   any(int(x[operation]) == 0 for x in counters)):
        raise RuntimeError('test did not route UCCL on every rank')
    if routed is False and any(sum(map(int, x[1:4])) for x in counters):
        raise RuntimeError('fallback test unexpectedly routed UCCL')
    rows = {}
    for line in log.splitlines():
        fields = line.split()
        if len(fields) == 13 and fields[0].isdigit():
            if fields[8] != '0' or fields[12] != '0':
                raise RuntimeError('incorrect nccl-tests output')
            rows[int(fields[0])] = [float(fields[5]), float(fields[9])]
    if not rows:
        raise RuntimeError('no measured rows')
    return {'rows_us': rows, 'route_counters': counters}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hosts', required=True, help='ordered node0,node1 SSH/MPI hosts')
    p.add_argument('--iface', required=True)
    p.add_argument('--hcas', default='mlx5_0,mlx5_1')
    p.add_argument('--mpirun', required=True)
    p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--real-nccl', type=Path, required=True)
    p.add_argument('--library', type=Path, default=ROOT / 'build/nccl_lite/libuccl_lite_nccl_mscclpp.so')
    p.add_argument('--abi', type=Path, help='optional NCCL-only ABI stress executable')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--max-bytes', type=int, default=1 << 20,
                   help='power-of-two complete extent, up to 1 GiB')
    p.add_argument('--shapes', type=Path, help='fixed JSON [operation, count] candidate rows')
    p.add_argument('--rounds', type=int, default=1, help='independent ABBA rounds')
    p.add_argument('--performance-only', action='store_true', help='omit additional ABI/graph/misalignment jobs; retain nccl-tests correctness checking in every sweep')
    p.add_argument('--max-regression', type=float, help='fail if any median candidate/native ratio exceeds 1 + this fraction')
    args = p.parse_args()
    if args.max_bytes < 1024 or args.max_bytes > 1 << 30 or args.max_bytes & (args.max_bytes - 1) or args.rounds < 1:
        p.error('max-bytes must be a power of two in 1 KiB–1 GiB; rounds must be positive')
    if args.max_regression is not None and (not math.isfinite(args.max_regression) or args.max_regression < 0):
        p.error('max-regression must be finite and nonnegative')
    hosts = args.hosts.split(',')
    if len(hosts) != 2 or hosts[0] == hosts[1]:
        p.error('two distinct ordered hosts required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'validation_driver.py')
    real = args.real_nccl.resolve(strict=True)
    library = out / args.library.name
    shutil.copy2(args.library, library)
    for name in ['mscclpp_profile.py', 'dropin_profile.py']:
        shutil.copy2(ROOT / 'integration/nccl_lite' / name, out / name)
    operations = [('all_gather', 1), ('reduce_scatter', 2), ('all_reduce', 3)]
    for op, _ in operations:
        shutil.copy2(args.tests / (op + '_perf'), out / (op + '_perf'))
    if args.abi:
        shutil.copy2(args.abi, out / 'abi_check')
    subprocess.run(['ssh', hosts[1], 'mkdir -p ' + shlex.quote(str(out))], check=True)
    subprocess.run(['rsync', '-a', str(out) + '/', f'{hosts[1]}:{out}/'], check=True)
    env = os.environ.copy()
    if env.get('LD_PRELOAD'):
        p.error('unset LD_PRELOAD before validation')
    env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT=socket.gethostbyname(hosts[0]),
               NCCL_SOCKET_IFNAME=args.iface, NCCL_IB_HCA=args.hcas,
               MSCCLPP_SOCKET_IFNAME=args.iface, MSCCLPP_IBV_MODE='host-no-atomic',
               LD_LIBRARY_PATH=str(real.parent) + ':' + env.get('LD_LIBRARY_PATH', ''))
    exported = ['UCCL_LITE_MSCCLPP_CONTROL_ROOT', 'NCCL_SOCKET_IFNAME', 'NCCL_IB_HCA', 'MSCCLPP_SOCKET_IFNAME',
                'MSCCLPP_IBV_MODE', 'LD_LIBRARY_PATH']
    def mpi(hosts, n, mapping):
        command = [args.mpirun, '--host', hosts, '-np', str(n), '--map-by', mapping,
                   '--bind-to', 'none', '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp',
                   '--mca', 'btl_tcp_if_include', args.iface,
                   '--mca', 'oob_tcp_if_include', args.iface]
        for key in exported:
            command += ['-x', key]
        return command
    launcher = ['python3', '-B', str(out / 'mscclpp_profile.py')]
    options = ['--library', str(library), '--real-nccl', str(real)]
    records = []
    for i, host in enumerate(hosts):
        value = subprocess.check_output(mpi(host, 1, 'slot') + launcher + ['probe'] + options,
                                        env=env, text=True, timeout=60)
        record = json.loads(value)
        records.append(record)
        (out / f'host{i}.json').write_text(value)
    shapes = [(op, (1 << power) // (4 if op == 3 else 32))
              for _, op in operations for power in range(10, 19)]
    if args.shapes:
        shapes = json.loads(args.shapes.read_text())
    (out / 'shapes.json').write_text(json.dumps(shapes, indent=2) + '\n')
    profile = out / 'candidate.tsv'
    mscclpp_profile.write_candidate(profile, records, shapes)
    subprocess.run(['rsync', '-a', str(profile), f'{hosts[1]}:{out}/'], check=True)
    prefix = mpi(','.join(h + ':4' for h in hosts), 8, 'ppr:4:node')
    manifest = {'scope': 'explicit experimental candidate; no deployment admission',
                'hardware': 'two hosts, four NVIDIA L4 per host; FP32 SUM',
                'timing': 'nccl-tests maximum-rank host enqueue/completion batch mean; both layouts',
                'baseline': 'native NCCL, identical environment and test binaries',
                'selection': 'one fixed GPU hierarchy, exact candidate extents; no best-of curve',
                'repetitions': f'{args.rounds} native/candidate/candidate/native rounds; {2 * args.rounds} jobs per backend',
                'warmup': args.warmup, 'iterations': args.iterations,
                'max_bytes': args.max_bytes, 'shapes': shapes,
                'max_regression': args.max_regression,
                'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
                'commands': [], 'results': {}}
    def run(label, op, number, backend, extra=(), routes=True, missing=False):
        command = prefix.copy()
        if backend == 'candidate':
            command += launcher + ['launch'] + options + ['--profile', str(out / 'missing.tsv' if missing else profile), '--']
        else:
            command += ['env', 'LD_PRELOAD=' + str(real)]
        command += [str(out / (op + '_perf')), '-b', '1K', '-e', str(args.max_bytes), '-f', '2',
                    '-g', '1', '-d', 'float', '-w', str(args.warmup), '-n', str(args.iterations),
                    '-c', '1', '-a', '3', *extra]
        manifest['commands'].append({'label': label, 'argv': command})
        print('Running', label, flush=True)
        with (out / (label + '.log')).open('w') as f:
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=900)
        log = (out / (label + '.log')).read_text()
        if result.returncode:
            raise RuntimeError(f'{label} exited {result.returncode}; see its log')
        manifest['results'][label] = checked(log, number, routes if backend == 'candidate' else False)
        expected = {1 << i for i in range(10, args.max_bytes.bit_length())}
        rows = manifest['results'][label]['rows_us']
        if set(rows) != expected or any(not math.isfinite(t) or t <= 0 for row in rows.values() for t in row):
            raise RuntimeError(f'{label}: incomplete size coverage or invalid timing')
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for op, number in operations:
        for i, backend in enumerate(['native', 'candidate', 'candidate', 'native'] * args.rounds):
            run(f'{op}_{i}_{backend}', op, number, backend,
                routes=any(row[0] == number for row in shapes))
        if args.performance_only:
            continue
        run(op + '_unaligned', op, number, 'candidate', ['-u', '1'])
        # Eager warmups can route before capture; capture itself stays native.
        run(op + '_graph', op, number, 'candidate', ['-G', '3'])
    if not args.performance_only:
        run('all_reduce_product', 'all_reduce', 3, 'candidate', ['-o', 'prod'], routes=False)
        run('all_reduce_missing_profile', 'all_reduce', 3, 'candidate', routes=False, missing=True)
    if args.abi:
        command = prefix + launcher + ['launch'] + options + ['--profile', str(profile), '--', str(out / 'abi_check')]
        manifest['commands'].append({'label': 'abi', 'argv': command})
        print('Running abi', flush=True)
        with (out / 'abi.log').open('w') as f:
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=180)
        log = (out / 'abi.log').read_text()
        counters = re.findall(r'UCCL Lite MSCCL\+\+: rank=(\d+) AG=9 RS=9 AR=9 fallback=9', log)
        if result.returncode or 'ABI full-buffer checks passed: 72 calls/rank, two lifetimes' not in log or any(counters.count(str(rank)) != 2 for rank in range(8)):
            raise RuntimeError('ABI correctness or exact per-lifetime route assertion failed')
        manifest['results']['abi'] = {'full_buffer_calls_per_rank': 72, 'lifetimes': 2}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    ratios = {}
    for op, _ in operations:
        order = ['native', 'candidate', 'candidate', 'native'] * args.rounds
        runs = [manifest['results'][f'{op}_{i}_{backend}']['rows_us']
                for i, backend in enumerate(order)]
        ratios[op] = {size: [statistics.median([r[size][layout] for r, b in zip(runs, order) if b == 'candidate']) /
                             statistics.median([r[size][layout] for r, b in zip(runs, order) if b == 'native'])
                             for layout in range(2)] for size in runs[0]}
    (out / 'candidate_over_nccl.json').write_text(json.dumps(ratios, indent=2) + '\n')
    print('Correctness and route assertions passed. Ratios > 1 mean slower than NCCL.', flush=True)
    if args.max_regression is not None:
        failures = [{'operation': op, 'bytes': size, 'layout': layout, 'ratio': ratio}
                    for op, rows in ratios.items() for size, pair in rows.items()
                    for layout, ratio in zip(['out-of-place', 'in-place'], pair)
                    if ratio > 1 + args.max_regression]
        verdict = {'threshold': args.max_regression, 'passed': not failures,
                   'rows': sum(len(rows) * 2 for rows in ratios.values()), 'failures': failures}
        (out / 'performance_verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')
        print(f'Performance gate: {len(failures)}/{verdict["rows"]} rows failed', flush=True)
        if failures:
            sys.exit(2)


if __name__ == '__main__':
    main()
