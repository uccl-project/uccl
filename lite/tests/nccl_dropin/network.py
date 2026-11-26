"""Exercise the two-host NCCL ABI against unchanged MPI nccl-tests."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from run import checked_output
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
import network_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hosts', required=True, help='ordered node0:4,node1:4')
    p.add_argument('--real-nccl', type=Path, required=True)
    p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mpirun', default='mpirun')
    p.add_argument('--iface', required=True)
    p.add_argument('--control-root', help='common rank-0 IPv4 address for CPU metadata')
    p.add_argument('--graph', type=int, default=0)
    p.add_argument('--min-bytes', default='128')
    p.add_argument('--max-bytes', default='131072')
    p.add_argument('--unalign', type=int, default=0)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--operations', nargs='+', choices=['all_reduce', 'all_gather', 'reduce_scatter'],
                   default=['all_reduce', 'all_gather', 'reduce_scatter'])
    p.add_argument('--abba', action='store_true')
    p.add_argument('--profile', type=Path, help='validate an existing deployment profile')
    p.add_argument('--cases', type=Path, help='candidate [operation,count] pairs; never deployment admission')
    p.add_argument('--stress-binary', type=Path, help='run the full-buffer ABI stress instead of nccl-tests')
    p.add_argument('--stale', action='store_true', help='corrupt the second host identity and require fallback')
    p.add_argument('--missing', action='store_true', help='hide the profile on the second host and require fallback')
    args = p.parse_args()
    if args.profile and (args.stale or args.missing):
        p.error('stale/missing tests use a fresh candidate, not an existing profile')
    hosts = args.hosts.split(',')
    if len(hosts) != 2 or any(not h.endswith(':4') for h in hosts):
        p.error('requires exactly two ordered four-slot hosts')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    if args.control_root:
        env['UCCL_LITE_DROPIN_CONTROL_ROOT'] = args.control_root
    env.update(MSCCLPP_SOCKET_IFNAME=args.iface, MSCCLPP_HCA_DEVICES='mlx5_0,mlx5_1',
               MSCCLPP_IBV_MODE='host-no-atomic', NCCL_SOCKET_IFNAME=args.iface,
               NCCL_IB_HCA='mlx5_0,mlx5_1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('LD_PRELOAD', None)
    env['LD_LIBRARY_PATH'] = str(args.real_nccl.resolve().parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    export = []
    for key in env:
        if key.startswith(('NCCL_', 'CUDA_', 'MSCCLPP_')) or key in ('LD_LIBRARY_PATH', 'PYTHONDONTWRITEBYTECODE', 'UCCL_LITE_DROPIN_CONTROL_ROOT', 'UCCL_LITE_DROPIN_NETWORK_PRIME_ITERS'):
            export += ['-x', key]
    def mpi(count, slots):
        return [args.mpirun, '--host', slots, '-np', str(count), '--map-by',
                f'ppr:{count // 2}:node', '--bind-to', 'none', '--mca', 'pml', 'ob1',
                '--mca', 'btl', 'self,vader,tcp', '--mca', 'btl_tcp_if_include', args.iface, *export]
    launcher = [sys.executable, str(ROOT / 'integration/nccl_lite/network_profile.py')]
    # Use the system Python path available identically on both hosts.
    launcher[0] = 'python3'
    probe = mpi(2, ','.join(h[:-1] + '1' for h in hosts)) + launcher + [
        'probe', '--real-nccl', str(args.real_nccl.resolve())]
    result = subprocess.run(probe, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120)
    (args.out / 'probes.log').write_text(result.stdout)
    result.check_returncode()
    records = sorted([json.loads(l) for l in result.stdout.splitlines() if l.startswith('{')], key=lambda r: r['probe_rank'])
    if len(records) != 2:
        raise RuntimeError('missing host identity probes')
    if args.stale:
        records[1]['identity'] = '0' * 64
    profile = args.out / ('profile.tsv' if args.profile else 'candidate.tsv')
    if args.profile:
        profile.write_text(args.profile.read_text())
    else:
        cases = json.loads(args.cases.read_text()) if args.cases else [[op, n] for op in (1, 2, 3) for n in (32, 256, 4096)]
        network_profile.write_profile(profile, records, [[op, n, 1, 1, 3] for op, n in cases], candidate=True)
    selected = {(int(f[0]), int(f[1])) for line in profile.read_text().splitlines()
                if len(f := line.split()) == 5 and f[0] in ('1', '2', '3')}
    mode = [] if args.profile else ['--calibrating']
    remote = hosts[1][:-2]
    subprocess.run(['ssh', remote, 'mkdir', '-p', shlex.quote(str(args.out))], check=True)
    subprocess.run(['rsync', '-a', str(profile), f'{remote}:{profile}'], check=True)
    if args.missing:
        subprocess.run(['ssh', remote, 'mv', shlex.quote(str(profile)), shlex.quote(str(profile) + '.missing')], check=True)
    manifest = {'records': records, 'commands': [], 'scope': 'deployment validation' if args.profile else 'explicit calibration candidate, not deployment admission',
                'timing': {'dtype': 'float32', 'reduction': 'sum', 'unalign_elements': args.unalign,
                           'warmup': args.warmup, 'iterations': args.iterations,
                           'scope': 'host enqueue+completion wall-time batch mean per ABI call; maximum rank; both layouts',
                           'variant_selection': 'one fixed double-buffer implementation'},
                'driver_sha256': network_profile.local.sha256(__file__)}
    if args.stress_binary:
        command = mpi(8, args.hosts) + launcher + [
            'launch', '--real-nccl', str(args.real_nccl.resolve()), '--profile', str(profile),
            *mode, '--', str(args.stress_binary.resolve())]
        log = args.out / 'stress.log'
        with log.open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=240)
        text = log.read_text()
        if 'PASS:' not in text:
            raise RuntimeError('missing stress success summary')
        counters = re.findall(r'rank=(\d+) AG=(\d+) RS=(\d+) AR=(\d+)', text)
        if args.stale or args.missing:
            if counters or 'completed IPC import' in text:
                raise RuntimeError('stale identity prepared or used the network route')
        else:
            if any(sum(int(row[0]) == rank for row in counters) != 2 for rank in range(8)):
                raise RuntimeError('a pre-existing unsupported graph allowed later network preparation')
            for slot in (1, 2, 3):
                if any(op == slot for op, count in selected) and {int(row[0]) for row in counters if int(row[slot]) > 0} != set(range(8)):
                    raise RuntimeError('missing stress route counts')
        manifest['commands'].append(command)
        (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('PASS: network ABI stress' + (' with stale/missing-profile fallback' if args.stale or args.missing else ''), flush=True)
        return
    order = ['nccl', 'uccl', 'uccl', 'nccl'] if args.abba else ['uccl']
    summaries = []
    for op, slot in [('all_reduce', 3), ('all_gather', 1), ('reduce_scatter', 2)]:
        if op not in args.operations:
            continue
        binary = args.tests.resolve() / (op + '_perf')
        dependencies = subprocess.check_output(['ldd', str(binary)], env=env, text=True)
        if 'libmpi.so' not in dependencies:
            raise RuntimeError('requires MPI-enabled nccl-tests')
        jobs = []
        for index, backend in enumerate(order):
            target = ([*launcher, 'launch', '--real-nccl', str(args.real_nccl.resolve()),
                       '--profile', str(profile), *mode, '--'] if backend == 'uccl'
                      else ['env', 'LD_PRELOAD=' + str(args.real_nccl.resolve())])
            command = mpi(8, args.hosts) + target + [str(binary), '-b', args.min_bytes, '-e', args.max_bytes,
                '-f', '2', '-g', '1', '-d', 'float', '-w', str(args.warmup), '-n', str(args.iterations), '-c', '1', '-G', str(args.graph), '-u', str(args.unalign), '-a', '3']
            log = args.out / f'{op}_{index}_{backend}.log'
            print(op, index, backend, flush=True)
            with log.open('w') as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=240)
            text = log.read_text()
            jobs.append(checked_output(text, 8))
            if backend == 'uccl':
                counters = re.findall(r'rank=(\d+) AG=(\d+) RS=(\d+) AR=(\d+)', text)
                if (args.stale or args.missing) and counters:
                    raise RuntimeError('stale profile unexpectedly selected a route')
                observed_counts = {int(size) // 4 // (1 if slot == 3 else 8) for size in jobs[-1]}
                admitted_counts = {count for op, count in selected if op == slot and count in observed_counts}
                # nccl-tests warms the entire size range eagerly before it
                # captures the first graph. Capture then permanently retains
                # NCCL. BenchTime also issues one eager sync call at the first
                # size immediately before the first capture.
                first = min((n for n in observed_counts if n), default=0)
                eager_graph_calls = args.warmup * len(admitted_counts) + int(first in admitted_counts)
                expect_route = not (args.stale or args.missing) and bool(admitted_counts) and (not args.graph or eager_graph_calls > 0)
                if not expect_route and counters:
                    raise RuntimeError('fallback-only call pattern unexpectedly prepared a route')
                if expect_route and {int(row[0]) for row in counters if int(row[slot]) > 0} != set(range(8)):
                    raise RuntimeError('missing selected-route counts on all eight ranks')
                if expect_route and args.graph:
                    if len(counters) != 8 or any(int(row[slot]) != eager_graph_calls for row in counters):
                        raise RuntimeError('graph route counts differ from eager pre-capture calls')
            manifest['commands'].append(command)
            (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if args.abba:
            if any(set(job) != set(jobs[0]) for job in jobs):
                raise RuntimeError('paired jobs have different size rows')
            for size in jobs[0]:
                count = int(size) // 4 // (1 if slot == 3 else 8)
                for layout in range(2):
                    timings = [job[size][layout] for job in jobs]
                    summaries.append({'operation': op, 'bytes': int(size),
                        'count': count, 'layout': ['out', 'inplace'][layout],
                        'candidate_selected': args.graph == 0 and not (args.stale or args.missing) and (slot, count) in selected,
                        'jobs_us': timings,
                        'worst_uccl_over_best_nccl': max(timings[1:3]) / min(timings[0], timings[3])})
            (args.out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    if args.profile and args.abba:
        failures = [r for r in summaries if r['worst_uccl_over_best_nccl'] > 1.05]
        print('Deployment mixed-curve 5% failures:', len(failures), '/', len(summaries), flush=True)
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
