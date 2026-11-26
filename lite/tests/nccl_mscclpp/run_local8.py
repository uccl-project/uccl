#!/usr/bin/env python3
"""Eight-L4 port of the strict multi-node nccl-tests qualification driver.

Uses the same ABI library source, identity launcher, and per-rank route audit.
Candidate experiments never admit a deployment policy implicitly.
"""
import argparse
from collections import Counter
import re
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import local8_process

from run import checked, checked_strict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
import mscclpp_profile
import dropin_profile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, default=ROOT / 'build/nccl_local8/libuccl_lite_nccl_local8.so')
    p.add_argument('--tests', type=Path, default=ROOT / 'build/nccl-tests-local8-src/build')
    p.add_argument('--real-nccl', type=Path, default=ROOT / 'build/nccl-2.28.9/lib/libnccl.so.2')
    p.add_argument('--mpirun', default='mpirun')
    p.add_argument('--max-bytes', type=int, default=1 << 30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--candidate-only', action='store_true', help='correctness screen; no performance qualification')
    p.add_argument('--queue', type=Path, help='mixed-operation queue/stream stress client')
    p.add_argument('--checks', action='store_true', help='also verify explicit unsupported-call and stale-profile rejection')
    p.add_argument('--abi', type=Path, help='strict local8 ABI stress client')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not 1024 <= args.max_bytes <= 1 << 30 or args.max_bytes & (args.max_bytes - 1):
        p.error('max-bytes must be a power of two in 1 KiB–1 GiB')
    if args.iterations < 1 or args.warmup < 0 or args.rounds < 1 or args.timeout < 1:
        p.error('invalid iteration, round or timeout count')
    if os.environ.get('LD_PRELOAD'):
        p.error('unset LD_PRELOAD before validation')
    lock = open('/tmp/uccl-lite-integration-wxz-l4.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    users = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'], text=True).strip()
    if users:
        raise RuntimeError('GPU occupied: ' + users)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    library = out / args.library.name
    shutil.copy2(args.library, library)
    (out / 'strict').mkdir()
    (out / 'strict/libnccl.so.2').symlink_to('../' + library.name)
    for name in ('mscclpp_profile.py', 'dropin_profile.py'):
        shutil.copy2(ROOT / 'integration/nccl_lite' / name, out / name)
    shutil.copy2(__file__, out / 'validation_driver.py')
    shutil.copy2(Path(__file__).with_name('local8_process.py'), out / 'local8_process.py')
    shutil.copy2(Path(__file__).with_name('run.py'), out / 'run.py')
    dependencies = subprocess.check_output(['ldd', str(library)], text=True)
    if 'libnccl.so' in dependencies or 'not found' in dependencies:
        raise RuntimeError('unresolved dependency or native NCCL dependency')
    (out / 'dependencies.txt').write_text(dependencies)
    operations = [('all_gather', 1), ('reduce_scatter', 2), ('all_reduce', 3)]
    binaries = {}
    for name, _ in operations:
        target = out / (name + '_perf')
        shutil.copy2(args.tests / target.name, target)
        binaries[target.name] = sha(target)
    real = args.real_nccl.resolve(strict=True)
    # Match the working native control on this host. Do not select algorithms,
    # protocols, channel counts or cuMem settings differently between backends.
    env = {k: v for k, v in os.environ.items()
           if k in ('PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR',
                    'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')}
    env.update(NCCL_NET='Socket', NCCL_NET_PLUGIN='none', NCCL_PROFILER_PLUGIN='none',
               NCCL_IB_DISABLE='1', UCCL_LITE_MSCCLPP_CONTROL_ROOT='127.0.0.1',
               LD_LIBRARY_PATH=str(real.parent) + ':/usr/local/cuda-12.9/lib64')
    env.update(mscclpp_profile.local_packet_environment())
    launcher = [sys.executable, '-B', str(out / 'mscclpp_profile.py')]
    options = ['--local8', '--library', str(library), '--real-nccl', str(real)]
    value = subprocess.check_output(launcher + ['probe', *options], env=env, text=True)
    record = json.loads(value)
    (out / 'host0.json').write_text(value)
    sizes = [1 << power for power in range(10, args.max_bytes.bit_length())]
    shapes = [(op, size // (2 if op == 3 else 16)) for _, op in operations for size in sizes]
    profile = out / 'candidate.tsv'
    mscclpp_profile.write_candidate(profile, [record], shapes, local8=True)
    prefix = [args.mpirun, '--timeout', str(args.timeout), '-np', '8',
              '--bind-to', 'none', '--map-by', 'slot', '--mca', 'pml', 'ob1',
              '--mca', 'btl', 'self,vader,tcp']
    manifest = dict(hardware='wxz-l4, eight NVIDIA L4, one GPU/process, BF16 SUM',
                    timing='nccl-tests maximum-rank batch mean, both layouts',
                    baseline='native NCCL in separate processes',
                    selection='one fixed local packet configuration; no best-of curve',
                    library_sha256=sha(library), nccl_sha256=sha(real),
                    binary_sha256=binaries, environment=env, sizes=sizes,
                    warmup=args.warmup, iterations=args.iterations, rounds=args.rounds,
                    candidate_only=args.candidate_only, commands=[], results={},
                    profile_sha256=sha(profile), host_identity=record['identity'],
                    source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    nccl_tests_revision=subprocess.check_output(['git', '-C', str(args.tests), 'rev-parse', 'HEAD'], text=True).strip())

    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def execute(label, command, rejection=None):
        entry = dict(label=label, argv=command, started=time.time())
        manifest['commands'].append(entry)
        save()
        print('Running', label, flush=True)
        with (out / (label + '.log')).open('x') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                status = local8_process.wait(child, args.timeout + 30,
                                              log_path=out / (label + '.log'),
                                              affinity_path=out / (label + '.affinity.json') if rejection is None else None)
            except BaseException:
                entry.update(returncode=child.returncode, interrupted=True)
                save()
                raise
        entry.update(returncode=status, elapsed=time.time() - entry['started'])
        save()
        text = (out / (label + '.log')).read_text()
        if rejection:
            if not status or rejection not in text:
                raise RuntimeError(label + ': expected strict rejection not observed')
            manifest.setdefault('rejections', {})[label] = rejection
            save()
            return text
        if status:
            raise RuntimeError(f'{label} exited {status}; see {out / (label + ".log")}')
        return (out / (label + '.log')).read_text()

    order = ['candidate'] if args.candidate_only else ['native', 'candidate', 'candidate', 'native'] * args.rounds
    for name, op in operations:
        for i, backend in enumerate(order):
            label = f'{name}_{i}_{backend}'
            command = prefix + (launcher + ['launch', *options, '--profile', str(profile), '--']
                                if backend == 'candidate' else ['env', 'LD_PRELOAD=' + str(real)])
            command += [str(out / (name + '_perf')), '-b', '1K', '-e', str(args.max_bytes),
                        '-f', '2', '-g', '1', '-d', 'bfloat16', '-o', 'sum',
                        '-w', str(args.warmup), '-n', str(args.iterations), '-c', '1',
                        '-a', '3', '-m', '1', '-G', '0', '-R', '0']
            log = execute(label, command)
            result = checked_strict(log, op) if backend == 'candidate' else checked(log, op, False)
            if backend == 'candidate' and args.max_bytes == 1 << 30 and not (out / (label + '.affinity.json')).is_file():
                raise RuntimeError('missing runtime worker-isolation audit')
            if backend == 'candidate' and any(int(row[3]) != 2 * (args.warmup + args.iterations + 1)
                                                for row in result['strict_routes']):
                raise RuntimeError('unexpected nccl-tests collective call counts')
            rows = result['rows_us']
            if set(rows) != set(sizes) or any(not math.isfinite(t) or t <= 0 for pair in rows.values() for t in pair):
                raise RuntimeError('incomplete size coverage or invalid timings')
            manifest['results'][label] = result
            save()
    if args.abi:
        shutil.copy2(args.abi, out / 'abi_check')
        log = execute('abi', prefix + launcher + ['launch', *options, '--profile', str(profile), '--', str(out / 'abi_check')])
        abi_sizes = [1024, 16384, 32768, 65536, 131072, 262144, 524288,
                     1048576, 8388608, 16777216, 33554432, 67108864, 134217728]
        expected = Counter({(rank, op, size, 3): 2 for rank in range(8)
                            for op in (1, 2, 3) for size in abi_sizes})
        routes = re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)', log)
        if ('Strict ABI full-buffer checks passed: 234 calls/rank, two lifetimes' not in log
                or Counter(tuple(map(int, row)) for row in routes) != expected):
            raise RuntimeError('ABI summary or exact per-lifetime route check failed')
        manifest['abi_sha256'] = sha(out / 'abi_check')
        save()
    if args.queue:
        shutil.copy2(args.queue, out / 'queue_check')
        log = execute('queue', prefix + launcher + ['launch', *options, '--profile', str(profile), '--', str(out / 'queue_check')])
        expected = Counter((rank, c % 3 + 1, min(1024 << (6 * ((c // 3) % 3)), 1 << 20))
                           for rank in range(8) for c in range(128))
        routes = re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)', log)
        actual = Counter({tuple(map(int, row[:3])): int(row[3]) for row in routes})
        if 'PASS local8 queued: 128 mixed calls/rank, alternating streams' not in log or actual != expected:
            raise RuntimeError('mixed queue success summary or route counts mismatch')
        manifest['queue_sha256'] = sha(out / 'queue_check')
        save()
    if args.checks:
        for label, flags, reason, selected_profile in [
            ('dtype_rejected', ['-d', 'float'], 'outside the strict profile', profile),
            ('product_rejected', ['-o', 'prod'], 'outside the strict profile', profile),
            ('group_rejected', ['-m', '2'], 'grouped collectives unsupported', profile),
            ('capture_rejected', ['-G', '1'], 'network graph capture unsupported', profile),
            ('missing_profile_rejected', [], 'profile/topology consensus failed', out / 'missing.tsv'),
        ]:
            command = prefix + launcher + ['launch', *options, '--profile', str(selected_profile), '--',
                      str(out / 'all_reduce_perf'), '-b', '1K', '-e', '1K', '-g', '1',
                      '-d', 'bfloat16', '-o', 'sum', '-w', '1', '-n', '2', '-c', '1', *flags]
            execute(label, command, rejection=reason)
        for name, op in operations:
            label = name + '_loader_alias'
            # Exercise the libnccl.so.2 dependency alias without interposition.
            command = prefix + launcher + ['launch', *options, '--profile', str(profile), '--',
                      'env', '-u', 'LD_PRELOAD', str(out / (name + '_perf')),
                      '-b', '1K', '-e', str(min(args.max_bytes, 1 << 20)), '-f', '2',
                      '-g', '1', '-d', 'bfloat16', '-o', 'sum', '-w', '1', '-n', '2',
                      '-c', '1', '-m', '1', '-G', '0', '-R', '0']
            result = checked_strict(execute(label, command), op)
            if any(int(row[3]) != 8 for row in result['strict_routes']):
                raise RuntimeError('loader alias route counts mismatch')
            manifest['results'][label] = result
            save()
    if not args.candidate_only:
        rows = []
        for name, _ in operations:
            for size in sizes:
                for layout in range(2):
                    values = {backend: statistics.median([
                        manifest['results'][f'{name}_{i}_{b}']['rows_us'][size][layout]
                        for i, b in enumerate(order) if b == backend]) for backend in ('native', 'candidate')}
                    rows.append(dict(operation=name, bytes=size,
                                     layout=['out-of-place', 'in-place'][layout],
                                     native_us=values['native'], candidate_us=values['candidate'],
                                     ratio=values['candidate'] / values['native']))
        (out / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(f'Correctness and strict routing passed; {sum(r["ratio"] > 1 for r in rows)}/{len(rows)} rows slower than native NCCL.', flush=True)
    else:
        print('Correctness and strict routing passed (performance not qualified).', flush=True)


if __name__ == '__main__':
    main()
