#!/usr/bin/env python3
"""Fresh per-call comparison against the exact original eight-L4 README binaries.

Both sides use input-ready maximum-rank CUDA events and full output checks.
The original is one process/eight GPUs; the NCCL ABI client is eight processes.
The original has AR in place and AG/RS out of place; both candidate layouts are
compared to those same original paths. No best-per-size or trimmed samples.
"""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

import local8_process

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = {
    'ar': ('full_refresh_8l4_20261114', '3d1b584e218df5c3cec99a7a7b65423f7b36a031f20407a8924527c7e963ad44'),
    'ag': ('full_refresh_8l4_20261114', '3d1b584e218df5c3cec99a7a7b65423f7b36a031f20407a8924527c7e963ad44'),
    'rs': ('rs_recovery_8l4_20261114', 'f6e80074af2c70e625932743f6af343b4a161e50cc7fc63d5b9517a02a94a45d'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def samples(log, sizes, label, repeats, iterations):
    groups = defaultdict(list)
    for line in log.splitlines():
        if not line.startswith('sample,'):
            continue
        _, n, variant, r, i, us = line.split(',')
        if variant != label:
            raise RuntimeError('unexpected variant in reference comparison')
        value = float(us)
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError('nonfinite/nonpositive sample')
        groups[int(n), int(r)].append((int(i), value))
    if set(groups) != {(n, r) for n in sizes for r in range(repeats)}:
        raise RuntimeError('missing size or repetition')
    for key, values in groups.items():
        if sorted(i for i, _ in values) != list(range(iterations)):
            raise RuntimeError('missing/duplicate sample index: ' + str(key))
    return {n: statistics.median(statistics.median(us for _, us in groups[n, r])
                                 for r in range(repeats)) for n in sizes}


def audit_routes(log, op, sizes, calls):
    routes = re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)', log)
    expected = Counter((rank, op, size, calls) for rank in range(8) for size in sizes)
    if Counter(tuple(map(int, row)) for row in routes) != expected:
        raise RuntimeError('incorrect exact route counts')
    loaded = re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0', log)
    fallback = re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)', log)
    if sorted(map(int, loaded)) != list(range(8)) or sorted(tuple(map(int, x)) for x in fallback) != [(r, 0) for r in range(8)]:
        raise RuntimeError('missing no-native/no-fallback proof')
    if 'PASS local8 reference full-output checks' not in log:
        raise RuntimeError('missing full-output success summary')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True, help='frozen run_local8 output directory')
    p.add_argument('--client', type=Path, default=ROOT / 'build/nccl_local8/reference_timing_check')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-bytes', type=int, default=1 << 30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--jobs', type=int, default=2, help='independent jobs/backend; two or more for qualification')
    p.add_argument('--ops', nargs='+', choices=list(REFERENCES), default=list(REFERENCES))
    p.add_argument('--layouts', nargs='+', choices=['out', 'in'], default=['out', 'in'])
    args = p.parse_args()
    if not 1024 <= args.max_bytes <= 1 << 30 or args.max_bytes & (args.max_bytes - 1):
        p.error('invalid maximum extent')
    if min(args.warmup, args.iterations, args.repeats, args.jobs) < 1:
        p.error('iteration/repetition counts must be positive')
    lock = open('/tmp/uccl-lite-integration-wxz-l4.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('GPU already occupied')
    candidate = args.candidate.resolve()
    source = json.loads((candidate / 'manifest.json').read_text())
    library = next(candidate.glob('libuccl_lite_nccl*.so'))
    if sha(library) != source['library_sha256']:
        raise RuntimeError('candidate binary changed')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    client = out / 'reference_timing_check'
    shutil.copy2(args.client, client)
    shutil.copy2(__file__, out / 'comparison_driver.py')
    shutil.copy2(Path(__file__).with_name('local8_process.py'), out / 'local8_process.py')
    refs = {}
    for op in args.ops:
        directory, expected = REFERENCES[op]
        original = ROOT / 'build' / directory / 'uccl_lite_compact_collectives'
        if sha(original) != expected:
            raise RuntimeError('original README binary identity mismatch: ' + op)
        refs[op] = out / ('original_' + op)
        shutil.copy2(original, refs[op])
    sizes = [1 << power for power in range(10, args.max_bytes.bit_length())]
    env = source['environment'].copy()
    real = json.loads((candidate / 'host0.json').read_text())['details']['nccl_sha256']
    manifest = dict(candidate=str(candidate), library_sha256=sha(library),
                    profile_sha256=sha(candidate / 'candidate.tsv'), host_identity=source.get('host_identity'),
                    reference_sha256={op: sha(ref) for op, ref in refs.items()},
                    client_sha256=sha(client), nccl_sha256=real, sizes=sizes,
                    timing=__doc__, warmup=args.warmup, iterations=args.iterations,
                    repeats=args.repeats, jobs=args.jobs, commands=[], results={})

    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def run(label, command, expected_variant, op, routed):
        entry = dict(label=label, argv=command, started=time.time())
        manifest['commands'].append(entry)
        save()
        print('Running', label, flush=True)
        with (out / (label + '.log')).open('x') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                status = local8_process.wait(child, 1800,
                                              log_path=out / (label + '.log'),
                                              affinity_path=out / (label + '.affinity.json') if routed else None)
            except BaseException:
                entry.update(returncode=child.returncode, interrupted=True)
                save()
                raise
        entry.update(returncode=status, elapsed=time.time() - entry['started'])
        save()
        if status:
            raise RuntimeError(label + ' failed; inspect retained log')
        text = (out / (label + '.log')).read_text()
        if routed:
            if args.max_bytes == 1 << 30 and not (out / (label + '.affinity.json')).is_file():
                raise RuntimeError('missing runtime worker-isolation audit')
            audit_routes(text, {'ag': 1, 'rs': 2, 'ar': 3}[op], sizes,
                         args.repeats * (args.warmup + args.iterations))
        result = samples(text, sizes, expected_variant, args.repeats, args.iterations)
        manifest['results'][label] = result
        save()

    # Reverse the relative order of original and candidate in the second job.
    for op in args.ops:
        for job in range(args.jobs):
            order = ['original', *args.layouts]
            if job % 2:
                order.reverse()
            for layout in order:
                if layout == 'original':
                    command = [str(refs[op]), '--gpus', '8', '--op', op,
                               '--begin', '1024', '--end', str(args.max_bytes),
                               '--warmup', str(args.warmup), '--iterations', str(args.iterations),
                               '--repeats', str(args.repeats), '--variants', 'current']
                    run(f'{op}_{job}_original', command, 'current', op, False)
                else:
                    # The saved launcher revalidates the frozen identity on each rank.
                    command = ['mpirun', '--timeout', '1700', '-np', '8', '--bind-to', 'none',
                               '--map-by', 'slot', '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp',
                               'python3', '-B', str(candidate / 'mscclpp_profile.py'), 'launch',
                               '--local8', '--library', str(library),
                               '--real-nccl', str(ROOT / 'build/nccl-2.28.9/lib/libnccl.so.2'),
                               '--profile', str(candidate / 'candidate.tsv'), '--', str(client),
                               op, layout, '1024', str(args.max_bytes), str(args.warmup),
                               str(args.iterations), str(args.repeats)]
                    run(f'{op}_{job}_{layout}', command, 'candidate', op, True)
    rows = []
    for op in args.ops:
        for size in sizes:
            baseline = statistics.median(manifest['results'][f'{op}_{j}_original'][size] for j in range(args.jobs))
            for layout in args.layouts:
                us = statistics.median(manifest['results'][f'{op}_{j}_{layout}'][size] for j in range(args.jobs))
                rows.append(dict(operation=op, bytes=size, layout=layout, reference_us=baseline,
                                 candidate_us=us, ratio=us / baseline, passed=us <= baseline))
    with (out / 'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    failures = [r for r in rows if not r['passed']]
    full = (args.max_bytes == 1 << 30 and args.jobs >= 2 and args.warmup == 5 and
            args.iterations == 10 and args.repeats == 3 and set(args.ops) == set(REFERENCES)
            and set(args.layouts) == {'out', 'in'})
    verdict = dict(qualified=full and not failures, full_protocol=full,
                   rows=len(rows), failures=failures, zero_regression=not failures)
    (out / 'verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')
    print(f'{len(failures)}/{len(rows)} rows slower than original UCCL; full protocol={full}', flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
