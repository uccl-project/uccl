#!/usr/bin/env python3
"""Recompute the local8 strict correctness and original-UCCL performance gates."""
import argparse
from collections import Counter
import math
import re
import csv
import json
from pathlib import Path
import statistics

from compare_local8_reference import REFERENCES, audit_routes, samples, sha
from run import checked, checked_strict


def audit_affinity(path):
    snapshot = json.loads(path.read_text())
    primary = {6 * rank + offset for rank in range(8) for offset in (4, 5)}
    reserved = primary | {cpu + 48 for cpu in primary}
    if set(snapshot['reserved_cpus']) != reserved or len(snapshot['processes']) != 8:
        raise RuntimeError('invalid recorded worker-core reservation')
    seen = set()
    for process in snapshot['processes']:
        roles = []
        for thread in process['threads']:
            cpus = set(thread['cpus'])
            if thread['name'] in ('uccl_rs_worker', 'uccl_ar_lane'):
                if len(cpus) != 1 or not cpus <= primary or seen & cpus:
                    raise RuntimeError('invalid recorded worker/helper CPU')
                seen.update(cpus)
                roles.append(thread['name'])
            elif cpus & reserved:
                raise RuntimeError('recorded application thread overlaps a reserved core')
        if sorted(roles) != ['uccl_ar_lane', 'uccl_rs_worker']:
            raise RuntimeError('missing worker/helper thread')
    if seen != primary:
        raise RuntimeError('incomplete recorded worker affinity')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tests', type=Path, required=True, help='completed run_local8 directory')
    p.add_argument('--paired', type=Path, required=True, help='completed compare_local8_reference directory')
    p.add_argument('--library', type=Path, required=True, help='current library being qualified')
    args = p.parse_args()
    batch = json.loads((args.tests / 'manifest.json').read_text())
    paired = json.loads((args.paired / 'manifest.json').read_text())
    digest = sha(args.library)
    profile_digest = sha(args.tests / 'candidate.tsv')
    if (Path(paired['candidate']).resolve() != args.tests.resolve() or
            batch['profile_sha256'] != profile_digest or paired['profile_sha256'] != profile_digest or
            batch['host_identity'] != paired['host_identity']):
        raise RuntimeError('batch/per-call profile or runtime configuration mismatch')
    if batch['library_sha256'] != digest or paired['library_sha256'] != digest:
        raise RuntimeError('current/batch/paired library mismatch')
    sizes = {1 << n for n in range(10, 31)}
    if set(batch['sizes']) != sizes or set(paired['sizes']) != sizes:
        raise RuntimeError('full 1 KiB–1 GiB coverage required')
    if (paired['warmup'], paired['iterations'], paired['repeats']) != (5, 10, 3) or paired['jobs'] < 2:
        raise RuntimeError('incomplete original-reference timing protocol')
    if sha(args.paired / 'reference_timing_check') != paired['client_sha256']:
        raise RuntimeError('reference client changed')
    batch_calls = 2 * (batch['warmup'] + batch['iterations'] + 1)
    operations = {'all_gather': 1, 'reduce_scatter': 2, 'all_reduce': 3}
    expected_batch = set()
    order = ['candidate'] if batch['candidate_only'] else ['native', 'candidate', 'candidate', 'native'] * batch['rounds']
    commands = {c['label']: c for c in batch['commands']}
    for name, op in operations.items():
        if sha(args.tests / (name + '_perf')) != batch['binary_sha256'][name + '_perf']:
            raise RuntimeError('nccl-tests binary changed')
        for i, backend in enumerate(order):
            label = f'{name}_{i}_{backend}'
            expected_batch.add(label)
            command = commands[label]
            if command['returncode'] != 0:
                raise RuntimeError('failed nccl-tests sweep')
            argv = command['argv']
            for flag, value in [('-b', '1K'), ('-e', str(1 << 30)), ('-d', 'bfloat16'),
                                ('-g', '1'), ('-w', str(batch['warmup'])),
                                ('-n', str(batch['iterations'])), ('-c', '1')]:
                if argv[argv.index(flag) + 1] != value:
                    raise RuntimeError('nccl-tests protocol mismatch')
            text = (args.tests / (label + '.log')).read_text()
            result = checked_strict(text, op) if backend == 'candidate' else checked(text, op, False)
            if set(result['rows_us']) != sizes or any(not math.isfinite(us) or us <= 0
                                                       for row in result['rows_us'].values() for us in row):
                raise RuntimeError('missing nccl-tests sizes')
            if backend == 'candidate':
                audit_affinity(args.tests / (label + '.affinity.json'))
            if backend == 'candidate' and any(int(row[3]) != batch_calls for row in result['strict_routes']):
                raise RuntimeError('incorrect nccl-tests route counts')
            # JSON normalizes integer keys to strings in the saved manifest.
            if json.loads(json.dumps(result)) != batch['results'][label]:
                raise RuntimeError('nccl-tests raw log/manifest mismatch')
    if not expected_batch <= set(batch['results']):
        raise RuntimeError('missing nccl-tests sweep')
    # Performance cannot qualify a library that skipped ABI/queue/error gates.
    abi_text = (args.tests / 'abi.log').read_text()
    if commands['abi']['returncode'] or sha(args.tests / 'abi_check') != batch['abi_sha256']:
        raise RuntimeError('ABI binary or process status mismatch')
    abi_sizes = [1024, 16384, 32768, 65536, 131072, 262144, 524288,
                 1048576, 8388608, 16777216, 33554432, 67108864, 134217728]
    routes = re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)', abi_text)
    expected = Counter({(r, op, n, 3): 2 for r in range(8) for op in (1, 2, 3) for n in abi_sizes})
    if ('Strict ABI full-buffer checks passed: 234 calls/rank, two lifetimes' not in abi_text
            or Counter(tuple(map(int, row)) for row in routes) != expected):
        raise RuntimeError('ABI full-output or route audit failed')
    loaded = re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0', abi_text)
    if Counter(map(int, loaded)) != Counter({r: 2 for r in range(8)}):
        raise RuntimeError('ABI native-library absence audit failed')
    queue_text = (args.tests / 'queue.log').read_text()
    if commands['queue']['returncode'] or sha(args.tests / 'queue_check') != batch['queue_sha256']:
        raise RuntimeError('queue binary or process status mismatch')
    expected = Counter((r, c % 3 + 1, min(1024 << (6 * ((c // 3) % 3)), 1 << 20))
                       for r in range(8) for c in range(128))
    routes = re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)', queue_text)
    actual = Counter({tuple(map(int, row[:3])): int(row[3]) for row in routes})
    if (len(routes) != len(expected) or actual != expected or
            'PASS local8 queued: 128 mixed calls/rank, alternating streams' not in queue_text):
        raise RuntimeError('queue full-output or route audit failed')
    for label, reason in [('dtype_rejected', 'outside the strict profile'),
                          ('product_rejected', 'outside the strict profile'),
                          ('group_rejected', 'grouped collectives unsupported'),
                          ('capture_rejected', 'network graph capture unsupported'),
                          ('missing_profile_rejected', 'profile/topology consensus failed')]:
        text = (args.tests / (label + '.log')).read_text()
        if not commands[label]['returncode'] or reason not in text:
            raise RuntimeError('unsupported-call/profile rejection failed: ' + label)
        if label in ('dtype_rejected', 'product_rejected', 'missing_profile_rejected') and 'UCCL Lite packet mode' in text:
            raise RuntimeError('rejected operation prepared the packet backend')
    for name, op in operations.items():
        label = name + '_loader_alias'
        argv = commands[label]['argv']
        if commands[label]['returncode'] or not any(argv[i:i+3] == ['env', '-u', 'LD_PRELOAD'] for i in range(len(argv)-2)):
            raise RuntimeError('missing direct loader alias check')
        result = checked_strict((args.tests / (label + '.log')).read_text(), op)
        if set(result['rows_us']) != {1 << n for n in range(10, 21)} or any(int(r[3]) != 8 for r in result['strict_routes']):
            raise RuntimeError('incomplete loader alias sweep')
    original, candidate = {}, {}
    expected_jobs = {f'{op}_{j}_{layout}' for op in REFERENCES
                     for j in range(paired['jobs']) for layout in ['original', 'out', 'in']}
    if {c['label'] for c in paired['commands']} != expected_jobs:
        raise RuntimeError('missing/extra reference jobs')
    for op, (_, expected) in REFERENCES.items():
        if paired['reference_sha256'][op] != expected or sha(args.paired / ('original_' + op)) != expected:
            raise RuntimeError('original README binary mismatch')
        for j in range(paired['jobs']):
            for layout in ['original', 'out', 'in']:
                label = f'{op}_{j}_{layout}'
                command = next(c for c in paired['commands'] if c['label'] == label)
                if command['returncode'] != 0:
                    raise RuntimeError('failed reference job')
                text = (args.paired / (label + '.log')).read_text()
                if layout != 'original':
                    audit_affinity(args.paired / (label + '.affinity.json'))
                    audit_routes(text, {'ag': 1, 'rs': 2, 'ar': 3}[op], sizes, 45)
                values = samples(text, sizes, 'current' if layout == 'original' else 'candidate', 3, 10)
                if {str(n): us for n, us in values.items()} != paired['results'][label]:
                    raise RuntimeError('paired raw log/manifest mismatch')
                target = original if layout == 'original' else candidate
                for n, us in values.items():
                    target[op, n, layout, j] = us
    stored = {(r['operation'], int(r['bytes']), r['layout']): r
              for r in csv.DictReader((args.paired / 'comparison.csv').open())}
    expected_rows = {(op, size, layout) for op in REFERENCES for size in sizes for layout in ['out', 'in']}
    if set(stored) != expected_rows:
        raise RuntimeError('incomplete comparison table')
    failures = []
    for op, n, layout in sorted(expected_rows):
        ref = statistics.median(original[op, n, 'original', j] for j in range(paired['jobs']))
        us = statistics.median(candidate[op, n, layout, j] for j in range(paired['jobs']))
        row = stored[op, n, layout]
        if float(row['reference_us']) != ref or float(row['candidate_us']) != us or float(row['ratio']) != us / ref:
            raise RuntimeError('stored comparison does not match raw samples')
        if us > ref:
            failures.append((op, n, layout, us / ref))
    print(f'Raw-log audit: {len(expected_batch)} nccl-tests sweeps, {len(expected_jobs)} paired jobs, '
          f'{len(failures)}/126 zero-regression failures.')
    if failures:
        for row in failures:
            print(*row)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
