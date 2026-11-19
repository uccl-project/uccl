#!/usr/bin/env python3
"""Audit completed paired UCCL runs from raw logs; zero tolerance, no trimming.

This verifies only the paired protocol. Full nccl-tests and release/build
qualification remain separate requirements. Exit 2 means slower rows remain.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

OPERATIONS = {'ag': (1, 'allgather'), 'ar': (3, 'allreduce'),
              'rs': (2, 'reducescatter')}
SIZES = {1 << power for power in range(10, 31)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    runs = {run['label']: run for run in manifest['runs']}
    labels = {f'{op}_{repeat}_{kind}' for op in OPERATIONS
              for repeat in range(3) for kind in ['reference', 'out', 'in']}
    require(len(manifest['runs']) == 27 and set(runs) == labels,
            'expected exactly 27 unique runs')
    means = {}
    for label, run in runs.items():
        require(run['state'] == 'complete' and run['exit_code'] == 0,
                label + ': incomplete/failed run')
        op, repeat, kind = label.split('_')
        command = run['command']
        reference = runs[f'{op}_{repeat}_reference']['command']
        warmup = int(reference[reference.index('-w') + 1])
        iterations = int(reference[reference.index('-n') + 1])
        require((warmup, iterations) == {'ag': (20, 30), 'ar': (20, 10),
                                        'rs': (10, 10)}[op],
                label + ': original warmup/iteration protocol changed')
        require(reference[reference.index('-b') + 1] == '1024' and
                reference[reference.index('-e') + 1] == '1073741824',
                label + ': reference range changed')
        log = (directory / (label + '.log')).read_text()
        values = collections.defaultdict(dict)
        if kind == 'reference':
            require(int(command[command.index('-np') + 1]) == (4 if op == 'ag' else 8),
                    label + ': reference process layout changed')
            if op == 'ag':
                require(command[command.index('--nccl') + 1] == '0',
                        label + ': native reference control enabled')
            matching = [Path(arg) for arg in command if Path(arg).is_file() and
                        hashlib.sha256(Path(arg).read_bytes()).hexdigest() ==
                        manifest['reference_hashes'][op]]
            require(bool(matching), label + ': original reference binary hash mismatch')
            for line in log.splitlines():
                if not line.startswith('# sample,'):
                    continue
                fields = line.split(',')
                require(fields[1] == OPERATIONS[op][1], label + ': wrong operation')
                size, index = int(fields[2]), int(fields[-3]) - 1
                value, error = float(fields[-2]), float(fields[-1])
                require(index not in values[size], label + ': duplicate sample')
                require(math.isfinite(value) and value > 0 and
                        math.isfinite(error) and 0 <= error <= 1e-5,
                        label + ': invalid timing/error')
                values[size][index] = value
        else:
            require('--strict' in command and int(command[command.index('-np') + 1]) == 8,
                    label + ': not an eight-rank strict candidate')
            delimiter = len(command) - 1 - command[::-1].index('--')
            args = command[delimiter + 2:]
            require(len(args) in [6, 7] and args[:4] ==
                    [op, kind, '1024', '1073741824'] and
                    list(map(int, args[4:6])) == [warmup, iterations] and
                    (len(args) == 6 or args[6] == '1073741824'),
                    label + ': candidate timing protocol changed')
            library = Path(command[command.index('--library') + 1])
            require(hashlib.sha256(library.read_bytes()).hexdigest() ==
                    manifest['candidate_sha256'], label + ': candidate hash mismatch')
            require('UCCL reference-protocol checks passed' in log,
                    label + ': missing full-output success')
            loaded = re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0', log)
            fallback = re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)', log)
            require(sorted(map(int, loaded)) == list(range(8)) and
                    sorted(tuple(map(int, item)) for item in fallback) ==
                    [(rank, 0) for rank in range(8)], label + ': native/fallback assertion')
            routes = collections.Counter(tuple(map(int, item)) for item in re.findall(
                r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)', log))
            require(routes == collections.Counter((rank, OPERATIONS[op][0], size,
                    warmup + iterations) for rank in range(8) for size in SIZES),
                    label + ': route coverage mismatch')
            for line in log.splitlines():
                if not line.startswith('sample,'):
                    continue
                fields = line.split(',')
                require(fields[1:3] == [op, kind], label + ': sample operation/layout')
                size, index = int(fields[3]), int(fields[4])
                value, error = float(fields[5]), float(fields[6])
                require(index not in values[size], label + ': duplicate sample')
                require(math.isfinite(value) and value > 0 and
                        math.isfinite(error) and 0 <= error <= 1e-5,
                        label + ': invalid timing/error')
                values[size][index] = value
        require(set(values) == SIZES and all(set(samples) == set(range(iterations))
                for samples in values.values()), label + ': incomplete sample coverage')
        means[label] = {size: statistics.mean(samples.values())
                        for size, samples in values.items()}
        require(all(means[label][size] == run['means_us'][str(size)] for size in SIZES),
                label + ': stored mean differs from raw, untrimmed samples')
    rows = {}
    for op in OPERATIONS:
        for size in sorted(SIZES):
            reference = statistics.median(means[f'{op}_{r}_reference'][size] for r in range(3))
            for layout in ['out', 'in']:
                candidate = statistics.median(means[f'{op}_{r}_{layout}'][size] for r in range(3))
                rows[op, size, layout] = (reference, candidate, candidate / reference)
    stored = list(csv.DictReader((directory / 'comparison.csv').open()))
    require(len(stored) == 126 and len({(r['operation'], int(r['bytes']), r['layout'])
                                      for r in stored}) == 126,
            'comparison CSV coverage mismatch')
    for row in stored:
        expected = rows[row['operation'], int(row['bytes']), row['layout']]
        require(tuple(float(row[k]) for k in ['reference_us', 'candidate_us', 'ratio']) == expected,
                'comparison CSV differs from recomputed medians')
    slower = [{'operation': op, 'bytes': size, 'layout': layout, 'ratio': value[2]}
              for (op, size, layout), value in rows.items() if value[2] > 1]
    result = json.loads((directory / 'result.json').read_text())
    require(result == {'exit_code': 0, 'rows': 126, 'slower_rows': len(slower)},
            'stored result differs from raw-log audit')
    return {'scope': 'paired protocol only; batch/build/release gates separate',
            'raw_log_checks_passed': True, 'runs': 27, 'rows': 126,
            'zero_slowdown_passed': not slower, 'slower_rows': len(slower),
            'failures': slower, 'candidate_sha256': manifest['candidate_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    target = args.out or args.directory / 'raw_log_audit.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Raw-log audit: {result['slower_rows']}/126 slower rows; "
          f"zero slowdown passed={result['zero_slowdown_passed']}")
    return 0 if result['zero_slowdown_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
