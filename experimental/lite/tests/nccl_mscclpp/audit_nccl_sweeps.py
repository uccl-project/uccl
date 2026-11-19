#!/usr/bin/env python3
"""Audit the nine frozen nccl-tests sweeps without rerunning GPU work.

This verifies the batch diagnostic, not matched per-call or release admission.
Exit 2 means slower batch rows remain. No samples or layouts are excluded.
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

OPERATIONS = {'ag': (1, 'all_gather_perf'), 'ar': (3, 'all_reduce_perf'),
              'rs': (2, 'reduce_scatter_perf')}
SIZES = {1 << power for power in range(10, 31)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, tests, reference):
    directory, tests, reference = map(Path, (directory, tests, reference))
    summary = json.loads((directory / 'full_diagnostic.json').read_text())
    library_hash = digest(directory / 'nccl_lite/libuccl_lite_nccl_strict.so')
    require(summary['library_sha256'] == library_hash, 'library hash mismatch')
    require(json.loads((directory / 'full_exit.json').read_text()) ==
            {'exit_code': 0}, 'batch campaign did not finish')
    reference_rows = list(csv.DictReader((reference / 'summary.csv').open()))
    identities, slower = {}, []
    for op, (operation, executable) in OPERATIONS.items():
        base = directory / (op + '_measure')
        manifest = json.loads((base / 'manifest.json').read_text())
        command = manifest['command']
        require('--strict' in command and command[command.index('-np') + 1] == '8',
                op + ': expected eight strict ranks')
        require(digest(command[command.index('--library') + 1]) == library_hash,
                op + ': staged library mismatch')
        args = command[command.index('--') + 1:]
        require(Path(args[0]).name == executable and
                digest(args[0]) == digest(tests / executable),
                op + ': nccl-tests executable differs from frozen binary')
        identities[op] = digest(args[0])
        expected_args = ['-b', '1K', '-e', '1073741824', '-f', '2', '-g', '1',
                         '-d', 'float', '-w', '20', '-n', '30', '-c', '1',
                         '-a', '3']
        require(args[1:] == expected_args, op + ': batch protocol changed')
        require(len(manifest['results']) == 3, op + ': expected three sweeps')
        require(json.loads((base / 'exit.json').read_text()) == {'exit_code': 0},
                op + ': sweep failed')
        values = []
        for repeat in range(3):
            label = f'{op}_{repeat}'
            log = (base / f'candidate_{repeat}.log').read_text()
            require(re.search(r'# Out of bounds values\s*:\s*0\s+OK', log),
                    label + ': missing correctness summary')
            ranks = re.findall(r'#\s+Rank\s+(\d+)\s+Group', log)
            require(sorted(map(int, ranks)) == list(range(8)), label + ': ranks')
            loaded = re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0', log)
            fallback = re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)', log)
            require(sorted(map(int, loaded)) == list(range(8)) and
                    sorted(tuple(map(int, item)) for item in fallback) ==
                    [(rank, 0) for rank in range(8)], label + ': native/fallback')
            routes = collections.Counter(tuple(map(int, item)) for item in re.findall(
                r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)', log))
            # Frozen nccl-tests: 20 warmups, then 30 timed + one sync + one
            # correctness call for each of the two layouts = 84 calls/size.
            require(routes == collections.Counter((rank, operation, size, 84)
                    for rank in range(8) for size in SIZES), label + ': routes')
            rows = {}
            for line in log.splitlines():
                fields = line.split()
                if len(fields) != 13 or not fields[0].isdigit():
                    continue
                size = int(fields[0])
                require(size not in rows, label + ': duplicate timing row')
                require(fields[2:4] == ['float', 'none' if op == 'ag' else 'sum']
                        and fields[8] == fields[12] == '0', label + ': dtype/output')
                timings = [float(fields[5]), float(fields[9])]
                require(all(math.isfinite(x) and x > 0 for x in timings),
                        label + ': invalid timing')
                rows[size] = timings
            require(set(rows) == SIZES, label + ': incomplete size coverage')
            recorded = manifest['results'][repeat]['rows_us']
            require(rows == {int(k): v for k, v in recorded.items()},
                    label + ': stored rows differ from raw log')
            values.append(rows)
        refs = {int(r['bytes']): float(r['median_us'])
                for r in reference_rows if r['operation'] == op}
        require(set(refs) == SIZES and all(math.isfinite(x) and x > 0
                for x in refs.values()), op + ': invalid original reference')
        stored = list(csv.DictReader((base / 'comparison.csv').open()))
        require(len(stored) == 42 and len({(r['bytes'], r['layout'])
                for r in stored}) == 42, op + ': comparison coverage')
        ratios = []
        for row in stored:
            size = int(row['bytes'])
            layout = ['out-of-place', 'in-place'].index(row['layout'])
            samples = [v[size][layout] for v in values]
            median = statistics.median(samples)
            expected = (median, min(samples), max(samples), refs[size],
                        median / refs[size])
            require(tuple(float(row[k]) for k in ['median_us', 'min_us',
                    'max_us', 'reference_us', 'diagnostic_ratio']) == expected,
                    op + ': comparison differs from raw medians')
            ratios.append(expected[-1])
            if expected[-1] > 1:
                slower.append({'operation': op, 'bytes': size,
                               'layout': row['layout'], 'ratio': expected[-1]})
        require(summary['operations'][op] == {'rows': 42,
                'slower_rows': sum(x > 1 for x in ratios),
                'worst_ratio': max(ratios)}, op + ': summary mismatch')
    return {'scope': 'batch diagnostic only; not matched or release admission',
            'raw_log_checks_passed': True, 'runs': 9, 'rows': 126,
            'candidate_sha256': library_hash, 'nccl_tests_sha256': identities,
            'slower_rows': len(slower), 'failures': slower}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.tests, args.reference)
    (args.directory / 'batch_raw_log_audit.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(2 if result['slower_rows'] else 0)


if __name__ == '__main__':
    main()
