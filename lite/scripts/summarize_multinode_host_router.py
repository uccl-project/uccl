#!/usr/bin/env python3
"""Validate fixed host-router comparisons and their own NCCL controls."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()
modes = ['gpu', 'legacy', 'early', 'shared', 'shared_ready', 'shared_rdma']
sizes = [1 << n for n in range(20, 31)]
samples = {}
for mode in modes:
    for repetition in range(1, 4):
        path = args.directory / f'{mode}_{repetition}.log'
        content = path.read_text()
        if mode != 'gpu':
            assert '# primed_nccl_calls=10,checked=1' in content, path
            assert 'warmups=10 iterations=10 repeats=1' in content, path
            assert 'paired NCCL version=22807' in content, path
        for line in content.splitlines():
            if not line.startswith('# sample,allgather,'):
                continue
            fields = line.split(',')
            size = int(fields[2])
            if mode == 'gpu':
                label, call, us, error = mode, int(fields[3]), float(fields[4]), float(fields[5])
            else:
                label = f'nccl_{mode}' if fields[3] == 'nccl' else mode
                call, us, error = int(fields[5]), float(fields[6]), float(fields[7])
            assert size in sizes and 1 <= call <= 10 and error == 0 and math.isfinite(us) and us > 0, line
            key = (size, label, repetition)
            values = samples.setdefault(key, {})
            assert call not in values, (path, key, call)
            values[call] = us
labels = modes + [f'nccl_{mode}' for mode in modes if mode != 'gpu']
rows = []
for size in sizes:
    for label in labels:
        runs = []
        for repetition in range(1, 4):
            values = samples.get((size, label, repetition), {})
            assert sorted(values) == list(range(1, 11)), (size, label, repetition)
            runs.append(mean(values.values()))
        us = median(runs)
        rows.append([size, label, f'{us:.3f}', f'{min(runs):.3f}', f'{max(runs):.3f}', f'{size*7/8/us/1000:.3f}'])
assert sum(len(v) for v in samples.values()) == 3630
with (args.directory / 'summary.csv').open('w') as output:
    writer = csv.writer(output)
    writer.writerow(['bytes', 'mode', 'median_us', 'min_us', 'max_us', 'bus_GBps'])
    writer.writerows(rows)
print('Validated 3,630 timed calls, 11 sizes, six configurations and five paired NCCL controls.')
