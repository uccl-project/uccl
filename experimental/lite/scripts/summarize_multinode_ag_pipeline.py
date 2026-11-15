#!/usr/bin/env python3
"""Summarize checked per-call samples; no best-variant curve selection."""
import argparse
import csv
import re
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('--modes', nargs='+', default=[
    'serial', 'ring', 'flat', 'sm4096', 'ce1024', 'ce4096', 'pair4096',
    'nccl_default', 'nccl_simple'])
parser.add_argument('--min-bytes', type=int)
parser.add_argument('--max-bytes', type=int)
args = parser.parse_args()
rows = {}
for path in sorted(args.directory.glob('*.log')):
    match = re.fullmatch(r'(serial|ring|flat|sm4096|ce1024|ce4096|pair4096|nccl_default|nccl_simple)_([123])', path.stem)
    if not match:
        continue
    mode, repetition = match.groups()
    samples = {}
    for line in path.read_text().splitlines():
        fields = line.split(',')
        if line.startswith('# sample,allgather,'):
            size, latency, error = int(fields[2]), float(fields[4]), float(fields[5])
        elif line.startswith('nccl,timed,'):
            size, latency, error = int(fields[3]), float(fields[4]), float(fields[5])
        else:
            continue
        if error != 0:
            raise ValueError(f'{path}: nonzero check error: {line}')
        samples.setdefault(size, []).append(latency)
    if not samples:
        raise ValueError(f'{path}: no samples')
    for size, values in samples.items():
        if len(values) != 20:
            raise ValueError(f'{path}: {size}: expected 20 samples, got {len(values)}')
        rows.setdefault((size, mode), {})[repetition] = statistics.mean(values)
if not rows:
    raise ValueError('No comparison logs found')
expected = set(args.modes)
if args.min_bytes is not None or args.max_bytes is not None:
    if (args.min_bytes is None or args.max_bytes is None or
            args.min_bytes < 1 or args.max_bytes < args.min_bytes):
        raise ValueError('Specify a valid positive --min-bytes and --max-bytes together')
    sizes = set()
    size = args.min_bytes
    while size <= args.max_bytes:
        sizes.add(size)
        size *= 2
    if {size for size, _ in rows} != sizes:
        raise ValueError('Missing or unexpected message sizes')
for size in {size for size, _ in rows}:
    if {mode for n, mode in rows if n == size} != expected:
        raise ValueError(f'{size}: missing comparison modes')
    for mode in expected:
        if len(rows[size, mode]) != 3:
            raise ValueError(f'{size}/{mode}: missing repetitions')
with (args.directory / 'summary.csv').open('w') as output:
    writer = csv.writer(output)
    writer.writerow(['bytes', 'mode', 'rounds', 'median_us', 'min_us', 'max_us'])
    for (size, mode), repetitions in sorted(rows.items()):
        if len(repetitions) != 3:
            raise ValueError(f'{size}/{mode}: missing repetitions')
        values = list(repetitions.values())
        writer.writerow([size, mode, 3, *[f'{v:.3f}' for v in
                         (statistics.median(values), min(values), max(values))]])
