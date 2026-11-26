#!/usr/bin/env python3
"""Summarize checked host-clock samples; keep each fixed variant separate."""
import csv
import pathlib
import statistics
import sys
from collections import defaultdict

root = pathlib.Path(sys.argv[1])
rows = []
for path in root.glob('*.csv'):
    if not path.stem.removeprefix('confirm_').isdigit():
        continue
    groups = defaultdict(lambda: defaultdict(list))
    with path.open() as stream:
        enqueue = defaultdict(lambda: defaultdict(list))
        for row in csv.DictReader(stream):
            groups[(int(row['bytes']), row['variant'])][int(row['round'])].append(
                float(row['host_us']))
            if row.get('enqueue_us'):
                enqueue[(int(row['bytes']), row['variant'])][int(row['round'])].append(float(row['enqueue_us']))
    for (size, variant), rounds in groups.items():
        means = [statistics.mean(samples) for samples in rounds.values()]
        rows.append((path.name, size, variant, sum(map(len, rounds.values())),
                     statistics.median(means), min(means), max(means),
                     statistics.median(statistics.mean(v) for v in enqueue[(size, variant)].values())
                     if enqueue[(size, variant)] else ''))
with (root / 'summary.csv').open('w') as stream:
    writer = csv.writer(stream)
    writer.writerow(['file', 'output_bytes', 'variant', 'calls',
                     'median_round_mean_us', 'min_round_mean_us', 'max_round_mean_us',
                     'median_round_enqueue_us'])
    writer.writerows(sorted(rows))
