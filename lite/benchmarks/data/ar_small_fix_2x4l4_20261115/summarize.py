"""Summarize fresh AllReduce comparisons; reuse the recorded NCCL control."""
import csv
import math
from pathlib import Path
from statistics import mean, median

DATA = Path(__file__).resolve().parent
rows = []
comparison = []
for mode in ('fixed', 'previous', 'native'):
    rounds = {}
    for repeat in range(1, 4):
        path = DATA / 'raw' / f'{mode}_{repeat}.log'
        samples = {}
        for line in path.read_text().splitlines():
            fields = line.split(',')
            if len(fields) != 6 or fields[:2] != ['# sample', 'allreduce']:
                continue
            size, value = int(fields[2]), float(fields[4])
            assert float(fields[5]) <= 1e-5 and math.isfinite(value) and value > 0, path
            samples.setdefault(size, []).append(value)
        assert set(samples) == {1 << p for p in range(10, 31 if mode == 'fixed' else 21)}, path
        for size, values in sorted(samples.items()):
            assert len(values) == 10, (path, size)
            rounds.setdefault(size, []).append(mean(values))
    for size, values in rounds.items():
        us = median(values)
        record = (mode, size, us, min(values), max(values), 1.75 * size / us / 1000)
        comparison.append(record)
        if mode == 'fixed':
            rows.append(('hybrid', *record[1:]))
with (DATA.parent / 'ar_latency_2x4l4_20261115/summary.csv').open() as source:
    rows.extend(tuple(row.values()) for row in csv.DictReader(source) if row['mode'] == 'nccl')
for name, data in [('summary.csv', rows), ('comparison.csv', comparison)]:
    with (DATA / name).open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('mode', 'input_bytes_per_gpu', 'median_us', 'min_us', 'max_us', 'bus_GBps'))
        writer.writerows(data)
