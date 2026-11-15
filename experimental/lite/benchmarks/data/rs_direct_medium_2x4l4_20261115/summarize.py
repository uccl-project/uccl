"""Require complete paired full-range ReduceScatter measurements."""
import csv
import math
from pathlib import Path
from statistics import mean, median

DATA = Path(__file__).resolve().parent
rows = []
for mode in ('baseline', 'candidate'):
    rounds = {}
    for repeat in range(1, 4):
        path = DATA / 'raw' / f'{mode}_{repeat}.log'
        samples = {}
        for line in path.read_text().splitlines():
            fields = line.split(',')
            if len(fields) != 6 or fields[:2] != ['# sample', 'reducescatter']:
                continue
            size, value = int(fields[2]), float(fields[4])
            assert float(fields[5]) <= 1e-5 and math.isfinite(value) and value > 0, path
            samples.setdefault(size, []).append(value)
        assert set(samples) == {1 << p for p in range(10, 31)}, path
        for size, values in sorted(samples.items()):
            assert len(values) == 10, (path, size)
            rounds.setdefault(size, []).append(mean(values))
    for size, values in rounds.items():
        us = median(values)
        rows.append((mode, size, us, min(values), max(values), 0.875 * size / us / 1000))
header = ('mode', 'input_bytes_per_gpu', 'median_us', 'min_us', 'max_us', 'bus_GBps')
with (DATA / 'comparison.csv').open('w', newline='') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
plot_rows = [row for row in rows if row[0] == 'candidate']
with (DATA.parent / 'nccl_tuned_2x4l4_20261111/tuned_summary.csv').open() as source:
    for row in csv.DictReader(source):
        if row['collective'] == 'reducescatter':
            size, us = int(row['bytes']), float(row['nccl_median_us'])
            plot_rows.append(('nccl', size, us, row['nccl_min_us'], row['nccl_max_us'],
                              0.875 * size / us / 1000))
assert len(plot_rows) == 42
with (DATA / 'summary.csv').open('w', newline='') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(plot_rows)
