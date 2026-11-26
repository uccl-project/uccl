"""Summarize exploratory RS screens without promoting an envelope."""
import csv
from pathlib import Path
from statistics import mean

DATA = Path(__file__).resolve().parent
RANGES = {
    'baseline': (10, 30), 'serial': (10, 23), 'direct_medium': (17, 24),
    'launch_96_256': (10, 30), 'launch_192_256': (10, 30),
    'piece16': (25, 30), 'piece32': (25, 30),
    'fused4': (10, 30), 'fused8': (25, 30),
}
rows = []
for mode, (first, last) in RANGES.items():
    path = DATA / 'raw' / f'{mode}.log'
    samples = {}
    for line in path.read_text().splitlines():
        fields = line.split(',')
        if len(fields) == 6 and fields[:2] == ['# sample', 'reducescatter']:
            assert float(fields[5]) <= 1e-5, path
            samples.setdefault(int(fields[2]), []).append(float(fields[4]))
    assert set(samples) == {1 << p for p in range(first, last + 1)}, path
    for size, values in sorted(samples.items()):
        assert len(values) == 5, (path, size)
        rows.append((mode, size, mean(values)))
with (DATA / 'summary.csv').open('w', newline='') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(('mode', 'input_bytes_per_gpu', 'mean_us'))
    writer.writerows(rows)
