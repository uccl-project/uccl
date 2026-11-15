"""Render separate, fixed-configuration small/large comparison tables."""
from pathlib import Path
import csv

root = Path(__file__).resolve().parent
modes = ['auto', 'fanout_sm', 'fanout_ce', 'dual_sm', 'dual_ce', 'matched_sm', 'matched_ce']
for sweep, divisor, unit in [('small', 1, 'µs'), ('large', 1000, 'ms')]:
    path = root / sweep / 'summary.csv'
    rows = list(csv.DictReader(path.open()))
    data = {(int(r['bytes']), r['mode']): r for r in rows}
    sizes = sorted({key[0] for key in data})
    assert len(rows) == len(sizes)*len(modes)
    lines = [f'Latency in **{unit}**, median of three round means.', '',
             '| Output/GPU | Existing policy | Fan-out SM | Fan-out CE | Dual RDMA SM | Dual RDMA CE | Matched SM | Matched CE |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for size in sizes:
        label = f'{size//1024} KiB' if size < 2**20 else f'{size//2**20} MiB'
        vals = [float(data[size,mode]['median_us'])/divisor for mode in modes]
        digits = 2 if sweep == 'small' else 3
        lines.append('| ' + ' | '.join([label, *[f'{v:.{digits}f}' for v in vals]]) + ' |')
    (root / f'{sweep}_results.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
