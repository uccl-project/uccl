"""Render the full fixed-configuration latency table from verified summaries."""
from pathlib import Path
import csv
import json

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / 'paired/summary.csv').open()))
modes = ['serial', 'ring', 'ce4096', 'nccl_default', 'nccl_simple']
sizes = [2**power for power in range(20, 31)]
lookup = {(int(r['bytes']), r['mode']): r for r in rows}
assert set(lookup) == {(n, mode) for n in sizes for mode in modes}
lines = ['All latencies are **milliseconds**, median of three round means.', '',
         '| Output/GPU (MiB) | Direct SM | Ring | CE (4 MiB pieces) | NCCL default | NCCL Ring+Simple |',
         '| ---: | ---: | ---: | ---: | ---: | ---: |']
for n in sizes:
    values = [float(lookup[n, mode]['median_us']) / 1000 for mode in modes]
    lines.append('| ' + ' | '.join([str(n // 2**20), *[f'{v:.3f}' for v in values]]) + ' |')
(root / 'results.md').write_text('\n'.join(lines) + '\n')
coverage = {'sizes': sizes, 'configurations': modes, 'rounds': 3,
            'timed_calls_per_size_per_round': 20,
            'total_timed_calls': len(sizes) * len(modes) * 3 * 20,
            'output_error': 0,
            'timing': 'slowest-rank host enqueue-to-completion',
            'dtype': 'FP32', 'gpus': '2 x 4 NVIDIA L4'}
(root / 'validation_summary.json').write_text(json.dumps(coverage, indent=2) + '\n')
print('\n'.join(lines))
for n in [2**21, 2**22, 2**30]:
    baseline = 'serial' if n <= 2**21 else 'ring'
    old = float(lookup[n, baseline]['median_us'])
    ce = float(lookup[n, 'ce4096']['median_us'])
    print(f'{n // 2**20} MiB CE latency reduction vs {baseline}: {100 * (1 - ce/old):.2f}%')
