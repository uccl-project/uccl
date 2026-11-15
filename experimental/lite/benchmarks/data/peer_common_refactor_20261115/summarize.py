#!/usr/bin/env python3
"""Rebuild per-case old/new comparisons from recorded logs; keep every variant."""
import csv
import io
import tarfile
import json
import math
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent
archive = tarfile.open(root / 'measurements.tar.gz', 'r:gz')

def read_text(path):
    if path.exists():
        return path.read_text()
    return archive.extractfile(str(path.relative_to(root))).read().decode()
names = {'all_reduce': 'allreduce', 'all_gather': 'allgather',
         'reduce_scatter': 'reducescatter'}

def rows(path, kind):
    # stdout and stderr share a log. Undo only the known diagnostic lines that
    # can split a buffered CSV row (in particular local RS at 1 MiB).
    text = re.sub(r'(?:all_reduce|all_gather|reduce_scatter) \d+: [^\n]*\n', '',
                  read_text(path))
    text = re.sub(r'DECISION [^\n]*\n', '', text)
    found = {}
    for row in csv.reader(text.splitlines()):
        if not row or row[0] not in set(names) | set(names.values()):
            continue
        expected, index = {'joint': (8, 5), 'local': (9, 4),
                           'fp32': (6, 3)}[kind]
        if len(row) != expected or not row[1].isdigit():
            continue
        key = (names.get(row[0], row[0]), int(row[1]),
               row[3] if kind == 'local' else row[2])
        if key in found:
            raise ValueError(f'Duplicate row in {path}: {key}')
        found[key] = float(row[index])
    return found

experiments = [
    ('pro6000', 'joint', '', 2, 63),
    ('pro6000-small-pinned', 'joint', '', 2, 33),
    ('l41', 'joint', '-joint', 4, 51),
    ('l41', 'fp32', '-fp32', 4, 77),
    ('eight-l4', 'local', '-local', 8, 51),
    ('eight-l4-pinned', 'local', '-local', 8, 51),
]
output = []
for folder, kind, suffix, ranks, count in experiments:
    runs = [rows(root / folder / (label + suffix + '.log'), kind)
            for label in ['0-before', '1-after', '2-after', '3-before']]
    keys = set(runs[0])
    if len(keys) != count or any(set(r) != keys for r in runs):
        raise ValueError(f'Incomplete {folder}/{kind}: {[len(r) for r in runs]}')
    for op, size, variant in sorted(keys):
        key = (op, size, variant)
        before = statistics.mean([runs[0][key], runs[3][key]])
        after = statistics.mean([runs[1][key], runs[2][key]])
        factor = (ranks - 1) / ranks * (2 if op == 'allreduce' else 1)
        output.append(dict(experiment=folder, dtype='FP32' if kind == 'fp32' else 'BF16',
                           collective=op, bytes=size, variant=variant,
                           before_us=before, after_us=after,
                           latency_ratio=after / before,
                           before_bus_gbs=factor * size / before / 1000,
                           after_bus_gbs=factor * size / after / 1000))
with (root / 'cases.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(output[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(output)
summary = []
for group in sorted({(r['experiment'], r['dtype'], r['collective'], r['variant'])
                     for r in output}):
    selected = [r for r in output if
                (r['experiment'], r['dtype'], r['collective'], r['variant']) == group]
    summary.append(dict(zip(['experiment', 'dtype', 'collective', 'variant'], group),
                        cases=len(selected),
                        geomean_latency_ratio=math.exp(statistics.mean(
                            math.log(r['latency_ratio']) for r in selected)),
                        worst_latency_ratio=max(r['latency_ratio'] for r in selected)))
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for s in summary:
    print(s['experiment'], s['dtype'], s['collective'], s['variant'],
          f"geomean {(s['geomean_latency_ratio']-1)*100:+.2f}%, "
          f"worst {(s['worst_latency_ratio']-1)*100:+.2f}%")

# Same-process pairs share input buffers, GPU state, timer and correctness
# checks. Report adjacent-pair ratios, retaining the standalone round medians
# in the raw logs as a separate estimator.
from collections import defaultdict
pairs = defaultdict(list)
for name in sorted(archive.getnames()):
    if not name.startswith('pro6000-in-process/paired-') or not name.endswith('.csv'):
        continue
    for row in csv.DictReader(io.StringIO(read_text(root / name))):
        pairs[(row['collective'], int(row['bytes']))].append(
            (float(row['before_us']), float(row['after_us'])))
if len(pairs) != 33 or any(len(v) != 1350 for v in pairs.values()):
    raise ValueError('Incomplete in-process paired samples')
paired = []
for (op, size), values in sorted(pairs.items()):
    ratios = [after / before for before, after in values]
    paired.append(dict(collective=op, bytes=size, pairs=len(values),
                       median_pair_ratio=statistics.median(ratios),
                       geomean_pair_ratio=math.exp(statistics.mean(
                           math.log(r) for r in ratios)),
                       median_pair_delta_us=statistics.median(
                           after-before for before, after in values)))
(root / 'paired_summary.json').write_text(json.dumps(paired, indent=2) + '\n')
