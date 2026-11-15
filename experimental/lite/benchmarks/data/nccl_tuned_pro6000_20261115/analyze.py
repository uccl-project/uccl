#!/usr/bin/env python3
"""Validate raw final samples and emit the 63-case tuned NCCL figure input."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
expected = {(op, 1 << power) for op in ['allreduce', 'allgather', 'reducescatter'] for power in range(10, 31)}
selection = json.loads((root / 'selection.json').read_text())
assert hashlib.sha256((root / 'schedule.json').read_bytes()).hexdigest() == selection['schedule_sha256']
schedule = {(r['operation'], r['bytes']): r for r in json.loads((root / 'schedule.json').read_text())}
assert set(schedule) == expected and not selection['final_measurements_used']
runs = json.loads((root / 'evaluate_runs.json').read_text())
data = {(kind, repeat): {} for kind in ['tuned', 'default'] for repeat in range(2)}
count = 0
for run in runs:
    assert run['returncode'] == 0 and run['correctness_pass']
    assert run['started_utc'] > selection['frozen_utc']
    name = run['name']
    kind = 'default' if name.startswith('eval_default_') else 'tuned'
    bucket = data[kind, run['repeat']]
    text = (root / (name + '.csv')).read_text()
    assert 'correctness: PASS' in text
    printed = {(r['operation'], int(r['bytes'])): r for r in csv.DictReader(
        line for line in text.splitlines() if line.startswith(('operation,', 'allreduce,', 'allgather,', 'reducescatter,')))}
    with (root / (name + '_samples.csv')).open() as f:
        samples = list(csv.DictReader(f))
    count += len(samples)
    grouped = {}
    for s in samples:
        key = s['operation'], int(s['bytes'])
        ri = int(s['repeat']), int(s['iteration'])
        assert 0 <= ri[0] < 5 and 0 <= ri[1] < 20
        group = grouped.setdefault(key, {})
        assert ri not in group
        group[ri] = float(s['slowest_rank_us'])
    assert set(grouped) == set(printed)
    for key, group in grouped.items():
        assert len(group) == 100 and key not in bucket and printed[key]['correct'] == '1'
        value = median(median(group[r, i] for i in range(20)) for r in range(5))
        assert abs(value - float(printed[key]['median_us'])) < 0.001
        assert value > 0 and math.isfinite(value)
        if kind == 'tuned':
            assert run['config'] == schedule[key]['configuration']
        bucket[key] = value
assert all(set(group) == expected for group in data.values())
assert count == 25200

rows = []
for op, size in sorted(expected):
    key = op, size
    factor = 1.0 if op == 'allreduce' else 0.5
    forward, reverse = data['tuned', 0][key], data['tuned', 1][key]
    nf, nr = data['default', 0][key], data['default', 1][key]
    rows.append(dict(operation=op, bytes=size, configuration=schedule[key]['configuration'],
                     forward_us=forward, reverse_us=reverse,
                     nccl_default_forward_us=nf, nccl_default_reverse_us=nr,
                     central_us=median([forward, reverse]),
                     bus_GB_s=factor * size / median([forward, reverse]) / 1000,
                     default_bus_GB_s=factor * size / median([nf, nr]) / 1000,
                     speedup_vs_fresh_default=median([nf, nr]) / median([forward, reverse])))
with (root / 'cases.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
summary = dict(valid=True, timed_invocations=count, measured_source='b0ac7b4a + archived NCCL-only allocation/window patch',
               selection=selection, operations={})
for op in ['allreduce', 'allgather', 'reducescatter']:
    rs = [r for r in rows if r['operation'] == op]
    summary['operations'][op] = dict(
        full_geomean_speedup=math.exp(sum(math.log(r['speedup_vs_fresh_default']) for r in rs) / len(rs)),
        wins=sum(r['speedup_vs_fresh_default'] > 1 for r in rs),
        large_geomean_speedup=math.exp(sum(math.log(r['speedup_vs_fresh_default']) for r in rs if r['bytes'] >= 1 << 20) / 11),
        one_gib=rs[-1])
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary['operations'], indent=2))
