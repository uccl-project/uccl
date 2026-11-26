#!/usr/bin/env python3
"""Reconstruct ascending results, preserving all raw fixed-config timings."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
selection = json.loads((root / 'selection.json').read_text())
assert hashlib.sha256((root / 'schedule.json').read_bytes()).hexdigest() == selection['schedule_sha256']
schedule = json.loads((root / 'schedule.json').read_text())
times, timed_calls = {}, 0
for run in json.loads((root / 'evaluate_runs.json').read_text()):
    assert run['correctness_pass'] and run['returncode'] == 0
    assert run['size_order'] == 'ascending' and '--reverse' not in run['command']
    assert run['started_utc'] > selection['frozen_utc']
    text = (root / (run['name'] + '.csv')).read_text()
    assert 'correctness: PASS' in text
    printed = list(csv.DictReader(line for line in text.splitlines()
                                 if line.startswith(('operation,', 'allreduce,', 'allgather,', 'reducescatter,'))))
    assert [int(r['bytes']) for r in printed] == [1 << i for i in range(10, 31)]
    with (root / (run['name'] + '_samples.csv')).open() as stream:
        samples = list(csv.DictReader(stream))
    assert len(samples) == 2100
    assert [int(s['bytes']) for s in samples] == sorted(int(s['bytes']) for s in samples)
    timed_calls += len(samples)
    for row in printed:
        size = int(row['bytes'])
        group = [s for s in samples if int(s['bytes']) == size]
        assert len({(s['repeat'], s['iteration']) for s in group}) == 100
        value = median(median(float(s['slowest_rank_us']) for s in group if int(s['repeat']) == r) for r in range(5))
        assert abs(value - float(row['median_us'])) < 0.001 and row['correct'] == '1'
        times[run['op'], run['config'], run['repeat'], size] = value
rows, raw_rows = [], []
for row in schedule:
    op, size, config = row['operation'], row['bytes'], row['configuration']
    factor = 1.0 if op == 'allreduce' else 0.5
    bulk = selection['bulk_configurations'][op]
    us = [times[op, config, r, size] for r in range(2)]
    defaults = [times[op, 'default', r, size] for r in range(2)]
    raw = [times[op, bulk, r, size] for r in range(2)]
    rows.append(dict(operation=op, bytes=size, configuration=config,
                     run1_us=us[0], run2_us=us[1], median_us=median(us),
                     bus_GB_s=factor*size/median(us)/1000,
                     default_bus_GB_s=factor*size/median(defaults)/1000,
                     speedup_vs_default=median(defaults)/median(us)))
    raw_rows.append(dict(operation=op, bytes=size, configuration=bulk,
                         run1_us=raw[0], run2_us=raw[1], median_us=median(raw),
                         bus_GB_s=factor*size/median(raw)/1000,
                         speedup_vs_default=median(defaults)/median(raw)))
for name, values in [('cases.csv', rows), ('fixed_config_cases.csv', raw_rows)]:
    with (root / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)
summary = dict(valid=True, timed_calls=timed_calls, size_order='ascending only',
               bulk_configurations=selection['bulk_configurations'],
               one_gib=[r for r in rows if r['bytes'] == 1 << 30],
               caveat='Each process executes every size from 1 KiB upward; no largest-first validation is claimed.')
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
