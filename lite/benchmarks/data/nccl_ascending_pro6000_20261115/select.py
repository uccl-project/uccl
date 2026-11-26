#!/usr/bin/env python3
"""Freeze ascending-sweep choices before independent final measurement."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
configs = json.loads((root / 'candidates.json').read_text())
runs = json.loads((root / 'calibrate_runs.json').read_text())
measurements = {}
for run in runs:
    if not run['correctness_pass']:
        continue
    text = (root / (run['name'] + '.csv')).read_text()
    rows = list(csv.DictReader(line for line in text.splitlines()
                              if line.startswith(('operation,', 'allreduce,', 'allgather,', 'reducescatter,'))))
    assert len(rows) == 21 and all(r['correct'] == '1' and r['operation'] == run['op'] for r in rows)
    assert [int(r['bytes']) for r in rows] == [1 << i for i in range(10, 31)]
    measurements[run['op'], run['config'], run['repeat']] = {int(r['bytes']): float(r['median_us']) for r in rows}
schedule, bulk = [], {}
for op in ['allreduce', 'allgather', 'reducescatter']:
    eligible = [name for name in configs if all((op, name, r) in measurements for r in range(2))]
    assert 'default' in eligible
    def score(name):
        return sum(math.log(measurements[op, name, r][1 << power]) for r in range(2) for power in range(24, 31)) / 14
    best = min(eligible, key=score)
    bulk[op] = best
    for power in range(10, 31):
        size = 1 << power
        ratios = [measurements[op, 'default', r][size] / measurements[op, best, r][size] for r in range(2)]
        # Fixed bulk threshold and gain gate; no final measurement is consulted.
        choice = best if size >= 16 << 20 and min(ratios) >= 1.05 else 'default'
        schedule.append(dict(operation=op, bytes=size, configuration=choice,
                             calibration_speedups=ratios, settings=configs[choice]))
(root / 'schedule.json').write_text(json.dumps(schedule, indent=2) + '\n')
(root / 'selection.json').write_text(json.dumps(dict(
    frozen_utc=datetime.now(timezone.utc).isoformat(), bulk_configurations=bulk,
    rule='Choose one configuration per operation by geometric-mean calibration latency at 16 MiB–1 GiB; select it only at >=16 MiB keys with >=5% gains in both calibrations, otherwise default.',
    schedule_sha256=hashlib.sha256((root / 'schedule.json').read_bytes()).hexdigest(),
    calibration_sha256=hashlib.sha256((root / 'calibrate_runs.json').read_bytes()).hexdigest(),
    final_measurements_used=False, size_order='ascending in every process'), indent=2) + '\n')
print(json.dumps(bulk, indent=2))
