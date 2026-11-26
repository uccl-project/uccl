#!/usr/bin/env python3
"""Freeze per-case NCCL choices using two calibration sweeps, never final data."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
configs = json.loads((root / 'candidates.json').read_text())
records = json.loads((root / 'calibrate_runs.json').read_text())
by_name = {r['name']: r for r in records}
if (root / 'supplement_runs.json').exists():
    by_name.update({r['name']: r for r in json.loads((root / 'supplement_runs.json').read_text())})
expected = {(op, 1 << power) for op in ['allreduce', 'allgather', 'reducescatter'] for power in range(10, 31)}

def read(name, operation=None):
    assert by_name[name]['correctness_pass'] and by_name[name]['returncode'] == 0
    text = (root / (name + '.csv')).read_text()
    rows = list(csv.DictReader(line for line in text.splitlines()
                              if line.startswith(('operation,', 'allreduce,', 'allgather,', 'reducescatter,'))))
    target = expected if operation is None else {k for k in expected if k[0] == operation}
    assert len(rows) == len(target) and all(r['correct'] == '1' for r in rows)
    data = {(r['operation'], int(r['bytes'])): float(r['median_us']) for r in rows}
    assert set(data) == target
    return data

baseline = []
for repeat in range(2):
    before, after = [read(f'cal_default_{position}_{repeat}') for position in ['before', 'after']]
    baseline.append({key: median([before[key], after[key]]) for key in expected})
measurements = {'default': baseline}
for name in configs:
    if name == 'default':
        continue
    values = []
    for repeat in range(2):
        run_name = f'cal_{name}_{repeat}'
        if by_name[run_name]['correctness_pass']:
            values.append(read(run_name))
        else:
            recovered = {}
            for op in ['allreduce', 'allgather']:
                supplement = f'supp_{name}_{op}_{repeat}'
                if supplement in by_name and by_name[supplement]['correctness_pass']:
                    recovered.update(read(supplement, op))
            values.append(recovered)
    measurements[name] = values

schedule = []
all_cases = []
for op, size in sorted(expected):
    key = (op, size)
    eligible = ['default']
    for name, runs in measurements.items():
        if any(key not in run for run in runs):
            continue
        ratios = [baseline[r][key] / runs[r][key] for r in range(2)]
        all_cases.append(dict(operation=op, bytes=size, configuration=name,
                              forward_us=runs[0][key], reverse_us=runs[1][key],
                              forward_speedup=ratios[0], reverse_speedup=ratios[1]))
        # Require at least a 5% speedup in both separately ordered calibration runs.
        if name != 'default' and min(ratios) >= 1.05:
            eligible.append(name)
    winner = min(eligible, key=lambda name: math.sqrt(measurements[name][0][key] * measurements[name][1][key]))
    schedule.append(dict(operation=op, bytes=size, configuration=winner, settings=configs[winner],
                         calibration_forward_speedup=baseline[0][key] / measurements[winner][0][key],
                         calibration_reverse_speedup=baseline[1][key] / measurements[winner][1][key]))
with (root / 'calibration_cases.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(all_cases[0]))
    w.writeheader()
    w.writerows(all_cases)
(root / 'schedule.json').write_text(json.dumps(schedule, indent=2) + '\n')
(root / 'selection.json').write_text(json.dumps(dict(
    frozen_utc=datetime.now(timezone.utc).isoformat(),
    rule='Choose lowest geometric-mean calibration latency among configurations that beat both bracketed default controls by >=5%; otherwise keep default.',
    schedule_sha256=hashlib.sha256((root / 'schedule.json').read_bytes()).hexdigest(),
    calibration_runs_sha256=hashlib.sha256((root / 'calibrate_runs.json').read_bytes()).hexdigest(),
    supplement_runs_sha256=(hashlib.sha256((root / 'supplement_runs.json').read_bytes()).hexdigest()
                            if (root / 'supplement_runs.json').exists() else None),
    final_measurements_used=False), indent=2) + '\n')
for op in ['allreduce', 'allgather', 'reducescatter']:
    counts = {}
    for row in schedule:
        if row['operation'] == op:
            counts[row['configuration']] = counts.get(row['configuration'], 0) + 1
    print(op, counts)
