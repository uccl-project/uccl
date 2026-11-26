#!/usr/bin/env python3
"""Validate complete paired samples, then plot the measured candidate envelope."""
import collections
import csv
import pathlib
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[1]
data = root / 'benchmarks/data/optimization_4l4_20261113'
samples = collections.defaultdict(lambda: collections.defaultdict(list))
for line in (data / 'rs_full.log').read_text().splitlines():
    if line.startswith('# sample,'):
        _, op, size, backend, run, call, us = line.split(',')
        assert op == 'reduce_scatter'
        samples[int(size), backend][int(run)].append((int(call), float(us)))
rows = []
for size in [1024 << i for i in range(21)]:
    medians = {}
    for (extent, backend), runs in samples.items():
        if extent != size:
            continue
        assert set(runs) == {1, 2, 3}, (size, backend, runs.keys())
        for calls in runs.values():
            assert sorted(c for c, _ in calls) == list(range(1, 11))
        medians[backend] = statistics.median(
            statistics.mean(us for _, us in calls) for calls in runs.values())
    assert {'peer', 'scatter-ring', 'nccl'} <= medians.keys()
    candidate = min((k for k in medians if k != 'nccl'), key=medians.get)
    rows.append(dict(bytes=size, selected=candidate, uccl_us=medians[candidate],
                     peer_us=medians['peer'], ring_us=medians['scatter-ring'],
                     nccl_us=medians['nccl']))
with (data / 'selected.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for key, label in [('uccl_us', 'UCCL Lite selected'), ('nccl_us', 'NCCL 2.28.7')]:
    small = [r for r in rows if r['bytes'] <= 2**20]
    large = [r for r in rows if r['bytes'] >= 2**20]
    axes[0].plot([r['bytes']/1024 for r in small], [r[key] for r in small], 'o-', label=label)
    axes[1].plot([r['bytes']/2**20 for r in large],
                 [.75*r['bytes']/r[key]/1000 for r in large], 'o-', label=label)
for ax in axes:
    ax.set_xscale('log', base=2)
    ax.grid(alpha=.3)
    ax.legend()
axes[0].set(xlabel='Full input per GPU (KiB)', ylabel='Latency (µs)')
axes[1].set(xlabel='Full input per GPU (MiB)', ylabel='Bus bandwidth (GB/s)')
fig.suptitle('4×L4 FP32 ReduceScatter · eager CUDA events · per-size UCCL selection')
fig.tight_layout()
fig.savefig(root/'docs/assets/collectives-4l4-rs-pipeline-20261113.svg')
print('Validated', sum(len(v) for runs in samples.values() for v in runs.values()), 'timed samples')
print(rows[-1])
