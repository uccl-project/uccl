#!/usr/bin/env python3
"""Validate the final AG/graph sweeps and publish their complete measurements."""
import collections
import csv
import pathlib
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[1]
data = root / 'benchmarks/data/optimization_4l4_20261113'

def summarize(filename, sizes, operations, output):
    samples = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in (data / filename).read_text().splitlines():
        if line.startswith('# sample,'):
            _, op, size, backend, run, call, us = line.split(',')
            samples[op, int(size), backend][int(run)].append((int(call), float(us)))
    assert {(op, size) for op, size, _ in samples} == {
        (op, size) for op in operations for size in sizes}
    rows = []
    for op in operations:
        for size in sizes:
            medians = {}
            for (operation, extent, backend), runs in samples.items():
                if (operation, extent) != (op, size):
                    continue
                assert set(runs) == {1, 2, 3}
                for calls in runs.values():
                    assert sorted(c for c, _ in calls) == list(range(1, 11))
                medians[backend] = statistics.median(
                    statistics.mean(us for _, us in calls) for calls in runs.values())
            assert 'nccl' in medians and len(medians) > 1
            best = min((k for k in medians if k != 'nccl'), key=medians.get)
            rows.append(dict(operation=op, bytes=size, selected=best,
                             uccl_us=medians[best], nccl_us=medians['nccl']))
    with (data / output).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(filename, 'validated samples=', sum(len(v) for runs in samples.values() for v in runs.values()))
    return rows

ag = summarize('ag_final_full.log', [1024 << i for i in range(21)],
               ['all_gather'], 'ag_selected.csv')
summarize('graph_full.log', [1024 << i for i in range(11)],
          ['all_gather', 'reduce_scatter'], 'graph_selected.csv')
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for key, label in [('uccl_us', 'UCCL Lite selected'), ('nccl_us', 'NCCL 2.28.7')]:
    small = [r for r in ag if r['bytes'] <= 2**20]
    large = [r for r in ag if r['bytes'] >= 2**20]
    axes[0].plot([r['bytes']/1024 for r in small], [r[key] for r in small], 'o-', label=label)
    axes[1].plot([r['bytes']/2**20 for r in large],
                 [.75*r['bytes']/r[key]/1000 for r in large], 'o-', label=label)
for ax in axes:
    ax.set_xscale('log', base=2)
    ax.grid(alpha=.3)
    ax.legend()
axes[0].set(xlabel='Full output per GPU (KiB)', ylabel='Latency (µs)')
axes[1].set(xlabel='Full output per GPU (MiB)', ylabel='Bus bandwidth (GB/s)')
fig.suptitle('4×L4 FP32 AllGather · eager CUDA events · per-size UCCL selection')
fig.tight_layout()
fig.savefig(root/'docs/assets/collectives-4l4-ag-latest-20261113.svg')
print('1 GiB AG:', ag[-1])

ar = summarize('hybrid_final.log', [16777216 << i for i in range(7)],
               ['all_reduce'], 'ar_hybrid_summary.csv')
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for key, label in [('uccl_us', 'UCCL Lite ring + packet (fixed)'), ('nccl_us', 'NCCL 2.28.7')]:
    axes[0].plot([r['bytes']/2**20 for r in ar], [r[key]/1000 for r in ar], 'o-', label=label)
    axes[1].plot([r['bytes']/2**20 for r in ar],
                 [1.5*r['bytes']/r[key]/1000 for r in ar], 'o-', label=label)
for ax in axes:
    ax.set_xscale('log', base=2)
    ax.grid(alpha=.3)
    ax.legend()
axes[0].set(xlabel='Full input per GPU (MiB)', ylabel='Latency (ms)')
axes[1].set(xlabel='Full input per GPU (MiB)', ylabel='Bus bandwidth (GB/s)')
fig.suptitle('4×L4 FP32 in-place AllReduce · experimental hybrid · eager CUDA events')
fig.tight_layout()
fig.savefig(root/'docs/assets/collectives-4l4-ar-hybrid-20261113.svg')
print('1 GiB hybrid AR:', ar[-1])
