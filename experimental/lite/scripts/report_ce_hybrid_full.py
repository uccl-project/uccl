#!/usr/bin/env python3
"""Combine both sweeps and plot fixed hybrid variants against the FP32 reference."""
import csv
import pathlib
import statistics
import sys
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = pathlib.Path(sys.argv[1])
rounds = defaultdict(list)
timed = 0
for sweep in ['forward', 'reverse']:
    for power in range(10, 31):
        size = 1 << power
        groups = defaultdict(list)
        with (root / sweep / f'{size}.csv').open() as stream:
            for row in csv.DictReader(stream):
                groups[(row['variant'], int(row['round']))].append(float(row['host_us']))
                timed += 1
        assert len(groups) == 18, (sweep, size, len(groups))
        for (variant, _), samples in groups.items():
            assert len(samples) == 10
            rounds[(size, variant)].append(statistics.mean(samples))
summary = {}
with (root / 'summary.csv').open('w') as stream:
    writer = csv.writer(stream)
    writer.writerow(['output_bytes','variant','median_us','min_round_mean_us','max_round_mean_us','reference_speedup'])
    for (size, variant), means in sorted(rounds.items()):
        value = statistics.median(means)
        ref = statistics.median(rounds[(size, 'uccl_fp32_reference')])
        summary[(size, variant)] = value
        writer.writerow([size, variant, value, min(means), max(means), ref/value])

def label(size):
    if size >= 1 << 30: return f'{size >> 30} GiB'
    if size >= 1 << 20: return f'{size >> 20} MiB'
    return f'{size >> 10} KiB'

variants = [('uccl_fp32_reference','UCCL FP32 reference','#222222','-'),
            ('packet_sm','Fixed packet SM','#999999',':'),
            ('nccl','NCCL default','#d39200','--'),
            ('batch_ce_4m','Batch CE / 4 MiB','#8d65aa','--'),
            ('hybrid_grouped_1m','Hybrid / 1 MiB','#1476bd','-'),
            ('hybrid_grouped_4m','Hybrid / 4 MiB','#df5835','-')]
sizes = [1 << p for p in range(10,31)]
fig, axes = plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
for variant, name, color, style in variants:
    values = [summary[(size, variant)] for size in sizes]
    axes[0].plot(sizes,values,label=name,color=color,linestyle=style,linewidth=1.6)
    if variant in ['uccl_fp32_reference','hybrid_grouped_1m','hybrid_grouped_4m']:
        axes[0].fill_between(sizes,[min(rounds[(s,variant)]) for s in sizes],
                            [max(rounds[(s,variant)]) for s in sizes],color=color,alpha=.12)
    if variant.startswith('hybrid'):
        axes[1].plot(sizes,[summary[(s,'uccl_fp32_reference')]/summary[(s,variant)] for s in sizes],
                     marker='o',markersize=3,label=name,color=color)
axes[0].set_yscale('log')
axes[0].set_ylabel('Host enqueue-to-completion latency (µs)')
axes[0].legend(fontsize=8)
axes[1].axhline(1,color='#333333',linestyle='--',linewidth=1)
axes[1].set_ylabel('Reference latency / hybrid latency (higher is better)')
axes[1].legend(fontsize=9)
for ax in axes:
    ax.set_xscale('log',base=2)
    ticks=[1 << p for p in [10,13,16,19,22,25,28,30]]
    ax.set_xticks(ticks,[label(s) for s in ticks],rotation=30,ha='right')
    ax.set_xlabel('Complete AllGather output per GPU')
    ax.grid(True,alpha=.2)
fig.suptitle('4× NVIDIA L4 • FP32 • fixed hybrid variants • fresh forward/reverse sweeps')
fig.savefig(root/'comparison.png',dpi=180)
fig.savefig(root/'comparison.svg')
with (root/'table.md').open('w') as stream:
    stream.write('| Output/GPU | Reference path | UCCL reference µs | Hybrid 1 MiB µs | Speedup | Hybrid 4 MiB µs | Speedup |\n')
    stream.write('|---:|---|---:|---:|---:|---:|---:|\n')
    for size in sizes:
        path='push-serial' if size <= 512*1024 else 'mini-host' if size <= 16*1024*1024 else 'packet'
        ref=summary[(size,'uccl_fp32_reference')]
        one=summary[(size,'hybrid_grouped_1m')]
        four=summary[(size,'hybrid_grouped_4m')]
        stream.write(f'| {label(size)} | {path} | {ref:.2f} | {one:.2f} | {ref/one:.3f}× | {four:.2f} | {ref/four:.3f}× |\n')
print('timed_calls',timed)
for variant in ['hybrid_grouped_1m','hybrid_grouped_4m']:
    wins=[label(s) for s in sizes if summary[(s,variant)] < summary[(s,'uccl_fp32_reference')]]
    print(variant,'wins',len(wins),wins)
