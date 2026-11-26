#!/usr/bin/env python3
"""Report fixed medium-message variants; never mix prepared and eager scope."""
import csv
import pathlib
import statistics
import sys
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path(sys.argv[1])
groups=defaultdict(list)
for sweep in ['forward','reverse']:
    for power in range(17,24):
        size=1<<power
        rounds=defaultdict(list)
        with (root/sweep/f'{size}.csv').open() as f:
            for r in csv.DictReader(f):
                rounds[(r['variant'],r['round'])].append((float(r['host_us']),float(r['arm_us'])))
        assert len(rounds)==33
        for (name,_),samples in rounds.items():
            assert len(samples)==20
            groups[(size,name)].append(tuple(statistics.mean(v[i] for v in samples) for i in [0,1]))
data={k:tuple(statistics.median(v[i] for v in values) for i in [0,1]) for k,values in groups.items()}
with (root/'summary.csv').open('w') as f:
    w=csv.writer(f)
    w.writerow(['output_bytes','variant','scope','median_us','min_round_mean_us','max_round_mean_us','arm_us','arm_plus_trigger_us','reference_speedup','timed_calls'])
    for (size,name),means in sorted(groups.items()):
        assert len(means)==6
        latency,arm=data[size,name]
        w.writerow([size,name,'prepared_trigger' if 'armed' in name else 'eager',latency,
                    min(v[0] for v in means),max(v[0] for v in means),arm,
                    statistics.median(sum(v) for v in means),
                    '' if 'armed' in name else data[size,'uccl_fp32_reference'][0]/latency,120])
variants=[('uccl_fp32_reference','UCCL FP32 reference'),('hybrid_grouped_1m','Host hybrid / 1 MiB'),
          ('p2p_xor_batch_ce_fixed4','P2P batch / 4 workers'),('p2p_memop_fixed4','Batch + memop / 4 workers'),
          ('p2p_memop_onewait_fixed4','Batch + memop / one final wait')]
sizes=[1<<p for p in range(17,24)]
def label(s):return f'{s>>20} MiB' if s >= 1<<20 else f'{s>>10} KiB'
with (root/'table.md').open('w') as f:
    f.write('| Output/GPU | '+ ' | '.join(n+' µs' for _,n in variants)+' |\n')
    f.write('|---:|'+ '---:|'*len(variants)+'\n')
    for size in sizes:f.write('| '+label(size)+' | '+' | '.join(f'{data[size,v][0]:.2f}' for v,_ in variants)+' |\n')
with (root/'prepared.md').open('w') as f:
    f.write('| Output/GPU | CE trigger µs | CE arm µs | SM push trigger µs | SM push arm µs |\n|---:|---:|---:|---:|---:|\n')
    for size in sizes:
        ce,ca=data[size,'p2p_memop_armed_ce'];sm,sa=data[size,'p2p_memop_armed_sm']
        f.write(f'| {label(size)} | {ce:.2f} | {ca:.2f} | {sm:.2f} | {sa:.2f} |\n')
fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
for v,name in variants:axes[0].plot(sizes,[data[s,v][0] for s in sizes],marker='o',label=name)
for v,name in [('p2p_memop_armed_ce','CE: trigger only'),('p2p_memop_armed_sm','SM push: trigger only')]:
    axes[1].plot(sizes,[data[s,v][0] for s in sizes],marker='o',label=name)
    axes[1].plot(sizes,[statistics.median(sum(x) for x in groups[s,v]) for s in sizes],linestyle='--',label=name.replace('trigger only','arm + trigger'))
for ax in axes:
    ax.set_xscale('log',base=2);ax.set_yscale('log')
    ax.set_xticks(sizes,[label(s) for s in sizes],rotation=25)
    ax.set_xlabel('Complete FP32 output/GPU');ax.set_ylabel('Latency (µs)');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0].set_title('Ordinary enqueue through completion')
axes[1].set_title('Prepared trigger scope: not an ordinary-call speedup')
fig.suptitle('4× L4 • batch P2P + stream memops • fresh forward/reverse sweeps')
fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.svg')
print((root/'table.md').read_text());print((root/'prepared.md').read_text())
