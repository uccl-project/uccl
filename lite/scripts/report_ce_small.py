#!/usr/bin/env python3
import csv
import pathlib
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader((root/'confirm'/'summary.csv').open()))
data = {(int(r['output_bytes']),r['variant']):float(r['median_round_mean_us']) for r in rows}
variants = [('uccl_fp32_reference','UCCL FP32 reference'),
            ('hybrid_grouped_1m','Original hybrid'),('sp_gpu_32','Single packet / GPU waits'),
            ('sp_cpu_32','CPU completion / 32 CTAs'),('sp_cpu_8','CPU completion / 8 CTAs'),
            ('sp_threads_8','Two submitters / 8 CTAs')]
sizes = sorted({s for s,v in data})
def label(s): return f'{s>>20} MiB' if s >= 1<<20 else f'{s>>10} KiB'
with (root/'table.md').open('w') as f:
    f.write('| Output/GPU | '+ ' | '.join(v for _,v in variants)+' |\n')
    f.write('|---:|'+ '---:|'*len(variants)+'\n')
    for size in sizes:
        f.write('| '+label(size)+' | '+' | '.join(f'{data[size,v]:.2f}' for v,_ in variants)+' |\n')
fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
for v,name in variants:
    axes[0].plot(sizes,[data[s,v] for s in sizes],marker='o',markersize=3,label=name)
axes[0].set_xscale('log',base=2); axes[0].set_yscale('log')
ticks=[1<<p for p in [10,13,16,19,21,23]]
axes[0].set_xticks(ticks,[label(s) for s in ticks],rotation=25)
axes[0].set_xlabel('Complete output/GPU'); axes[0].set_ylabel('Host enqueue-to-completion (µs)')
axes[0].grid(alpha=.25); axes[0].legend(fontsize=8)
y=list(range(len(variants)))
axes[1].barh(y,[data[1024,v] for v,_ in variants])
axes[1].set_yticks(y,[name for _,name in variants],fontsize=9); axes[1].invert_yaxis()
for i,(v,_) in enumerate(variants):axes[1].text(data[1024,v]+.5,i,f'{data[1024,v]:.2f}',va='center')
axes[1].set_xlim(0,62); axes[1].set_xlabel('1 KiB output latency (µs)'); axes[1].grid(axis='x',alpha=.25)
fig.suptitle('4× L4 FP32 • single-packet ablation • fresh confirmation, fixed configurations')
fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.svg')
