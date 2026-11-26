"""Separate sweeps; no interpolation across the allocation/geometry boundary."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
fig, axes=plt.subplots(1,2,figsize=(12,4.5))
labels={'auto':'Existing policy','fanout_sm':'Fan-out SM','fanout_ce':'Fan-out CE',
        'dual_sm':'Dual RDMA SM','dual_ce':'Dual RDMA CE',
        'matched_sm':'Matched SM','matched_ce':'Matched CE'}
for ax,sweep in zip(axes,['small','large']):
    rows=list(csv.DictReader((root/sweep/'summary.csv').open()))
    modes=['auto','fanout_sm','dual_sm','matched_sm'] if sweep=='small' else list(labels)
    scale=1 if sweep=='small' else 1000
    for mode in modes:
        items=sorted([r for r in rows if r['mode']==mode],key=lambda r:int(r['bytes']))
        x=[int(r['bytes'])/(1024 if sweep=='small' else 2**20) for r in items]
        y=[float(r['median_us'])/scale for r in items]
        err=[[(float(r['median_us'])-float(r['min_us']))/scale for r in items],
             [(float(r['max_us'])-float(r['median_us']))/scale for r in items]]
        ax.errorbar(x,y,yerr=err,label=labels[mode],marker='o',markersize=3,capsize=2)
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.grid(True,alpha=.25)
    ax.set_xlabel('Complete output/GPU ('+('KiB' if sweep=='small' else 'MiB')+')')
    ax.set_ylabel('Host latency ('+('µs' if sweep=='small' else 'ms')+')')
    ax.set_title('Small: all new paths use 1 CTA' if sweep=='small' else 'Large: matched 16 CTAs, other new paths 1 CTA')
    ax.legend(fontsize=7)
fig.suptitle('2 × 4 L4 · FP32 AllGather · three rounds · error bars: round range')
fig.tight_layout();fig.savefig(root/'results.png',dpi=180)
