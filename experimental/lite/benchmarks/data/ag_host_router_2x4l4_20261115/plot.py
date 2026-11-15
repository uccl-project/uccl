"""Fixed configurations; target markers are the user's same-metric reference."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'validated/summary.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for mode,label in [('gpu','Existing GPU policy (8 processes)'),('legacy','Original host pipeline'),('shared','Shared packets, step progression'),('shared_ready','Shared packets, ready progression'),('nccl_shared_ready','Paired NCCL Ring+Simple (4 processes)')]:
 selected=sorted([r for r in rows if r['mode']==mode],key=lambda r:int(r['bytes']))
 x=[int(r['bytes'])/2**20 for r in selected]
 y=[float(r['median_us'])/1000 for r in selected]
 err=[[(float(r['median_us'])-float(r['min_us']))/1000 for r in selected],[(float(r['max_us'])-float(r['median_us']))/1000 for r in selected]]
 axes[0].errorbar(x,y,yerr=err,marker='o',markersize=3,capsize=2,label=label)
 axes[1].plot(x,[float(r['bus_GBps']) for r in selected],marker='o',markersize=3,label=label)
axes[1].plot([1,2,4,8],[12,14,16.3,17.5],'kx--',label='User reference')
for ax in axes:
 ax.set_xscale('log',base=2)
 ax.set_xticks([1,4,16,64,256,1024],labels=['1','4','16','64','256','1024'])
 ax.set_xlabel('Complete output per GPU (MiB)');ax.grid(True,alpha=.25)
axes[0].set_yscale('log');axes[0].set_ylabel('Host enqueue-to-completion latency (ms)')
axes[1].set_ylabel('Bus bandwidth (GB/s), 7N/(8t)');axes[1].legend(fontsize=7)
fig.suptitle('2 × 4 NVIDIA L4 · FP32 AllGather · 10 warmups + 10 calls × 3 rounds')
fig.tight_layout();fig.savefig(root/'results.png',dpi=180)
