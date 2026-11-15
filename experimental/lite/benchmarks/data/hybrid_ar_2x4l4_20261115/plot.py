"""Fixed hybrid configuration against fresh tuned GPU and NCCL controls."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'summary.csv').open()))
fig,axes=plt.subplots(2,1,figsize=(4.6,6))
styles=[('hybrid','UCCL Lite: GPU RS + host AG','#0068b5','-',2.6),
        ('old','UCCL Lite: previous GPU path','#7a858f','-.',1.7),
        ('nccl','NCCL Ring+Simple','#d65a00','--',1.8)]
for mode,label,color,ls,lw in styles:
    selected=sorted((r for r in rows if r['mode']==mode),key=lambda r:int(r['input_bytes_per_gpu']))
    small=[r for r in selected if int(r['input_bytes_per_gpu'])<=8*2**20]
    x=[int(r['input_bytes_per_gpu'])/2**20 for r in small]
    y=[float(r['median_us']) for r in small]
    err=[[float(r['median_us'])-float(r['min_us']) for r in small],
         [float(r['max_us'])-float(r['median_us']) for r in small]]
    axes[0].errorbar(x,y,yerr=err,color=color,linestyle=ls,linewidth=lw,marker='o',markersize=3,capsize=2,label=label)
    axes[1].plot([int(r['input_bytes_per_gpu'])/2**20 for r in selected],
                 [float(r['bus_GBps']) for r in selected],color=color,linestyle=ls,linewidth=lw,marker='o',markersize=3,label=label)
for ax in axes:
    ax.set_xscale('log',base=2)
    ax.grid(True,color='#e6edf2',linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_xlabel('Input per GPU (MiB)',fontsize=9)
axes[0].set_xticks([1,2,4,8],labels=['1','2','4','8'])
axes[0].set_ylabel('Latency (µs)',fontsize=9)
axes[0].set_ylim(0,1050)
axes[0].set_title('1–8 MiB latency',fontsize=10)
axes[1].set_xticks([1,4,16,64,256,1024],labels=['1','4','16','64','256','1024'])
axes[1].set_ylabel('Bus bandwidth (GB/s)',fontsize=9)
axes[1].set_ylim(0,22)
axes[1].set_title('1 MiB–1 GiB throughput',fontsize=10)
axes[1].legend(loc='lower right',fontsize=6.7,frameon=False)
fig.suptitle('AllReduce · 2 × 4 NVIDIA L4',fontsize=12,y=.98)
fig.text(.5,.936,'FP32 · host enqueue-to-completion · three rounds',ha='center',fontsize=7.5)
fig.text(.5,.025,'20 warmups + 10 checked calls · fixed 256-KiB host pieces\nBus BW = 7N/(4t); N is input bytes/GPU',ha='center',fontsize=7)
fig.subplots_adjust(left=.15,right=.975,top=.865,bottom=.15,hspace=.53)
fig.savefig(root/'results.svg')
fig.savefig(root/'results.png',dpi=180)
fig.savefig(root.parents[2]/'docs/assets/collectives-2x4l4-allreduce-hybrid.svg')

# Keep generated SVGs clean in source diffs.
for path in (root/'results.svg', root.parents[2]/'docs/assets/collectives-2x4l4-allreduce-hybrid.svg'):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
