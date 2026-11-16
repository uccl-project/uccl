#!/usr/bin/env python3
"""Refresh only AllGather latency; retain the archived throughput points."""
import csv
import hashlib
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from figure_style import clean_axis, collective_panels, configure, save
DATA = Path(__file__).resolve().parent
STYLES = (
    ('uccl','UCCL Lite','#0068B5','o','-'),
    ('prior','NCCL (default)','#D8A17A','s',(0,(5,2.5))),
    ('sm','NCCL SM (tuned)','#82B4A3','D',(0,(2,1.5))),
    ('ce','NCCL CE (tuned)','#B899C8','^',(0,(6,2,1,2))),
)

def main():
    with (DATA/'summary.csv').open()as f:fresh=list(csv.DictReader(f))
    historical=ROOT/'benchmarks/data/nccl_tuned_2x4l4_20261116/figure_inputs.csv'
    with historical.open()as f:large=[r for r in csv.DictReader(f)if r['operation']=='allgather']
    configure(black_axes=True)
    fig,axes=collective_panels();exported=[]
    for ax,panel,powers,metric,ylabel,rows,source in (
        (axes[0],'latency',range(10,21),'median_us','Latency (µs)',fresh,DATA/'summary.csv'),
        (axes[1],'throughput',range(20,31),'bus_GBps','Bus throughput (GB/s)',large,historical),
    ):
        sizes=[1<<p for p in powers]
        clean_axis(ax);ax.set_xscale('log',base=2);ax.set_xlim(sizes[0]/1.35,sizes[-1]*1.35)
        ticks=[1<<p for p in ((10,12,14,16,18,20)if panel=='latency'else(20,22,24,26,28,30))]
        ax.set_xticks(ticks,[f'{n>>10}K'if n<1<<20 else f'{n>>20}M'if n<1<<30 else '1G'for n in ticks])
        ax.set_xlabel('Complete tensor bytes per rank');ax.set_ylabel(ylabel);ax.yaxis.set_label_coords(-.16,.5)
        upper=0
        for key,label,color,marker,line in STYLES:
            points={int(r['bytes']):r for r in rows if r['series']==key};assert sorted(points)==sizes
            values=[float(points[n][metric])for n in sizes]
            ax.plot(sizes,values,color=color,marker=marker,linestyle=line,
                    linewidth=3.6 if key=='uccl'else 2.2,markersize=6.5 if key=='uccl'else 5,
                    markeredgecolor='white',markeredgewidth=.7,label=label)
            upper=max(upper,*values)
            for n in sizes:
                exported.append(dict(panel=panel,series=key,bytes=n,value=points[n][metric],
                                     source=str(source.relative_to(ROOT))))
        ax.set_ylim(0,upper*1.13)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.995),ncol=2,
               frameon=False,fontsize=11.5,handlelength=1.6,handletextpad=.5,columnspacing=.8,labelspacing=.4)
    save(fig,'collectives-2x4l4-allgather-nccl-tuned',preview=True)
    with (DATA/'figure_inputs.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=['panel','series','bytes','value','source'],lineterminator='\n');w.writeheader();w.writerows(exported)
    assert len(exported)==88
    (DATA/'historical_throughput.sha256').write_text(hashlib.sha256(historical.read_bytes()).hexdigest()+'  '+str(historical.relative_to(ROOT))+'\n')

if __name__=='__main__':main()
