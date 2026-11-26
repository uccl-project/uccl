#!/usr/bin/env python3
"""Fixed chunk-size curves and whole-shard P2P controls, no winner envelope."""
import csv
import pathlib
import statistics
import sys
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
root=pathlib.Path(sys.argv[1])
groups=defaultdict(list)
sizes=[1<<p for p in range(23,31)]
chunks=[1,2,4,8,16]
for sweep in ['forward','reverse']:
    for size in sizes:
        rounds=defaultdict(list)
        with (root/sweep/f'{size}.csv').open() as f:
            for r in csv.DictReader(f):
                rounds[(r['variant'],r['round'])].append((float(r['host_us']),float(r['enqueue_us'])))
        assert len(rounds)==24,(sweep,size,len(rounds))
        for (name,_),samples in rounds.items():
            assert len(samples)==10
            groups[size,name].append(tuple(statistics.mean(v[i] for v in samples) for i in [0,1]))
data={k:statistics.median(v[0] for v in values) for k,values in groups.items()}
with (root/'summary.csv').open('w') as f:
    w=csv.writer(f)
    w.writerow(['output_bytes','variant','effective_chunk_bytes','median_us','min_round_mean_us','max_round_mean_us','enqueue_us','reference_speedup','timed_calls'])
    for (size,name),means in sorted(groups.items()):
        assert len(means)==6
        requested=int(name.split('_')[-1][:-1])*(1<<20) if name.startswith('hybrid_grouped_') else None
        effective=min(requested,size//4) if requested else size//4 if name.startswith('p2p_') else ''
        w.writerow([size,name,effective,data[size,name],min(v[0] for v in means),max(v[0] for v in means),
                    statistics.median(v[1] for v in means),data[size,'uccl_fp32_reference']/data[size,name],60])
def label(s):return '1 GiB' if s==1<<30 else f'{s>>20} MiB'
with (root/'table.md').open('w') as f:
    f.write('| Output/GPU | UCCL ref ms | '+ ' | '.join(f'Hybrid {c} MiB ms' for c in chunks)+' | Batch P2P ms | Batch+memop ms |\n')
    f.write('|---:|'+ '---:|'*8+'\n')
    for size in sizes:
        names=['uccl_fp32_reference']+[f'hybrid_grouped_{c}m' for c in chunks]+['p2p_xor_batch_ce_fixed4','p2p_memop_onewait_fixed4']
        f.write('| '+label(size)+' | '+' | '.join(f'{data[size,n]/1000:.3f}' for n in names)+' |\n')
fig,axes=plt.subplots(1,2,figsize=(14,5.3),layout='constrained')
curves=[('uccl_fp32_reference','UCCL FP32 reference','black','-')]
colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
curves += [(f'hybrid_grouped_{c}m',f'Hybrid / {c} MiB',color,'-') for c,color in zip(chunks,colors)]
curves += [('p2p_xor_batch_ce_fixed4','Whole-shard batch P2P','#777777','--'),
           ('p2p_memop_onewait_fixed4','Whole-shard batch + memop','#8c564b',':')]
for name,title,color,style in curves:
    axes[0].plot(sizes,[data[s,name]/1000 for s in sizes],marker='o',markersize=3,color=color,linestyle=style,label=title)
axes[0].set_xscale('log',base=2);axes[0].set_yscale('log')
axes[0].set_xticks(sizes,[label(s) for s in sizes],rotation=30,ha='right')
axes[0].set_xlabel('Complete output/GPU');axes[0].set_ylabel('Host enqueue-to-completion (ms)')
axes[0].grid(alpha=.2);axes[0].legend(fontsize=8,ncol=2)
ratios=[[data[s,'uccl_fp32_reference']/data[s,f'hybrid_grouped_{c}m'] for c in chunks] for s in sizes]
lo=min(.99,min(min(r) for r in ratios));hi=max(1.01,max(max(r) for r in ratios))
im=axes[1].imshow(ratios,aspect='auto',cmap='RdYlGn',norm=TwoSlopeNorm(vmin=lo,vcenter=1,vmax=hi))
axes[1].set_xticks(range(len(chunks)),[f'{c} MiB' for c in chunks]);axes[1].set_yticks(range(len(sizes)),[label(s) for s in sizes])
axes[1].set_xlabel('Configured chunk per source (* clipped to source size)')
axes[1].set_title('Reference / fixed hybrid latency: >1 is faster')
for i,s in enumerate(sizes):
    for j,c in enumerate(chunks):
        star='*' if c*(1<<20)>s//4 else ''
        axes[1].text(j,i,f'{ratios[i][j]:.3f}×{star}',ha='center',va='center',fontsize=8)
fig.colorbar(im,ax=axes[1],shrink=.8)
fig.suptitle('4× L4 • FP32 • fresh forward/reverse sweeps • separate fixed chunk configurations')
fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.svg')
print((root/'table.md').read_text())
