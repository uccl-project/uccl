#!/usr/bin/env python3
"""Render paired end-to-end bars from all cases in the checked summary."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--summary',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--preview-dir',type=Path,default=Path('build/l4_e2e/figures'))
a=p.parse_args()
s=json.loads(a.summary.read_text())
a.out.mkdir(parents=True,exist_ok=True)
a.preview_dir.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
for kind in ('serving','training'):
    rows=[r for r in s['cases'] if r['kind']==kind]
    fig,ax=plt.subplots(figsize=(6.3,3.8))
    x=np.arange(len(rows));w=.33
    ax.bar(x-w/2,[r['nccl_ms'] for r in rows],w,label='NCCL (default)',color='#8c98a7',yerr=[abs(r['phase_medians_ms'][3]-r['phase_medians_ms'][0])/2 for r in rows],capsize=3)
    bars=ax.bar(x+w/2,[r['peer_ms'] for r in rows],w,label='UCCL Lite' + (' + NCCL DP sync' if s['nodes']==2 and kind=='training' else ''),color='#168c87',yerr=[abs(r['phase_medians_ms'][2]-r['phase_medians_ms'][1])/2 for r in rows],capsize=3)
    for bar,r in zip(bars,rows):
        change=r['latency_reduction_pct']
        label=f"{abs(change):.1f}% {'lower' if change >= 0 else 'slower'}\n{r['throughput_ratio']:.2f}×"
        ax.annotate(label,(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
    ax.set_xticks(x,[r['case'].replace(':',' × ') for r in rows])
    ax.set_xlabel('Global batch × input × output tokens' if kind=='serving' else 'Global batch × sequence length')
    ax.set_ylabel('Whole HTTP batch (ms)' if kind=='serving' else 'Full SGD step (ms)')
    ax.set_title(f"{s.get('model_label','GPT-2-large')} · {'FP32 params / BF16 compute' if s.get('parameter_dtype')=='float32' else 'BF16'}\n{s['nodes']} × 4 L4 · DP{s['nodes']} × shard4 · {kind.title()}",fontweight='bold',pad=30)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,fontsize=9)
    ax.set_ylim(0,max(r['nccl_ms'] for r in rows)*1.3)
    ax.spines[['top','right']].set_visible(False)
    ax.yaxis.grid(True,alpha=.18);ax.set_axisbelow(True)
    fig.text(.5,.015,f"Explicit parameter-sharded adapter · 2 jobs/backend · {s.get('warmups',5)} warmups + {s.get('repeats_per_job',20)} repeats/job\nMedian of job medians; whiskers span job medians; initialization excluded",ha='center',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.08,1,1))
    fig.savefig(a.out/f'{kind}-e2e-{s["nodes"]}x4l4.svg')
    fig.savefig(a.preview_dir/f'{kind}-e2e-{s["nodes"]}x4l4.png',dpi=140)
    plt.close(fig)
