#!/usr/bin/env python3
"""Extract last captured collective after three profiling warmups."""
import csv
import json
import pathlib
import sqlite3
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path(sys.argv[1])
variants=['uccl_fp32_reference','sp_direct_8','sp_fixed4_barrier_8']
fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained')
summary=[]
max_end=0
for ax,variant in zip(axes,variants):
    c=sqlite3.connect('build/ce_1k_bubble_'+variant+'.sqlite')
    name='pushSerialGather' if variant=='uccl_fp32_reference' else 'downloadPacket'
    kernels=c.execute('SELECT k.start,k.end,k.deviceId,k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE ? ORDER BY k.start',('%'+name+'%',)).fetchall()[-4:]
    assert len(kernels)==4 and len({k[2] for k in kernels})==4
    copies=[]
    if variant!='uccl_fp32_reference':
        copies=c.execute('SELECT start,end,deviceId,correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE bytes=256 ORDER BY start').fetchall()[-4:]
        assert len(copies)==4 and len({k[2] for k in copies})==4
    ops=[('D2H',x) for x in copies]+[('SM',x) for x in kernels]
    api={}
    for kind,x in ops:
        api[x[3]]=c.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=?',(x[3],)).fetchone()
        assert api[x[3]] is not None
    origin=min(a[0] for a in api.values())
    ready=max(x[1] for x in copies) if copies else None
    report={'variant':variant,'trace_only_not_normal_latency':True,'devices':[]}
    for d in range(4):
        k=next(x for x in kernels if x[2]==d)
        row={'gpu':d,'kernel_start_us':(k[0]-origin)/1000,'kernel_duration_us':(k[1]-k[0])/1000,
             'kernel_api_start_us':(api[k[3]][0]-origin)/1000}
        if copies:
            u=next(x for x in copies if x[2]==d)
            row.update(upload_start_us=(u[0]-origin)/1000,upload_duration_us=(u[1]-u[0])/1000,
                       wait_for_other_uploads_us=(ready-u[1])/1000,
                       after_all_uploads_to_kernel_us=(k[0]-ready)/1000,
                       local_upload_end_to_kernel_us=(k[0]-u[1])/1000)
        report['devices'].append(row)
    report['kernel_start_spread_us']=(max(k[0] for k in kernels)-min(k[0] for k in kernels))/1000
    if copies:
        report['upload_start_spread_us']=(max(k[0] for k in copies)-min(k[0] for k in copies))/1000
        report['all_uploads_done_to_first_kernel_us']=(min(k[0] for k in kernels)-ready)/1000
    summary.append(report)
    max_end=max(max_end,max((max(x[1],api[x[3]][1])-origin)/1000 for _,x in ops))
    if copies:
        for d in range(4):
            u=next(x for x in copies if x[2]==d)
            k=next(x for x in kernels if x[2]==d)
            ax.broken_barh([((u[1]-origin)/1000,(k[0]-u[1])/1000)],(d-.28,.56),facecolors='#f7dddd')
    for kind,x in ops:
        start,end,device,corr=x
        color='#4088bb' if kind=='D2H' else '#df873b'
        ax.broken_barh([((start-origin)/1000,(end-start)/1000)],(device-.28,.56),facecolors=color)
        # Gray outline: API submission interval, shown below each GPU lane.
        a,b=api[corr]
        ax.broken_barh([((a-origin)/1000,(b-a)/1000)],(device+.3,.1),facecolors='#888888')
    if ready:
        ax.axvline((ready-origin)/1000,color='#444444',linestyle='--',linewidth=1,label='All uploads finished')
    ax.set_yticks(range(4),['GPU '+str(d) for d in range(4)]);ax.invert_yaxis()
    ax.set_title(variant);ax.grid(axis='x',alpha=.2);ax.set_xlabel('µs from first transfer/kernel API submission (profiled)')
for ax in axes: ax.set_xlim(0,max_end*1.04)
fig.suptitle('1 KiB: D2H (blue), SM (orange), CPU submission intervals (gray)\nProfiled timelines show gaps; absolute timings are perturbed by tracing',fontsize=12)
fig.savefig(root/'timeline.png',dpi=180);fig.savefig(root/'timeline.svg')
(root/'timeline.json').write_text(json.dumps(summary,indent=2)+'\n')
for r in summary:
 print(r['variant'],{k:v for k,v in r.items() if k not in ['devices','variant','trace_only_not_normal_latency']})
 print(r['devices'])
