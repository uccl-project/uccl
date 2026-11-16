"""Compare fresh NCCL transport runs with historical PCIeCCL fine measurements."""
import csv
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
OUT = R / 'forced-p2p'

def read(p):
    return json.loads(p.read_text())

def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

roots = {'nccl_default_fresh': OUT / 'default', 'nccl_p2p': OUT / 'p2p',
         'nccl_default_prior': R / 'results/final-validation/nccl',
         'pcieccl_original_prior': R / 'results/final-original/pcieccl',
         'pcieccl_c03_prior': R / 'results/final-validation/pcieccl'}
for mode in ('p2p', 'default'):
    assert read(OUT / (mode+'-before.json')) == read(OUT / (mode+'-after.json'))
    t = read(OUT / mode / 'transport.json')
    if mode == 'p2p':
        assert t['p2p_logging_ranks'] == list(range(8)) and not t['other_rows']
    assert not (OUT / mode / 'external-gpu-work.json').exists()
rows = []
for bs in (1, 2, 4, 8, 16, 32, 64):
    row = {'concurrency': bs}
    for key, folder in roots.items():
        p = folder / 'cases' / f'isl8192-osl1024-bs{bs}.json'
        d = read(p); m = d['metrics']
        assert d['status'] == 'complete' and m['request_errors'] == 0
        assert d['case'] == {'name':f'isl8192-osl1024-bs{bs}', 'isl':8192, 'osl':1024, 'bs':bs, 'warmup_seconds':60.0, 'measurement_seconds':180.0}
        assert m['requests_completed'] > 0
        assert math.isclose(m['output_token_throughput'], m['measurement_output_tokens']/180, rel_tol=1e-12)
        row[key+'_tok_s'] = m['output_token_throughput']
        row[key+'_tpot_ms'] = m['tpot_ms']['p50']
        row[key+'_ttft_ms'] = m['ttft_ms']['p50']
    for key, a, b in [('p2p_vs_default','nccl_p2p','nccl_default_fresh'), ('default_fresh_vs_prior','nccl_default_fresh','nccl_default_prior'), ('original_vs_p2p','pcieccl_original_prior','nccl_p2p'), ('c03_vs_p2p','pcieccl_c03_prior','nccl_p2p')]:
        row[key] = row[a+'_tok_s']/row[b+'_tok_s']
    rows.append(row)
ratios = {k:gm([r[k] for r in rows]) for k in ('p2p_vs_default','default_fresh_vs_prior','original_vs_p2p','c03_vs_p2p')}
data = {'rows':rows, 'geomeans':ratios, 'quality':'Not reevaluated in this transport follow-up; previous PCIeCCL quality acceptance remains unconfirmed.', 'limitations':['One fresh matrix per NCCL transport, fixed order P2P then default; no confidence interval.', 'PCIeCCL curves are previous measurements; fresh default control measures temporal drift.', 'Forcing NCCL transport affects all NCCL communication in the serving process, not only AllReduce.', 'Same upstream baseline: vLLM custom and symmetric-memory AllReduce disabled; NCCL 2.28.9.'], 'transport':{k:read(OUT/k/'transport.json') for k in ('p2p','default')}}
(OUT/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
with (OUT/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
fig, axes = plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
xs=[r['concurrency'] for r in rows]
styles=[('nccl_default_fresh','NCCL default (fresh)','#555555','-'), ('nccl_p2p','NCCL forced P2P (fresh)','#8a4ebf','-'), ('pcieccl_original_prior','Original PCIeCCL (prior)','#1877b8','--'), ('pcieccl_c03_prior','PCIeCCL + EAT (prior)','#d45b22','--')]
for key,label,color,style in styles:
    for ax,metric in zip(axes[:2],('tok_s','tpot_ms')):
        ax.plot(xs,[r[key+'_'+metric] for r in rows],marker='o',color=color,linestyle=style,label=label)
for key,label,color in [('p2p_vs_default','NCCL P2P / default','#8a4ebf'), ('original_vs_p2p','Original PCIeCCL / NCCL P2P','#1877b8'),('c03_vs_p2p','PCIeCCL + EAT / NCCL P2P','#d45b22')]:
    axes[2].plot(xs,[r[key] for r in rows],'o-',color=color,label=label)
axes[2].axhline(1,color='#777777',linestyle=':')
for ax in axes:
    ax.set_xscale('log',base=2); ax.set_xticks(xs,[str(x) for x in xs]); ax.set_xlabel('Concurrent requests'); ax.grid(alpha=.2)
axes[0].set_ylabel('Output tokens/s ↑'); axes[1].set_ylabel('Median TPOT (ms) ↓'); axes[2].set_ylabel('Throughput ratio')
axes[0].legend(fontsize=8); axes[2].legend(fontsize=8)
fig.suptitle('MiniMax-M2.7 FP8 · 8× RTX PRO 6000 · TP8+EP · 8192 input / 1024 output tokens\n60 s warmup + 180 s measurement per case · Prior PCIeCCL quality remains unconfirmed',fontsize=11)
fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.svg')
lines=['Fresh NCCL transport E2E comparison on wxz-pro6000 (8 GPUs)', '', 'Concurrency  NCCL default  NCCL P2P  Original PCIeCCL*  EAT PCIeCCL*  P2P/default']
for r in rows:
    lines.append(f"{r['concurrency']:11d}  {r['nccl_default_fresh_tok_s']:12.2f}  {r['nccl_p2p_tok_s']:8.2f}  {r['pcieccl_original_prior_tok_s']:17.2f}  {r['pcieccl_c03_prior_tok_s']:12.2f}  {r['p2p_vs_default']:11.4f}")
lines += ['', '* PCIeCCL columns are previous measurements, in output tokens/s.', '', 'Geometric means: '+json.dumps(ratios), '', data['quality'], *data['limitations']]
(OUT/'report.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(ratios,indent=2))
