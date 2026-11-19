import csv,json,hashlib,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();src=root/'build/uccl_cooperative_rs8_candidate'
assert json.loads((src/'full_exit.json').read_text())['exit_code']==0
summary=json.loads((src/'full_diagnostic.json').read_text())
dst=root/'benchmarks/data/uccl_reference_gap_20261118/cooperative_rs8_candidate';dst.mkdir(exist_ok=True)
for sub in ['abi','ag_measure','ar_measure','rs_measure']:
 for p in (src/sub).rglob('*'):
  if p.is_file() and p.suffix in ['.json','.log','.csv','.py','.txt']:
   q=dst/sub/p.relative_to(src/sub);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
for p in src.iterdir():
 if p.is_file() and p.suffix in ['.json','.log','.py']:shutil.copyfile(p,dst/p.name)
fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
selected=[]
for ax,op in zip(axes,['ag','ar','rs']):
 rows=list(csv.DictReader((src/f'{op}_measure/comparison.csv').open()))
 for layout,marker in [('out-of-place','o'),('in-place','s')]:
  rr=[r for r in rows if r['layout']==layout]
  ax.plot([int(r['bytes']) for r in rr],[float(r['diagnostic_ratio']) for r in rr],label=layout,marker=marker,markersize=3)
 ax.axhline(1,color='black',linestyle='--',linewidth=1)
 ax.set_xscale('log',base=2);ax.set_xticks([1<<10,1<<20,1<<30],['1 KiB','1 MiB','1 GiB'])
 ax.set_title(f"{op.upper()}: {summary['operations'][op]['slower_rows']}/42 slower rows")
 ax.set_ylim(0,max(1.1,1.05*max(v['worst_ratio'] for v in summary['operations'].values())));ax.grid(alpha=.2)
 for r in rows:
  if int(r['bytes']) in [262144,524288,1048576,2097152,4194304,8388608,1073741824]:selected.append(dict(operation=op,**r))
axes[0].set_ylabel('Candidate / fresh UCCL reference latency');axes[0].legend(fontsize=8)
fig.suptitle('Fixed-library diagnostic; harness conventions differ, no admission claim',fontsize=11)
fig.text(.5,.015,'2 hosts × 4 L4 | FP32 | median of 3 sweeps | 20 warmups / 30 iterations | no best-per-size selection',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.06,1,.95));fig.savefig(dst/'diagnostic_ratios.png',dpi=180);plt.close(fig)
with (dst/'selected_rows.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
lines=['# Fixed cooperative AG + 8-MiB streamed RS candidate','', '**The zero-regression goal remains unmet.** This is one experimental binary;', 'all three complete curves were measured with that same SHA-256:', '', '`'+summary['library_sha256']+'`', '', 'Two four-L4 hosts (10.10.55.2 and 10.10.55.1), FP32 SUM AR/RS and bitwise AG.', 'Three unchanged nccl-tests sweeps per operation, all 21 powers of two from', '1 KiB to 1 GiB, out-of-place and in-place, 20 warmups and 30 timed iterations.', 'Timing includes host enqueue through completion and reports the maximum rank.', 'Each row uses its three-run median. No curve selects the best variant per size.', '', 'The comparison uses the freshly reproduced original README UCCL implementations.', 'Their process layout, per-call checking, pointer reuse and timing conventions', 'differ from nccl-tests, so these ratios are diagnostic, not matched admission.', 'The original baseline and zero-slowdown criterion have not been relaxed.', '', '| Operation | Slower diagnostic rows / 42 | Worst ratio |', '| --- | ---: | ---: |']
for op,v in summary['operations'].items():lines.append(f"| {op.upper()} | {v['slower_rows']} | {v['worst_ratio']:.4f} |")
lines += ['', '![Diagnostic ratios](diagnostic_ratios.png)', '', 'The fixed compile-time policy uses cooperative host-packet AG at 256 KiB–4 MiB,', 'with a single source packet per call and explicit grid residency/fence/ACK', 'rules. Larger AG retains the three-lane CE path. Hybrid AR uses GPU RS and', 'that host gather. RS uses the existing streamed path at 8 MiB and CE graphs', 'from 16 MiB, retaining the original 8-MiB graph-piece target. Other sizes keep', 'their existing UCCL algorithms. This build is not enabled by default.', '', 'The extended ABI passes 126 calls/rank over two communicator lifetimes,', 'including alignment, mixed per-rank layout, streams and allocation recycling.', 'The cooperative component separately passes 54 calls/rank with packet tails;', 'see the preceding one-packet record. Every timed job passes nccl-tests numerical', 'checks, per-rank/per-size strict route checks and native-library-absent checks.', 'The library has SONAME libnccl.so.2 and no native NCCL dependency or fallback.', '', 'Build command, source hashes, ABI evidence, raw logs, all 126 comparison rows', 'and this report generator are retained here. Exact source snapshots and binaries', 'remain under build/uccl_cooperative_rs8_candidate/. Negative experiments are', 'retained in the neighboring host_control_experiments record.']
(dst/'README.md').write_text('\n'.join(lines)+'\n')
print(summary['operations'])
print(dst)
