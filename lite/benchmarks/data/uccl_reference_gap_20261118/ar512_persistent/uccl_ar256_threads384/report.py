import csv,json,hashlib,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();base=root/'build/uccl_ar256_threads384';dst=root/'benchmarks/data/uccl_reference_gap_20261118/ar512_persistent/uccl_ar256_threads384';dst.mkdir(exist_ok=True)
assert json.loads((base/'full_exit.json').read_text())['exit_code']==0
s=json.loads((base/'full_diagnostic.json').read_text());rows={op:list(csv.DictReader((base/f'{op}_measure/comparison.csv').open())) for op in ['ag','ar','rs']}
failed=sum(v['slower_rows'] for v in s['operations'].values())
fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
for ax,op in zip(axes,rows):
 for layout,marker in [('out-of-place','o'),('in-place','s')]:
  rr=[r for r in rows[op] if r['layout']==layout];ax.plot([int(r['bytes']) for r in rr],[float(r['diagnostic_ratio']) for r in rr],label=layout,marker=marker,markersize=3)
 ax.axhline(1,color='black',linestyle='--',linewidth=1);ax.set_xscale('log',base=2);ax.set_xticks([1<<10,1<<20,1<<30],['1 KiB','1 MiB','1 GiB']);ax.set_ylim(0,max(1.1,1.05*max(v['worst_ratio'] for v in s['operations'].values())));ax.grid(alpha=.2);ax.set_title(f"{op.upper()}: {s['operations'][op]['slower_rows']}/42 slower")
axes[0].set_ylabel('Candidate / UCCL reference latency');axes[0].legend(fontsize=8);fig.suptitle('Fixed fused candidate: diagnostic only, harness conventions differ',fontsize=11)
fig.text(.5,.015,'2 × 4 L4 | FP32 | median of 3 sweeps | 20 warmups / 30 iterations | no best-per-size selection',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.06,1,.95));fig.savefig(dst/'diagnostic_ratios.png',dpi=180);plt.close(fig)
lines=['# Combined candidate with 384-thread AR boundary','',f'**Zero-regression remains unmet: {failed}/126 diagnostic rows are slower.**',f"All operations use one binary: `{s['library_sha256']}`.",'','Two four-L4 hosts, FP32 SUM AR/RS and bitwise AG, 21 powers of two from','1 KiB to 1 GiB, both layouts, three unchanged nccl-tests sweeps per operation,','20 warmups and 30 timed iterations. Timing includes host enqueue/completion','and uses the maximum rank. Each row is the three-run median. This is one fixed','policy, not an envelope of per-size minima. Original UCCL harness process,','pointer-reuse and timing conventions differ, so this is not matched admission.','','| Operation | Slower rows / 42 | 1-GiB candidate out/in (ms) | UCCL (ms) |','| --- | ---: | --- | --- |']
for op in rows:
 rr=[r for r in rows[op] if int(r['bytes'])==1<<30]
 lines.append(f"| {op.upper()} | {s['operations'][op]['slower_rows']} | {float(rr[0]['median_us'])/1000:.3f} / {float(rr[1]['median_us'])/1000:.3f} | {float(rr[0]['reference_us'])/1000:.3f} |")
lines += ['','![Diagnostic ratios](diagnostic_ratios.png)','','The fixed policy uses one-packet cooperative AG through 8 MiB, the persistent','four-uploader/16-consumer/one-publisher path above it, alternating RS banks','through 32 MiB, and hybrid AR from 1 MiB. The 512-KiB direct-input AR path is','retained. Exactly 256-KiB AR uses 384 threads. RS waits for the remote sum before its final own-half addition, then','writes the result once. Arithmetic order remains (own + local partial) + remote.','All forwarding has finished before that wait, and only the opposite half is','sent over IB. In-place input is read before overwrite; future pieces are disjoint.','','The mixed ABI passes 144 calls/rank over','two lifetimes. All nine timing jobs pass numerical, exact per-size/per-rank','routing, native-library-absent and zero-fallback checks. No default release','has been promoted. No positive diagnostic ratio is rounded into a pass.','','Build commands, raw logs, complete curves and this generator are retained.','Negative experiments are preserved in the surrounding records.']
(dst/'README.md').write_text('\n'.join(lines)+'\n')
shutil.copyfile(base/'report.py',dst/'report.py');print(s['operations'])
