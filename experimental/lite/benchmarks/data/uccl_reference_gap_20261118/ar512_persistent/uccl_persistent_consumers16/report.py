import csv,json,hashlib,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();base=root/'build/uccl_persistent_consumers16';dst=root/'benchmarks/data/uccl_reference_gap_20261118/ar512_persistent/uccl_persistent_consumers16'
summary=json.loads((base/'full_diagnostic.json').read_text());lib=base/'nccl_lite/libuccl_lite_nccl_strict.so';sha=hashlib.sha256(lib.read_bytes()).hexdigest()
summary['library_sha256']=sha;summary['scope']='fixed library, three sweeps per operation; cross-harness UCCL diagnostic, no matched admission';(base/'full_diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n');shutil.copyfile(base/'full_diagnostic.json',dst/'full_diagnostic.json')
fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
for ax,op in zip(axes,['ag','ar','rs']):
 rows=list(csv.DictReader((base/f'{op}_measure/comparison.csv').open()))
 for layout,marker in [('out-of-place','o'),('in-place','s')]:
  rr=[r for r in rows if r['layout']==layout];ax.plot([int(r['bytes']) for r in rr],[float(r['diagnostic_ratio']) for r in rr],label=layout,marker=marker,markersize=3)
 ax.axhline(1,color='black',linestyle='--',linewidth=1);ax.set_xscale('log',base=2);ax.set_xticks([1<<10,1<<20,1<<30],['1 KiB','1 MiB','1 GiB']);ax.set_ylim(0,1.25);ax.grid(alpha=.2);ax.set_title(f"{op.upper()}: {summary['operations'][op]['slower_rows']}/42 slower")
axes[0].set_ylabel('Candidate / UCCL reference latency');axes[0].legend(fontsize=8)
fig.suptitle('Fixed persistent candidate: diagnostic comparison, harness conventions differ',fontsize=11)
fig.text(.5,.015,'2 × 4 L4 | FP32 | median of 3 sweeps | 20 warmups / 30 iterations | no best-per-size selection',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.06,1,.95));fig.savefig(dst/'diagnostic_ratios.png',dpi=180);plt.close(fig)
(dst/'README.md').write_text(f'''# Fixed persistent candidate: 16 consumers

**Zero-regression remains unmet: 22/126 diagnostic rows are slower.**
The same binary was used for all three operations:
`{sha}`.

Two four-L4 hosts, FP32 SUM AR/RS and bitwise AG, all 21 powers of two from
1 KiB to 1 GiB, out-of-place and in-place. Three unchanged nccl-tests sweeps
per operation, 20 warmups and 30 timed iterations, maximum-rank host enqueue
through completion batch means, with medians across the three sweeps. This is
one fixed compiled policy; no curve selects the best variant per size.

The original UCCL reference has different process/layout, pointer-reuse and
per-call timing conventions. These comparisons are diagnostic, not matched
admission. The baseline and zero-slowdown criterion have not been relaxed.

| Operation | Slower rows / 42 | 1-GiB candidate out/in | Fresh UCCL |
| --- | ---: | --- | --- |
| AG | 5 | 38.149 / 38.235 ms | 41.922 ms |
| AR | 6 | 92.180 / 92.247 ms | 96.445 ms |
| RS | 11 | 54.162 / 53.874 ms | 53.693 ms |

![Diagnostic ratios](diagnostic_ratios.png)

The persistent grid has one control block, four upload blocks and 16 consumer
blocks. The 64 vector tiles of a 256-KiB source packet divide evenly among the
consumers. The cooperative launch checks whole-grid residency. Producer system
fences, per-slot completion counts, system-release upload/reader flags, receive
credits and outgoing retirement remain enforced. The CPU worker performs only
host atomics and MSCCL++ RDMA. Caller stream completion includes all GPU actors.

The fixed policy retains cooperative one-packet AG through 4 MiB and uses this
persistent path above it. Public 8-MiB RS is streamed; hybrid AR retains graph
RS for its private shard producer. The direct-input AR experiment applies only
at 512 KiB, retaining its original single-bank reuse/drain protocol.

The component passes 54 calls/rank over two lifetimes. The expanded mixed ABI
passes 144 calls/rank, including the 512-KiB boundary, offsets, per-rank layout
variations, caller streams and recycled allocations. All nine nccl-tests jobs
pass numerical, exact per-size/per-rank route, native-library-absent and zero
fallback checks. No sanitizer or universal NCCL compatibility claim is made.

Remaining diagnostic failures: AG at 8 and 16 MiB, plus a small 32-MiB
out-of-place excess; AR at 256 KiB, 4 MiB and 8 MiB; eleven RS rows. Even tiny
positive ratios remain failures here. Same-condition paired admission and a
complete zero-regression gate are still required. This is not a default release.
''')
shutil.copyfile(base/'report.py',dst/'report.py');print(sha)
