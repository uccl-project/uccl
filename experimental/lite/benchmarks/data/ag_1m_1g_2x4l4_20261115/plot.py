"""Plot five fixed configurations, with three-round ranges and no envelope."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / 'paired/summary.csv').open()))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
styles = [('serial', 'SM direct'), ('ring', 'Ring'), ('ce4096', 'CE, 4 MiB pieces'),
          ('nccl_default', 'NCCL default'), ('nccl_simple', 'NCCL Ring+Simple')]
for mode, label in styles:
    selected = sorted([r for r in rows if r['mode'] == mode], key=lambda r: int(r['bytes']))
    x = [int(r['bytes']) / 2**20 for r in selected]
    y = [float(r['median_us']) / 1000 for r in selected]
    error = [[(float(r['median_us'])-float(r['min_us'])) / 1000 for r in selected],
             [(float(r['max_us'])-float(r['median_us'])) / 1000 for r in selected]]
    axes[0].errorbar(x, y, yerr=error, marker='o', markersize=3, capsize=2, label=label)
    # Bus bandwidth convention for eight-rank AllGather: (N / t) * 7/8.
    bw = [int(r['bytes']) / (float(r['median_us']) * 1e-6) / 1e9 * 7/8 for r in selected]
    axes[1].plot(x, bw, marker='o', markersize=3, label=label)
for ax in axes:
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 1024], labels=['1', '4', '16', '64', '256', '1024'])
    ax.set_xlabel('Complete output per GPU (MiB)')
    ax.grid(True, alpha=.25)
axes[0].set_yscale('log')
axes[0].set_ylabel('Host enqueue-to-completion latency (ms)')
axes[0].legend(fontsize=8)
axes[1].set_ylabel('Bus bandwidth (GB/s), N/time × 7/8')
fig.suptitle('2 × 4 NVIDIA L4 · FP32 AllGather · 20 warmups + 20 calls × 3 rounds')
fig.tight_layout()
fig.savefig(root / 'results.png', dpi=180)
