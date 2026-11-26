"""Plot fixed-configuration latency; error bars are the three-round range."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / 'paired/summary.csv').open()))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
styles = [('serial', 'Existing SM direct'), ('ring', 'Existing ring'),
          ('ce4096', 'CE, 4 MiB pieces'), ('ce1024', 'CE, 1 MiB pieces'),
          ('nccl_simple', 'NCCL Ring+Simple')]
for ax, low, high, title in [(axes[0], 1024, 2**21, 'Small output sizes'),
                             (axes[1], 2**20, 2**26, 'Intermediate and larger output sizes')]:
    selected_styles = styles if ax is axes[1] else [
        ('serial', 'Existing SM direct'), ('sm4096', 'New SM, batched notifications'),
        ('ce4096', 'CE, 4 MiB pieces'), ('nccl_default', 'NCCL default'),
        ('nccl_simple', 'NCCL Ring+Simple')]
    for mode, label in selected_styles:
        selected = sorted([r for r in rows if r['mode'] == mode and low <= int(r['bytes']) <= high], key=lambda r: int(r['bytes']))
        x = [int(r['bytes']) / 2**20 for r in selected]
        y = [float(r['median_us']) for r in selected]
        error = [[float(r['median_us'])-float(r['min_us']) for r in selected],
                 [float(r['max_us'])-float(r['median_us']) for r in selected]]
        ax.errorbar(x, y, yerr=error, marker='o', markersize=3, capsize=2, label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Complete output per GPU (MiB)')
    ax.set_ylabel('Host enqueue-to-completion latency (µs)')
    ax.set_title(title)
    ax.grid(True, alpha=.25)
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
fig.suptitle('2 × 4 NVIDIA L4 · FP32 AllGather · 20 warmups + 20 calls × 3 rounds')
fig.tight_layout()
fig.savefig(root / 'latency.png', dpi=180)
