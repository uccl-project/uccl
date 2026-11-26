#!/usr/bin/env python3
"""Summarize every measured size/layout, preserving all independent jobs."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    shapes = {tuple(x) for x in manifest['shapes']}
    rows = []
    for op, number in [('all_gather', 1), ('reduce_scatter', 2), ('all_reduce', 3)]:
        runs = {backend: [v['rows_us'] for key, v in manifest['results'].items()
                          if re.fullmatch(op + r'_\d+_' + backend, key)]
                for backend in ['native', 'candidate']}
        if not runs['native'] or not runs['candidate']:
            raise ValueError('missing backend jobs for ' + op)
        sizes = set(runs['native'][0])
        if any(set(r) != sizes for group in runs.values() for r in group):
            raise ValueError('inconsistent measured size coverage')
        for size in sorted(sizes, key=int):
            for index, layout in enumerate(['out-of-place', 'in-place']):
                native = [r[size][index] for r in runs['native']]
                candidate = [r[size][index] for r in runs['candidate']]
                n, c = statistics.median(native), statistics.median(candidate)
                rows.append(dict(operation=op, bytes=int(size), layout=layout,
                                 route='uccl_lite' if (number, int(size) // (4 if number == 3 else 32)) in shapes else 'nccl',
                                 native_median_us=n, candidate_median_us=c,
                                 native_min_us=min(native), native_max_us=max(native),
                                 candidate_min_us=min(candidate), candidate_max_us=max(candidate),
                                 candidate_over_native=c / n,
                                 native_jobs=len(native), candidate_jobs=len(candidate)))
    with (args.directory / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failures = [r for r in rows if r['candidate_over_native'] > 1 + manifest['max_regression']]
    print(f'{len(failures)}/{len(rows)} rows exceed the {manifest["max_regression"]:.0%} limit')
    for op in ['all_gather', 'reduce_scatter', 'all_reduce']:
        subset = [r for r in rows if r['operation'] == op]
        worst = max(subset, key=lambda r: r['candidate_over_native'])
        print(op, 'worst:', worst['bytes'], worst['layout'], f'{worst["candidate_over_native"]:.4f}x')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for ax, op in zip(axes, ['all_gather', 'reduce_scatter', 'all_reduce']):
            for layout, marker in [('out-of-place', 'o'), ('in-place', 's')]:
                subset = [r for r in rows if r['operation'] == op and r['layout'] == layout]
                ax.plot([r['bytes'] for r in subset], [r['candidate_over_native'] for r in subset],
                        marker=marker, markersize=3, label=layout)
            ax.axhline(1, color='black', linewidth=0.8)
            ax.axhline(1 + manifest['max_regression'], color='red', linestyle='--', label=f'{manifest["max_regression"]:.0%} limit')
            ax.set_xscale('log', base=2)
            ax.set_xticks([1 << 10, 1 << 15, 1 << 20, 1 << 25, 1 << 30],
                          ['1 KiB', '32 KiB', '1 MiB', '32 MiB', '1 GiB'])
            ax.set_title({'all_gather': 'AllGather', 'reduce_scatter': 'ReduceScatter',
                          'all_reduce': 'AllReduce' if any(op == 3 for op, _ in shapes) else 'AllReduce (native NCCL throughout)'}[op])
            ax.grid(alpha=0.2)
        axes[0].set_ylabel('Adapter / native NCCL latency (lower is better)')
        axes[-1].legend(fontsize=8)
        fig.suptitle('Two four-L4 hosts, FP32; medians of independent maximum-rank batch means')
        fig.text(0.5, 0.02, 'All candidate operations/sizes execute UCCL; native NCCL is a separate baseline.' if 'strict UCCL-only' in manifest['scope'] else 'Frozen profile selects UCCL/NCCL routes.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(args.directory / 'latency_ratios.png', dpi=160)


if __name__ == '__main__':
    main()
