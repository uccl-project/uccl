"""Add independently tuned NCCL curves while preserving UCCL/prior-NCCL inputs."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from figure_style import clean_axis, collective_panels, configure, save

DATA = Path(__file__).resolve().parent
SOURCES = {
    'allreduce': ('ar_small_fix_2x4l4_20261115/summary.csv', 'input_bytes_per_gpu',
                  'hybrid', 'nccl'),
    'allgather': ('ag_host_router_2x4l4_20261115/validated/summary.csv', 'bytes',
                  'shared_ready', 'nccl_shared_ready'),
    'reducescatter': ('rs_direct_medium_2x4l4_20261115/summary.csv', 'input_bytes_per_gpu',
                      'candidate', 'nccl'),
}
STYLES = (
    ('uccl', 'UCCL Lite', '#0068B5', 'o', '-'),
    ('prior', 'NCCL (default)', '#D8A17A', 's', (0, (5, 2.5))),
    ('sm', 'NCCL SM (tuned)', '#82B4A3', 'D', (0, (2, 1.5))),
    ('ce', 'NCCL CE (tuned)', '#B899C8', '^', (0, (6, 2, 1, 2))),
)


def main():
    with (DATA / 'summary.csv').open() as source:
        tuned = list(csv.DictReader(source))
    environment = json.loads((DATA / 'environment.json').read_text())
    configure(black_axes=True)
    exported = []
    for op, (filename, size_field, uccl_mode, prior_mode) in SOURCES.items():
        path = DATA.parent / filename
        relative = str(path.relative_to(ROOT))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == environment['sha256'][relative]
        curves = {}
        with path.open(newline='') as source:
            original = list(csv.DictReader(source))
        for key, mode in [('uccl', uccl_mode), ('prior', prior_mode)]:
            curves[key] = {int(row[size_field]): row for row in original if row['mode'] == mode}
        for family in ('sm', 'ce'):
            curves[family] = {int(row['bytes']): row for row in tuned
                              if row['operation'] == op and row['family'] == family}
        all_sizes = [1 << p for p in range(20 if op == 'allgather' else 10, 31)]
        for key, points in curves.items():
            assert sorted(points) == all_sizes, (op, key)
            for size, row in points.items():
                low, us, high = (float(row[f]) for f in ('min_us', 'median_us', 'max_us'))
                assert all(math.isfinite(v) and v > 0 for v in (low, us, high))
                assert low <= us <= high
                factor = 1.75 if op == 'allreduce' else 0.875
                bus = factor * size / us / 1000
                assert abs(bus - float(row['bus_GBps'])) < 0.001, (op, key, size)
                exported.append((op, key, size, row['median_us'], row['min_us'],
                                 row['max_us'], row['bus_GBps'], relative if key in ('uccl', 'prior')
                                 else str((DATA / 'summary.csv').relative_to(ROOT))))
        fig, axes = collective_panels()
        for ax, powers, metric, ylabel in (
            (axes[0], range(20, 24) if op == 'allgather' else range(10, 21),
             'median_us', 'Latency (µs)'),
            (axes[1], range(20, 31), 'bus_GBps', 'Bus throughput (GB/s)'),
        ):
            sizes = [1 << power for power in powers]
            ticks = sizes if op == 'allgather' and metric == 'median_us' else [
                1 << p for p in ((10, 12, 14, 16, 18, 20) if metric == 'median_us'
                                 else (20, 22, 24, 26, 28, 30))]
            clean_axis(ax)
            ax.set_xscale('log', base=2)
            ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
            ax.set_xticks(ticks, [f'{n >> 10}K' if n < 1 << 20 else
                                 f'{n >> 20}M' if n < 1 << 30 else '1G' for n in ticks])
            ax.set_xlabel('Complete tensor bytes per rank')
            ax.set_ylabel(ylabel)
            ax.yaxis.set_label_coords(-0.16, 0.5)
            upper = 0
            for key, label, color, marker, linestyle in STYLES:
                if key not in curves:
                    continue
                points = curves[key]
                values = [float(points[n][metric]) for n in sizes]
                style = dict(color=color, marker=marker, linestyle=linestyle,
                             linewidth=3.6 if key == 'uccl' else 2.2,
                             markersize=6.5 if key == 'uccl' else 5,
                             markeredgecolor='white', markeredgewidth=0.7, label=label)
                if key == 'uccl' and metric == 'median_us' and op != 'allgather':
                    low = [float(points[n]['min_us']) for n in sizes]
                    high = [float(points[n]['max_us']) for n in sizes]
                    error = [[v - bound for v, bound in zip(values, low)],
                             [bound - v for v, bound in zip(values, high)]]
                    ax.errorbar(sizes, values, yerr=error, capsize=2, **style)
                    upper = max(upper, *high)
                else:
                    ax.plot(sizes, values, **style)
                    upper = max(upper, *values)
            ax.set_ylim(0, upper * 1.13)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.995),
                   ncol=2, frameon=False, fontsize=11.5, handlelength=1.6,
                   handletextpad=0.5, columnspacing=0.8, labelspacing=0.4)
        name = f'collectives-2x4l4-{op}-nccl-tuned'
        save(fig, name, preview=True)
        shutil.copyfile(ROOT / 'docs/assets' / f'{name}.svg', DATA / f'{op}.svg')
        shutil.copyfile(ROOT / 'build/figures' / f'{name}.png', DATA / f'{op}.png')
    assert len(exported) == 212
    with (DATA / 'figure_inputs.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('operation', 'series', 'bytes', 'median_us', 'min_us',
                         'max_us', 'bus_GBps', 'source'))
        writer.writerows(exported)


if __name__ == '__main__':
    main()
