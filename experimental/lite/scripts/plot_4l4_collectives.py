#!/usr/bin/env python3
"""Plot four-L4 collective summaries and independently validated NCCL tuning."""
import csv
import math

import matplotlib
matplotlib.use("Agg")

from figure_style import (ROOT, clean_axis, collective_legend, collective_panels,
                          configure, save as save_figure)

TUNED = ROOT / "benchmarks/data/nccl_tuned_4l4_20261115/cases.csv"

def plot(rows, name, timer, factor):
    by_size = {row['bytes']: row for row in rows}
    with TUNED.open() as stream:
        tuned = {int(row["bytes"]): row for row in csv.DictReader(stream)
                 if row["operation"] == name.lower()}
    assert sorted(tuned) == [1 << power for power in range(10, 31)]
    assert len(by_size) == len(rows), f'duplicate sizes: {name}'
    latency_sizes = [1 << power for power in range(10, 21)]
    throughput_sizes = [1 << power for power in range(20, 31)]
    assert all(size in by_size for size in latency_sizes + throughput_sizes)
    fig, axes = collective_panels()
    for ax, sizes, measure, ylabel, ticks in (
        (axes[0], latency_sizes, lambda size, us: us, 'Latency (µs)',
         [1 << power for power in (10, 12, 14, 16, 18, 20)]),
        (axes[1], throughput_sizes, lambda size, us: factor * size / us / 1000,
         'Bus throughput (GB/s)',
         [1 << power for power in (20, 22, 24, 26, 28, 30)]),
    ):
        clean_axis(ax)
        ax.set_xscale('log', base=2)
        ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
        ax.set_xticks(ticks, [size_name(size).replace(' KiB', 'K').replace(' MiB', 'M').replace(' GiB', 'G')
                             for size in ticks])
        ax.set_xlabel('Complete tensor bytes per rank')
        ax.set_ylabel(ylabel)
        upper = 0.0
        for key, label, color, marker, style in (
            ('nccl_median_us', 'NCCL (default)', '#D8A17A', 's', (0, (5, 2.5))),
            ('sm_median_us', 'NCCL SM (tuned)', '#82B4A3', 'D', (0, (2, 1.5))),
            ('ce_median_us', 'NCCL CE (tuned)', '#B899C8', '^', (0, (6, 2, 1, 2))),
            ('uccl_median_us', 'UCCL Lite', '#0068B5', 'o', '-'),
        ):
            selected = tuned if key.startswith(('sm_', 'ce_')) else by_size
            values = [measure(size, float(selected[size][key])) for size in sizes]
            prefix = key.removesuffix('median_us')
            bounds = [[measure(size, float(selected[size][prefix + suffix]))
                       for suffix in ('min_us', 'max_us')] for size in sizes]
            low = [min(pair) for pair in bounds]
            high = [max(pair) for pair in bounds]
            assert len(values) == 11 and all(math.isfinite(v) and v > 0 for v in values)
            upper = max(upper, *high)
            is_uccl = key == 'uccl_median_us'
            if key != 'ce_median_us':
                ax.fill_between(sizes, low, high, color=color,
                                alpha=.12 if is_uccl else .08, linewidth=0)
            ax.plot(sizes, values, color=color, linewidth=3.6 if is_uccl else 2.2,
                    marker=marker, linestyle=style, markersize=6.5 if is_uccl else 5,
                    markeredgecolor='white', markeredgewidth=.7, label=label,
                    zorder=4 if is_uccl else 2)
        ax.set_ylim(0, upper * 1.13)
    collective_legend(fig, axes[0])
    save_figure(fig, f'collectives-4l4-{name.lower()}-all-sizes', preview=True)

def size_name(n):
    return f'{n//2**30} GiB' if n>=2**30 else f'{n//2**20} MiB' if n>=2**20 else f'{n//1024} KiB' if n>=1024 else f'{n} B'


if __name__ == "__main__":
    configure()
    data = ROOT / "benchmarks/data/small_medium_4l4_20261113"
    for prefix, name, timer, factor in (
        ("ar", "AllReduce", "4 processes, host timer", 1.5),
        ("ag", "AllGather", "1 process, eager GPU events", .75),
        ("rs", "ReduceScatter", "1 process, eager GPU events", .75),
    ):
        with (data / f"{prefix}_selected.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["bytes"] = int(row["bytes"])
        plot(rows, name, timer, factor)
