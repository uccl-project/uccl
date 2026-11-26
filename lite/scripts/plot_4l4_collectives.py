#!/usr/bin/env python3
"""Render drop-in UCCL Lite with the shared collective style and recorded NCCL controls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from figure_style import ROOT, clean_axis, collective_legend, collective_panels, configure, save

LOCAL = ROOT / 'benchmarks/data/nccl_local_strict_4l4_20261119'
MULTI = ROOT / 'benchmarks/data/uccl_reference_gap_20261118/reference_protocol/qualified_release'
OPS = [('ar', 'allreduce'), ('ag', 'allgather'), ('rs', 'reducescatter')]
STYLES = [('prior', '#D8A17A', 's', (0, (5, 2.5))),
          ('sm', '#82B4A3', 'D', (0, (2, 1.5))),
          ('ce', '#B899C8', '^', (0, (6, 2, 1, 2))),
          ('uccl', '#0068B5', 'o', '-')]
PANELS = [('latency', range(10, 21), 'Latency (µs)', (10, 12, 14, 16, 18, 20)),
          ('throughput', range(20, 31), 'Bus throughput (GB/s)', (20, 22, 24, 26, 28, 30))]


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def size_label(n):
    return f'{n >> 10}K' if n < 1 << 20 else f'{n >> 20}M' if n < 1 << 30 else '1G'


def points(topology, op, name):
    result = {}
    ranks = 4 if topology == '4l4' else 8
    factor = (2 if op == 'ar' else 1) * (ranks - 1) / ranks

    def add_us(series, size, us, source):
        for panel, powers, _, _ in PANELS:
            if size in [1 << p for p in powers]:
                value = us if panel == 'latency' else factor * size / us / 1000
                result[panel, series, size] = (value, source)

    if topology == '4l4':
        source = LOCAL / 'comparison.csv'
        for row in read(source):
            if row['operation'] == op and row['layout'] == 'out':
                add_us('uccl', int(row['bytes']), float(row['candidate_us']), source)
        source = ROOT / f'benchmarks/data/small_medium_4l4_20261113/{op}_selected.csv'
        for row in read(source):
            add_us('prior', int(row['bytes']), float(row['nccl_median_us']), source)
        source = ROOT / 'benchmarks/data/nccl_tuned_4l4_20261115/cases.csv'
        for row in read(source):
            if row['operation'] == name:
                for series in ['sm', 'ce']:
                    add_us(series, int(row['bytes']), float(row[series + '_median_us']), source)
    else:
        source = MULTI / 'nccl_tests_vs_old.csv'
        for row in read(source):
            if row['operation'] == op:
                add_us('uccl', int(row['bytes']), float(row['new_nccl_tests_out_us']), source)
        if op == 'ag':
            source = ROOT / 'benchmarks/data/ag_latency_2x4l4_20261116/figure_inputs.csv'
            for row in read(source):
                if row['series'] != 'uccl':
                    result[row['panel'], row['series'], int(row['bytes'])] = (float(row['value']), source)
        else:
            source = ROOT / 'benchmarks/data/nccl_tuned_2x4l4_20261116/figure_inputs.csv'
            for row in read(source):
                if row['operation'] == name and row['series'] != 'uccl':
                    add_us(row['series'], int(row['bytes']), float(row['median_us']), source)
    expected = {(panel, series, 1 << p) for panel, powers, _, _ in PANELS
                for series, *_ in STYLES for p in powers}
    assert set(result) == expected, (topology, op, expected - set(result))
    assert all(math.isfinite(v) and v > 0 for v, _ in result.values())
    return result


def plot(topology, op, name, data, preview):
    fig, axes = collective_panels()
    spike_axis = None
    if topology == '2x4l4' and op == 'ag':
        # Preserve the recorded NCCL spikes using the existing AG axis break.
        box = axes[0].get_position()
        axes[0].set_position([box.x0, box.y0, box.width, box.height - .082])
        spike_axis = fig.add_axes([box.x0, box.y0 + box.height - .06, box.width, .06], sharex=axes[0])
        clean_axis(spike_axis)
        spike_axis.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        spike_axis.set_ylim(500, 2100)
        spike_axis.set_yticks([600, 2000])
        for badge in axes[0].texts:
            badge.set_transform(spike_axis.transAxes)
            badge.set_position((1, 1.27))
        for ax, y in [(axes[0], 1), (spike_axis, 0)]:
            for x in [0, 1]:
                ax.plot([x - .014, x + .014], [y - .025, y + .025],
                        transform=ax.transAxes, color='#172A3A', linewidth=1.2, clip_on=False)
    exported = []
    for ax, (panel, powers, ylabel, ticks) in zip(axes, PANELS):
        sizes = [1 << p for p in powers]
        clean_axis(ax)
        ax.set_xscale('log', base=2)
        ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
        ax.set_xticks([1 << p for p in ticks], [size_label(1 << p) for p in ticks])
        ax.set_xlabel('Complete tensor bytes per rank')
        ax.set_ylabel(ylabel)
        upper = 0
        for series, color, marker, style in STYLES:
            label = {'uccl': 'UCCL Lite', 'sm': 'NCCL SM (tuned)', 'ce': 'NCCL CE (tuned)',
                     'prior': 'NCCL (default)' if topology == '4l4' else 'NCCL (prior)'}[series]
            values = [data[panel, series, n][0] for n in sizes]
            kwargs = dict(color=color, marker=marker, linestyle=style,
                          linewidth=3.6 if series == 'uccl' else 2.2,
                          markersize=6.5 if series == 'uccl' else 5,
                          markeredgecolor='white', markeredgewidth=.7, label=label,
                          zorder=4 if series == 'uccl' else 2)
            ax.plot(sizes, values, **kwargs)
            if panel == 'latency' and spike_axis is not None:
                assert all(v <= 400 or 500 <= v <= 2100 for v in values)
                spike_axis.plot(sizes, values, **kwargs)
            upper = max(upper, *values)
            for n, value in zip(sizes, values):
                exported.append(dict(topology=topology, operation=op, panel=panel, series=series,
                                     bytes=n, value=value, source=str(data[panel, series, n][1].relative_to(ROOT))))
        ax.set_ylim(0, 400 if panel == 'latency' and spike_axis is not None else upper * 1.13)
    collective_legend(fig, axes[0])
    filename = f'collectives-4l4-{name}-all-sizes' if topology == '4l4' else f'collectives-2x4l4-dropin-{name}'
    save(fig, filename, preview)
    return exported


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preview', action='store_true')
    p.add_argument('--topology', choices=['4l4', '2x4l4', 'all'], default='4l4')
    args = p.parse_args()
    configure(black_axes=True)
    for topology in (['4l4', '2x4l4'] if args.topology == 'all' else [args.topology]):
        rows = []
        for op, name in OPS:
            rows += plot(topology, op, name, points(topology, op, name), args.preview)
        directory = LOCAL if topology == '4l4' else MULTI
        with (directory / 'dropin_readme_inputs.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        sources = sorted({row['source'] for row in rows})
        metadata = dict(hardware='4 L4, one host' if topology == '4l4' else '2 hosts × 4 L4',
                        dtype='FP32', layout='out-of-place', uccl_series='fixed drop-in library; no per-size best-of selection',
                        uccl_timing=('three repetitions of ten per-call samples; AR host enqueue/completion (5 warmups), AG/RS max-GPU events (20 warmups); core2 placement'
                                     if topology == '4l4' else 'unchanged nccl-tests batch timing; three runs, 20 warmups, 30 timed iterations'),
                        nccl_controls='Recorded NCCL controls retained from existing figures; not rerun or newly matched to the drop-in harness. Multi-node prior curve is not uniformly default NCCL.',
                        throughput='bus-normalized: 2*(ranks-1)/ranks for AR; (ranks-1)/ranks for AG/RS; bytes / us / 1000',
                        size_definition='AG full output; RS full input; AR input/output per rank',
                        source_sha256={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
                        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), points=len(rows))
        (directory / 'dropin_readme_provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
