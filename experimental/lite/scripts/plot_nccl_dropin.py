#!/usr/bin/env python3
"""Plot qualified drop-in per-call results with the recorded NCCL controls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

from figure_style import ROOT
from figure_style import clean_axis, collective_panels, configure, save

DATA = ROOT / 'benchmarks/data/uccl_reference_gap_20261118/reference_protocol/qualified_release'
INPUT = DATA / 'paired_comparison.csv'
NCCL_INPUT = ROOT / 'benchmarks/data/nccl_tuned_2x4l4_20261116/figure_inputs.csv'
AG_INPUT = ROOT / 'benchmarks/data/ag_combined_latest_2x4l4_20261117/readme_figure_inputs.csv'
SERIES = [
    ('uccl', 'UCCL Lite', '#0068B5', 'o', '-'),
    ('prior', 'NCCL (default)', '#D8A17A', 's', (0, (5, 2.5))),
    ('sm', 'NCCL SM (tuned)', '#82B4A3', 'D', (0, (2, 1.5))),
    ('ce', 'NCCL CE (tuned)', '#B899C8', '^', (0, (6, 2, 1, 2))),
]
OPERATIONS = [('ar', 'AllReduce', 'allreduce'),
              ('ag', 'AllGather', 'allgather'),
              ('rs', 'ReduceScatter', 'reducescatter')]


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def figure_rows():
    records = read_csv(INPUT)
    expected = {(op, 1 << power, layout) for op, _, _ in OPERATIONS
                for power in range(10, 31) for layout in ('out', 'in')}
    candidates = {(r['operation'], int(r['bytes']), r['layout']): r
                  for r in records}
    assert len(records) == len(candidates) == 126 and set(candidates) == expected
    controls = read_csv(NCCL_INPUT)
    ag_controls = read_csv(AG_INPUT)
    exported = []
    for op, _, filename in OPERATIONS:
        factor = 1.75 if op == 'ar' else 0.875
        for panel, powers in [('latency', range(10, 21)),
                              ('throughput', range(20, 31))]:
            metric = 'median_us' if panel == 'latency' else 'bus_GBps'
            sizes = [1 << power for power in powers]
            for key, *_ in SERIES:
                if key != 'uccl':
                    baseline_rows = [
                        r for r in (ag_controls if op == 'ag' else controls)
                        if r['series'] == key and (
                            r['panel'] == panel if op == 'ag'
                            else r['operation'] == filename)]
                    baseline = {int(r['bytes']): r for r in baseline_rows}
                    assert len(baseline) == len(baseline_rows)
                    assert set(sizes) <= baseline.keys()
                for n in sizes:
                    if key == 'uccl':
                        us = float(candidates[op, n, 'out']['candidate_us'])
                        value = us if panel == 'latency' else factor * n / us / 1000
                        source = INPUT.relative_to(ROOT).as_posix()
                        source_series = 'candidate_us:out'
                    else:
                        row = baseline[n]
                        value = float(row['value'] if op == 'ag' else row[metric])
                        source = row['source']
                        source_series = key
                    assert math.isfinite(value) and value > 0
                    exported.append(dict(operation=op, panel=panel, series=key,
                                         bytes=n, plotted_value=value,
                                         source=source, source_series=source_series))
    assert len(exported) == 264
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    exported = figure_rows()
    configure(black_axes=True)
    for op, _, filename in OPERATIONS:
        fig, axes = collective_panels()
        spikes = None
        if op == 'ag':
            # Retain the genuine NCCL spikes without flattening normal latency.
            latency = axes[0]
            box = latency.get_position()
            gap, spike_height = .022, .060
            normal_height = box.height - gap - spike_height
            latency.set_position([box.x0, box.y0, box.width, normal_height])
            spikes = fig.add_axes(
                [box.x0, box.y0 + normal_height + gap, box.width, spike_height],
                sharex=latency)
            clean_axis(spikes)
            spikes.tick_params(axis='both', labelsize=11)
            spikes.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            spikes.set_ylim(500, 2100)
            spikes.set_yticks([600, 2000])
            for badge in latency.texts:
                badge.set_transform(spikes.transAxes)
                badge.set_position((1, 1.27))
            for ax, y in ((latency, 1), (spikes, 0)):
                for x in (0, 1):
                    ax.plot([x - .014, x + .014], [y - .025, y + .025],
                            transform=ax.transAxes, color='#172A3A',
                            linewidth=1.2, clip_on=False)
        for ax, panel, powers, ticks in [
            (axes[0], 'latency', range(10, 21), [10, 12, 14, 16, 18, 20]),
            (axes[1], 'throughput', range(20, 31), [20, 22, 24, 26, 28, 30]),
        ]:
            clean_axis(ax)
            sizes = [1 << power for power in powers]
            ax.set_xscale('log', base=2)
            ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
            ax.set_xticks([1 << power for power in ticks],
                          [f'{1 << (p - 10)}K' if p < 20 else
                           f'{1 << (p - 20)}M' if p < 30 else '1G'
                           for p in ticks])
            ax.set_xlabel('Complete tensor bytes per rank')
            ax.set_ylabel('Latency (µs)' if panel == 'latency'
                          else 'Bus throughput (GB/s)')
            ax.yaxis.set_label_coords(-.16, .5)
            upper = 0
            for key, label, color, marker, line in SERIES:
                points = {r['bytes']: r['plotted_value'] for r in exported
                          if (r['operation'], r['panel'], r['series']) ==
                          (op, panel, key)}
                assert sorted(points) == sizes
                values = [points[n] for n in sizes]
                style = dict(label=label, color=color, marker=marker,
                             linestyle=line, linewidth=3.6 if key == 'uccl' else 2.2,
                             markersize=6.5 if key == 'uccl' else 5,
                             markeredgecolor='white', markeredgewidth=.7)
                ax.plot(sizes, values, **style)
                if spikes is not None and panel == 'latency':
                    assert all(0 <= v <= 400 or 500 <= v <= 2100 for v in values)
                    spikes.plot(sizes, values, **style)
                upper = max(upper, *values)
            if spikes is not None and panel == 'latency':
                ax.set_ylim(0, 400)
                ax.set_yticks([0, 100, 200, 300, 400])
            else:
                ax.set_ylim(0, upper * 1.13)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(.5, .995), ncol=2, frameon=False,
                   fontsize=11.5, handlelength=1.6, handletextpad=.5,
                   columnspacing=.8, labelspacing=.4)
        save(fig, f'collectives-2x4l4-dropin-{filename}', args.preview)
    with (DATA / 'readme_figure_inputs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(exported)
    provenance = {
        'sources': {p.relative_to(ROOT).as_posix():
                    hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (INPUT, NCCL_INPUT, AG_INPUT)},
        'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'uccl_harness': 'aligned maximum-rank per-call mean, full send retirement',
        'uccl_warmups_iterations': {'ag': [20, 30], 'ar': [20, 10], 'rs': [10, 10]},
        'repetitions': 3,
        'uccl_layout': 'out-of-place',
        'uccl_placement': '8 MPI ranks, 1 GPU/rank, 2 hosts × 4 L4',
        'nccl_controls': 'unchanged historical controls, separate measurement windows',
        'nccl_placement': 'AG 4 MPI ranks × 2 GPUs; AR/RS 8 × 1 GPU',
        'throughput': 'AR: 1.75*N/t; AG/RS: 0.875*N/t; decimal GB/s',
        'size_definition': 'AG complete output; RS complete input; AR buffer',
        'best_per_size_envelope': False, 'native_nccl_fallback': False,
        'points': len(exported),
    }
    (DATA / 'readme_figure_provenance.json').write_text(
        json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
