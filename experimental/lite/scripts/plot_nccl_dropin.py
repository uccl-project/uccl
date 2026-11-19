#!/usr/bin/env python3
"""Plot the qualified .so's stock nccl-tests results and old standalone data."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from figure_style import BLUE, TEAL, BASELINE, MUTED, ROOT
from figure_style import clean_axis, configure, save

DATA = ROOT / 'benchmarks/data/uccl_reference_gap_20261118/reference_protocol/qualified_release'
INPUT = DATA / 'nccl_tests_vs_old.csv'
SERIES = [
    ('new_nccl_tests_out_us', 'New .so · nccl-tests · out', BLUE, 'o', '-'),
    ('new_nccl_tests_in_us', 'New .so · nccl-tests · in', TEAL, 's', '--'),
    ('old_uccl_standalone_us', 'Old UCCL Lite · standalone', BASELINE, '^', ':'),
]
OPERATIONS = [('ar', 'AllReduce', 'allreduce'),
              ('ag', 'AllGather', 'allgather'),
              ('rs', 'ReduceScatter', 'reducescatter')]


def size_label(value):
    if value >= 1 << 30:
        return '1 GiB'
    if value >= 1 << 20:
        return f'{value >> 20} MiB'
    return f'{value >> 10} KiB'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    records = list(csv.DictReader(INPUT.open()))
    expected = {(op, 1 << power) for op, _, _ in OPERATIONS
                for power in range(10, 31)}
    rows = {(r['operation'], int(r['bytes'])): r for r in records}
    assert len(records) == len(rows) == 63 and set(rows) == expected
    assert all(math.isfinite(float(r[field])) and float(r[field]) > 0
               for r in records for field, *_ in SERIES)
    configure(black_axes=True)
    exported = []
    for op, title, filename in OPERATIONS:
        fig, axes = plt.subplots(2, 1, figsize=(4.8, 6.6))
        fig.subplots_adjust(left=.19, right=.97, top=.73, bottom=.15,
                            hspace=.65)
        fig.text(.5, .985, title, ha='center', va='top', fontsize=16,
                 fontweight='bold')
        fig.text(.5, .941, '2 × 4 L4 · FP32 · three-run medians',
                 ha='center', fontsize=10)
        for ax, panel, powers, ticks in [
            (axes[0], 'latency', range(10, 21), [10, 12, 14, 16, 18, 20]),
            (axes[1], 'throughput', range(20, 31), [20, 22, 24, 26, 28, 30]),
        ]:
            clean_axis(ax)
            sizes = [1 << power for power in powers]
            ax.set_xscale('log', base=2)
            ax.set_xlim(sizes[0] / 1.15, sizes[-1] * 1.15)
            ax.set_xticks([1 << power for power in ticks],
                          [size_label(1 << power).replace('iB', '')
                           for power in ticks])
            ax.tick_params(axis='both', labelsize=10)
            ax.set_xlabel('Complete buffer size', fontsize=11)
            ax.set_ylabel('Latency (µs)' if panel == 'latency'
                          else 'Effective throughput (GB/s)', fontsize=11)
            ax.set_title('↓ Lower is better' if panel == 'latency'
                         else '↑ Higher is better', loc='right', fontsize=10)
            upper = 0
            for field, label, color, marker, line in SERIES:
                latency = [float(rows[op, n][field]) for n in sizes]
                # Common payload-rate metric, not nccl-tests' bus-bandwidth
                # normalization: complete-buffer bytes / measured latency.
                values = latency if panel == 'latency' else [
                    n / (us * 1000) for n, us in zip(sizes, latency)]
                ax.plot(sizes, values, label=label, color=color, marker=marker,
                        linestyle=line, linewidth=2.2, markersize=4.5,
                        markeredgecolor='white', markeredgewidth=.5)
                upper = max(upper, *values)
                for n, us, value in zip(sizes, latency, values):
                    exported.append({'operation': op, 'panel': panel,
                                     'series': field, 'bytes': n,
                                     'latency_us': us, 'plotted_value': value})
            ax.set_ylim(0, upper * 1.12)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(.5, .92), ncol=1, frameon=False,
                   fontsize=10, handlelength=2.4, labelspacing=.35)
        fig.text(.5, .064, 'Different harness / timing scopes',
                 ha='center', fontsize=9, color=MUTED)
        fig.text(.5, .035, 'Visual gaps are not matched speedup ratios',
                 ha='center', fontsize=8.5, color=MUTED)
        save(fig, f'collectives-2x4l4-dropin-{filename}', args.preview)
    assert len(exported) == 198
    with (DATA / 'readme_figure_inputs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    provenance = {
        'source': str(INPUT.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_harness': 'unchanged nccl-tests, maximum-rank batch timing',
        'old_harness': 'original UCCL standalone, per-call full retirement',
        'new_warmups': 20, 'new_timed_iterations': 30, 'repetitions': 3,
        'old_warmups_iterations': {'ag': [20, 30], 'ar': [20, 10], 'rs': [10, 10]},
        'new_layouts': ['out-of-place', 'in-place'],
        'old_layout': 'original recorded reference layout',
        'new_placement': '8 MPI ranks, 1 GPU/rank, 2 hosts × 4 L4',
        'old_placement': 'AG 4 MPI ranks × 2 GPUs; AR/RS 8 × 1 GPU',
        'throughput': 'complete-buffer bytes / latency_us / 1000, decimal GB/s',
        'size_definition': 'AG complete output; RS complete input; AR buffer',
        'best_per_size_envelope': False, 'native_nccl_fallback': False,
        'points': len(exported),
    }
    (DATA / 'readme_figure_provenance.json').write_text(
        json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
