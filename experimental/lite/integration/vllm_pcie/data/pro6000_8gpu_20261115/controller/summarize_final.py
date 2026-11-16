"""Report measured fine performance separately from EAT screening scores."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/final'


def read(path):
    return json.loads(path.read_text())


def gm(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def quality(path):
    q = read(path)
    assert q['num_questions'] == 1319 and q['num_shots'] == 5 and q['max_tokens'] == 256
    return dict(correct=round(q['accuracy'] * 1319), total=1319,
                invalid_rate=q['invalid_rate'], accuracy=q['accuracy'], artifact=str(path))


def main():
    state = read(ROOT / 'state.json')
    assert state['completed_loops'] == 10
    roots = {'nccl': ROOT / 'results/final-validation/nccl',
             'original': ROOT / 'results/final-original/pcieccl',
             'best': ROOT / 'results/final-validation/pcieccl'}
    for folder in ('final-validation', 'final-original'):
        assert read(ROOT / 'results' / folder / 'COMPLETE.json')['complete']
    for tag in ('best', 'original'):
        assert read(ROOT / f'results/final-audit/{tag}-before.json') == read(ROOT / f'results/final-audit/{tag}-after.json')
    rows = []
    for bs in (1, 2, 4, 8, 16, 32, 64):
        row = {'concurrency': bs}
        for label, root in roots.items():
            d = read(root / 'cases' / f'isl8192-osl1024-bs{bs}.json')
            m = d['metrics']
            assert d['status'] == 'complete' and m['request_errors'] == 0
            assert d['case']['warmup_seconds'] == 60 and d['case']['measurement_seconds'] == 180
            row[label + '_tok_s'] = m['output_token_throughput']
            row[label + '_tpot_ms'] = m['tpot_ms']['p50']
            row[label + '_ttft_ms'] = m['ttft_ms']['p50']
        row['original_vs_nccl'] = row['original_tok_s'] / row['nccl_tok_s']
        row['best_vs_nccl'] = row['best_tok_s'] / row['nccl_tok_s']
        row['best_vs_original'] = row['best_tok_s'] / row['original_tok_s']
        rows.append(row)
    q = {
        'nccl': quality(ROOT / 'results/final-validation/quality/gsm8k-nccl.json'),
        'original': quality(ROOT / 'results/quality/gsm8k-pcieccl.json'),
        'best': quality(ROOT / 'results/final-validation/quality/gsm8k-pcieccl.json'),
    }
    result = {'platform': 'wxz-pro6000, RTX PRO 6000 Blackwell x8',
              'model': 'MiniMax-M2.7 FP8, TP8+EP', 'loops': 10,
              'incumbent': state['incumbent'], 'rows': rows,
              'throughput_geomean': {k: gm([r[k] for r in rows]) for k in ('original_vs_nccl', 'best_vs_nccl', 'best_vs_original')},
              'quality': q,
              'quality_repeats_required': q['best']['correct'] < q['original']['correct'],
              'limitations': ['One final fine matrix per implementation; no confidence interval.',
                             'EAT finite-wave scores and original fine steady-state throughput are distinct metrics.',
                             'NCCL uses original experiment settings: custom/symmetric-memory AR disabled; transport selection unforced.',
                             'Original PCIeCCL quality is the initial reproduction sample, per frozen repeat policy.',
                             'Construction uses a 70% shared prefix; this is not a measured cache hit rate.']}
    if not result['quality_repeats_required']:
        result['quality_confirmation'] = {'status': 'no_observed_accuracy_reduction_in_first_sample',
                                          'statistical_equivalence_claimed': False}
    elif not (ROOT / 'results/quality-repeats/quality-summary.json').exists():
        result['quality_confirmation'] = {'status': 'pending_required_repeats'}
    else:
        confirmation = read(ROOT / 'results/quality-repeats/quality-summary.json')
        assert confirmation['correct_counts']['original'][0] == q['original']['correct']
        assert confirmation['correct_counts']['best'][0] == q['best']['correct']
        assert confirmation['status'] in ('quality-unconfirmed', 'no-observed-quality-decline')
        result['quality_confirmation'] = confirmation
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    with (OUT / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    colors = {'nccl': '#556270', 'original': '#1877b8', 'best': '#d45b22'}
    labels = {'nccl': 'NCCL baseline', 'original': 'Original PCIeCCL', 'best': 'PCIeCCL after 10 EAT loops'}
    xs = [r['concurrency'] for r in rows]
    for key in roots:
        for ax, metric in zip(axes[:2], ('tok_s', 'tpot_ms')):
            ax.plot(xs, [r[key + '_' + metric] for r in rows], 'o-', color=colors[key], label=labels[key])
    axes[2].plot(xs, [100 * (r['best_vs_original'] - 1) for r in rows], 'o-', color=colors['best'], label='After EAT vs original PCIeCCL')
    axes[2].axhline(0, color=colors['nccl'], linewidth=1, linestyle='--')
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_xlabel('Concurrent requests')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Output tokens / second (higher is better)')
    axes[1].set_ylabel('Median TPOT, ms (lower is better)')
    axes[2].set_ylabel('Additional throughput from EAT (%)')
    axes[0].legend(fontsize=8, loc='upper left')
    axes[2].legend(fontsize=8)
    quality_status = result['quality_confirmation']['status']
    quality_note = {
        'pending_required_repeats': 'Accuracy verdict pending required repeats',
        'quality-unconfirmed': 'Quality acceptance not met under the declared GSM8K protocol',
        'no_observed_accuracy_reduction_under_repeat_protocol': 'No observed median GSM8K reduction under the three-run comparison',
        'no-observed-quality-decline': 'No observed median GSM8K reduction under the three-run comparison',
        'no_observed_accuracy_reduction_in_first_sample': 'No observed GSM8K reduction in the first quality sample',
    }[quality_status]
    fig.suptitle('MiniMax-M2.7 · 8× RTX PRO 6000 · TP8+EP · 8192 input / 1024 output tokens\nOriginal fine protocol: 60 s warmup + 180 s measurement per case\n' + quality_note, fontsize=11)
    fig.savefig(OUT / 'performance.png', dpi=180)
    fig.savefig(OUT / 'performance.svg')
    print(json.dumps({'throughput_geomean': result['throughput_geomean'],
                      'quality_status': result['quality_confirmation']['status'],
                      'median_correct': result['quality_confirmation'].get('median_correct'),
                      'quality_repeats_required': result['quality_repeats_required']}, indent=2))


if __name__ == '__main__':
    main()
