"""Independently audit the completed four-L4 AR paired campaign."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'integration/nccl_lite'))
import dropin_profile
import local_cpu_profile as placement


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit(directory, max_bytes=1 << 30):
    manifest = json.loads((directory / 'manifest.json').read_text())
    require(manifest['warmups'] == 5 and manifest['iterations'] == 10 and
            manifest['repeats'] == 3, 'timing protocol changed')
    runs = {run['label']: run for run in manifest['runs']}
    require(len(manifest['runs']) == 9 and set(runs) ==
            {f'{i}_{kind}' for i in range(3) for kind in ('reference', 'out', 'in')},
            'missing/duplicate runs')
    samples = collections.defaultdict(list)
    require(1024 <= max_bytes <= 1 << 30 and not max_bytes & (max_bytes - 1), 'invalid maximum size')
    sizes = {1 << n for n in range(10, max_bytes.bit_length())}
    allocation = manifest.get('allocation_bytes', max_bytes)
    require(max_bytes <= allocation <= 1 << 30 and not allocation & (allocation - 1), 'invalid allocation')
    reference_suffix = ['--allocation-bytes', str(allocation)] if allocation != max_bytes else []
    candidate_suffix = [str(allocation)] if allocation != max_bytes else []
    for label, run in runs.items():
        kind = label.split('_', 1)[1]
        cmd = run['command']
        require(run['exit_code'] == 0, label + ': failed process')
        if kind == 'reference':
            require(cmd[1:] == ['--sm-variants', '--fast-sweep', '-b', '1024',
                                '-e', str(max_bytes), '-w', '5', '-n', '10'] + reference_suffix,
                    label + ': original command changed')
            require(dropin_profile.sha256(cmd[0]) == manifest['reference_sha256'],
                    label + ': original binary changed')
            source = Path(cmd[0]).parents[1] / 'src/collectives/allreduce.cu'
            require(dropin_profile.sha256(source) == manifest['reference_source_sha256'],
                    label + ': original source changed')
        else:
            require(cmd[cmd.index('-np') + 1] == '4' and
                    cmd[cmd.index('--') + 2:] == ['ar', kind, '1024', str(max_bytes), '5', '10'] + candidate_suffix,
                    label + ': candidate command changed')
            require(dropin_profile.sha256(cmd[cmd.index('--') + 1]) == manifest['client_sha256'],
                    label + ': client changed')
            require(dropin_profile.sha256(cmd[cmd.index('--library') + 1]) == manifest['library_sha256'],
                    label + ': candidate changed')
        text = (directory / (label + '.log')).read_text()
        if kind != 'reference':
            placement.validate_run(directory, manifest.get('cpu_placement', 'unbound'), cmd, text)
        rows = collections.defaultdict(dict)
        summaries = {}
        for line in text.splitlines():
            fields = line.split()
            if kind == 'reference' and len(fields) == 7 and fields[0].isdigit():
                if fields[2].startswith('A0'): continue
                error = float(fields[-1])
                require(math.isfinite(error) and 0 <= error <= 1e-5,
                        label + ': warmup/timed reference error')
                summaries[int(fields[0]), fields[2]] = True
            if kind == 'reference' and line.startswith('# sample,'):
                _, size, variant, index, latency, error = line.split(',')
                if variant.startswith('A0'): continue
                index = int(index) - 1
            elif kind != 'reference' and line.startswith('sample,'):
                _, op, layout, size, index, latency, error = line.split(',')
                require(op == 'ar' and layout == kind, label + ': wrong operation/layout')
                variant, index = kind, int(index)
            else:
                continue
            size, latency, error = int(size), float(latency), float(error)
            require(size in sizes and index not in rows[size, variant] and
                    math.isfinite(latency) and latency > 0 and math.isfinite(error)
                    and 0 <= error <= 1e-5, label + ': invalid sample')
            rows[size, variant][index] = latency
        expected = set()
        for size in sizes:
            variants = {kind}
            if kind == 'reference':
                variants = {'A3-ring-ce'}
                if size <= 4 << 20:
                    variants.update({'A1-hier-sm', 'A1-hier-ce', 'A1-hier-cp',
                                     'A2-tree-sm', 'A2-tree-ce', 'A5-rsflat-ce',
                                     'A1-hier-sm-1cta', 'A1-hier-sm-8cta'})
                    if size <= 16 << 10: variants.update({'A1-hier-ll', 'A2-tree-ll'})
                    if size <= 64 << 10: variants.add('A4-flat-ll')
            expected.update((size, variant) for variant in variants)
        require(set(rows) == expected, label + ': missing variants/sizes')
        if kind == 'reference':
            require(set(summaries) == expected, label + ': missing reference correctness summaries')
        else:
            require('UCCL reference-protocol checks passed' in text, label + ': correctness')
            for rank in range(4):
                require(text.count(f'rank={rank} native_nccl_loaded=0') == 1 and
                        text.count(f'rank={rank} fallback=0') == 1, label + ': fallback/native')
                for size in sizes:
                    require(text.count(f'rank={rank} op=3 bytes={size} calls=15\n') == 1,
                            label + ': routing')
        for key, values in rows.items():
            require(set(values) == set(range(10)), label + ': missing sample indices')
            samples[key].append(statistics.mean(values.values()))
    recorded = list(csv.DictReader((directory / 'comparison.csv').open()))
    require(len(recorded) == 2 * len(sizes) and {(int(r['bytes']), r['layout']) for r in recorded} ==
            {(size, layout) for size in sizes for layout in ('out', 'in')}, 'comparison rows')
    slower = []
    for row in recorded:
        size, layout = int(row['bytes']), row['layout']
        refs = {variant: statistics.median(values) for (n, variant), values in samples.items()
                if n == size and variant not in ('out', 'in')}
        require(all(len(values) == 3 for values in samples.values()), 'missing repeats')
        best = min(refs, key=refs.get)
        candidate = statistics.median(samples[size, layout])
        require(row['reference_variant'] == best and float(row['reference_us']) == refs[best]
                and float(row['candidate_us']) == candidate
                and float(row['ratio']) == candidate / refs[best], 'stored comparison mismatch')
        if candidate > refs[best]: slower.append(row)
    return dict(scope='AR paired gate only', max_bytes=max_bytes, allocation_bytes=allocation, rows=2 * len(sizes), slower_rows=len(slower), failures=slower)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--max-bytes', type=int, default=1 << 30)
    args = parser.parse_args()
    result = audit(args.directory, args.max_bytes)
    print(json.dumps(result, indent=2))
    sys.exit(2 if result['slower_rows'] else 0)
