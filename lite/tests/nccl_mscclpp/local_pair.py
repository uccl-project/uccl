"""Paired four-L4 AR gate against frozen original UCCL; no sample trimming.

This AR-only gate complements the stock nccl-tests and ABI checks. AG/RS
require their original event-timing preparation protocol and are not certified
by this driver.
"""
import argparse
import collections
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
import mscclpp_profile as profile
import local_cpu_profile as placement


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True, help='frozen source directory with build/')
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--client', type=Path, required=True)
    p.add_argument('--real-nccl', type=Path, required=True)
    p.add_argument('--mpirun', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cpu-placement', choices=('unbound', 'numa', 'core', 'core2'), default='unbound')
    p.add_argument('--max-bytes', type=int, default=1 << 30)
    p.add_argument('--allocation-bytes', type=int, help='retain the full workspace while screening a size subset')
    args = p.parse_args()
    require(1024 <= args.max_bytes <= 1 << 30 and
            not args.max_bytes & (args.max_bytes - 1), 'invalid maximum size')
    allocation = args.allocation_bytes or args.max_bytes
    require(args.max_bytes <= allocation <= 1 << 30 and not allocation & (allocation - 1), 'invalid allocation size')
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['UCCL_LITE_MSCCLPP_CONTROL_ROOT'] = '127.0.0.1'
    library, real, client = args.library.resolve(), args.real_nccl.resolve(), args.client.resolve()
    source = args.reference.resolve()
    reference = source / 'build/uccl_lite_allreduce'
    expected_source = json.loads((ROOT / 'benchmarks/data/small_medium_4l4_20261113/identity.json').read_text())['sha256']['src/collectives/allreduce.cu']
    require(profile.common.sha256(source / 'src/collectives/allreduce.cu') == expected_source,
            'reference source differs from published source')
    record = placement.make_record(args.cpu_placement, real, library, args.out)
    (args.out / 'identity.json').write_text(json.dumps(record, indent=2) + '\n')
    candidate = (args.out / 'profile.tsv').resolve()
    sizes = {1 << n for n in range(10, args.max_bytes.bit_length())}
    profile.write_candidate(candidate, [record], [(3, (1 << n) // 4) for n in range(10, allocation.bit_length())])
    manifest = dict(reference_source_sha256=expected_source,
                    library_sha256=profile.common.sha256(library),
                    reference_sha256=profile.common.sha256(reference),
                    client_sha256=profile.common.sha256(client),
                    warmups=5, iterations=10, repeats=3, max_bytes=args.max_bytes, allocation_bytes=allocation,
                    timing='maximum rank, per-call host enqueue and stream completion',
                    reference_selection='lowest median of three untrimmed run means per correct original variant',
                    scope='AR only, FP32, four L4s', cpu_placement=args.cpu_placement, runs=[])
    for key, binary in [('reference', reference), ('client', client), ('library', library)]:
        manifest[key + '_ldd'] = subprocess.check_output(['ldd', str(binary)], text=True)
    launcher = [args.mpirun, '-np', '4', '--bind-to', 'none', '--mca', 'pml', 'ob1',
                '--mca', 'btl', 'self,vader,tcp', sys.executable, '-B',
                str(placement.launcher_path(args.cpu_placement)), 'launch',
                '--library', str(library), '--real-nccl', str(real),
                '--profile', str(candidate), '--', str(client)]
    samples = collections.defaultdict(list)
    for repeat in range(3):
        order = ['reference', 'out', 'in'] if repeat != 1 else ['in', 'out', 'reference']
        for kind in order:
            command = ([str(reference), '--sm-variants', '--fast-sweep', '-b', '1024',
                        '-e', str(args.max_bytes), '-w', '5', '-n', '10'] if kind == 'reference'
                       else launcher + ['ar', kind, '1024', str(args.max_bytes), '5', '10'])
            if allocation != args.max_bytes:
                command += ['--allocation-bytes', str(allocation)] if kind == 'reference' else [str(allocation)]
            label = f'{repeat}_{kind}'
            manifest['runs'].append(dict(label=label, command=command))
            (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print('running', label, flush=True)
            log = args.out / (label + '.log')
            with log.open('w') as output:
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=3600)
            manifest['runs'][-1]['exit_code'] = result.returncode
            (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            require(result.returncode == 0, label + ': process failed')
            rows = collections.defaultdict(dict)
            text = log.read_text()
            if kind != 'reference':
                placement.validate_run(args.out, args.cpu_placement, command, text)
            for line in text.splitlines():
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
                require(size in sizes and 0 <= index < 10 and
                        math.isfinite(latency) and latency > 0 and
                        math.isfinite(error) and 0 <= error <= 1e-5,
                        label + ': invalid sample')
                require(index not in rows[size, variant], label + ': duplicate sample')
                rows[size, variant][index] = latency
            require({size for size, _ in rows} == sizes, label + ': incomplete extent coverage')
            if kind == 'reference':
                expected_variants = set()
                for size in sizes:
                    variants = {'A3-ring-ce'}
                    if size <= 4 << 20:
                        variants.update({'A1-hier-sm', 'A1-hier-ce', 'A1-hier-cp',
                                         'A2-tree-sm', 'A2-tree-ce', 'A5-rsflat-ce',
                                         'A1-hier-sm-1cta', 'A1-hier-sm-8cta'})
                        if size <= 16 << 10:
                            variants.update({'A1-hier-ll', 'A2-tree-ll'})
                        if size <= 64 << 10:
                            variants.add('A4-flat-ll')
                    expected_variants.update((size, variant) for variant in variants)
                require(set(rows) == expected_variants, label + ': missing original variants')
            if kind != 'reference':
                require('UCCL reference-protocol checks passed' in text, label + ': missing correctness')
                for rank in range(4):
                    require(f'rank={rank} native_nccl_loaded=0' in text and
                            f'rank={rank} fallback=0' in text, label + ': native/fallback')
                    for size in sizes:
                        require(f'rank={rank} op=3 bytes={size} calls=15\n' in text,
                                label + ': route counts')
            for key, values in rows.items():
                require(set(values) == set(range(10)), label + ': missing samples')
                samples[key].append(statistics.mean(values.values()))
    output = []
    for size in sorted(sizes):
        refs = {variant: statistics.median(values) for (n, variant), values in samples.items()
                if n == size and variant not in ('out', 'in') and len(values) == 3}
        require(refs, 'no complete original variants')
        best = min(refs, key=refs.get)
        for layout in ('out', 'in'):
            values = samples[size, layout]
            require(len(values) == 3, 'missing candidate repeats')
            candidate_us = statistics.median(values)
            output.append(dict(bytes=size, layout=layout, reference_variant=best,
                               reference_us=refs[best], candidate_us=candidate_us,
                               ratio=candidate_us / refs[best]))
    with (args.out / 'comparison.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    failures = [r for r in output if r['ratio'] > 1]
    result = dict(scope=manifest['scope'], slower_rows=len(failures), rows=len(output),
                  performance='fail' if failures else 'pass', failures=failures,
                  full_goal_complete=False)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'{len(failures)}/{len(output)} AR rows slower than original UCCL', flush=True)
    return 2 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
