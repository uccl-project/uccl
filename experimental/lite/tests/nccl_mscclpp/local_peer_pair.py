"""Bracketed four-L4 AG/RS comparison with the frozen original peer harness.

Two original three-round sweeps bracket three candidate runs per layout.
Acceptance uses the faster original sweep median for each variant, then the
best variant. No samples are removed. Original: one process/four GPUs;
candidate: four processes. Both use maximum-GPU per-call CUDA events.
"""
import argparse
import collections
import csv
import json
import math
import os
from pathlib import Path
import re
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


def variants(op, size):
    result = {'peer', 'nccl'}
    if size <= 32768: result.add('pull')
    if op == 'ag':
        if size <= 2 << 20: result.add('push-serial')
        if size <= 16 << 20: result.add('mini-host')
        if size >= 4 << 20: result.add('packet-router')
    else:
        result.add('scatter-ring')
        if size <= 8 << 20: result.add('push-serial')
        if 32 <= size <= 8192: result.add('scatter-ll')
        if 32 <= size <= 32768: result.add('scatter-hier-ll')
        if size <= 1 << 20: result.add('scatter-fence')
    return result


def reference_args(op, maximum):
    extra = (['--packet-router-ag', '--packet-router-per-source',
              '--mini-host-max-bytes', '16777216', '--push-grid-blocks', '16']
             if op == 'ag' else ['--rs-only', '--rs-ring', '--rs-hier-ll-max-bytes', '32768'])
    return extra + ['-b', '16', '-e', str(maximum), '-w', '20', '-n', '10', '-r', '3']


def audit(directory):
    m = json.loads((directory / 'manifest.json').read_text())
    op, maximum = m['operation'], m['max_bytes']
    longop, number = ('all_gather', 1) if op == 'ag' else ('reduce_scatter', 2)
    sizes = {1 << n for n in range(10, maximum.bit_length())}
    ref_sizes = {1 << n for n in range(4, maximum.bit_length())}
    expected_source = json.loads((ROOT / 'benchmarks/data/small_medium_4l4_20261113/identity.json').read_text())['sha256']['src/collectives/peer_ag_rs.cu']
    require(profile.common.sha256(Path(m['reference']).parents[1] / 'src/collectives/peer_ag_rs.cu') == expected_source,
            'reference source differs from published source')
    for key in ('library', 'client', 'reference'):
        require(profile.common.sha256(m[key]) == m[key + '_sha256'], key + ' binary changed')
    expected_labels = {'reference_before', 'reference_after'} | {f'{i}_{layout}' for i in range(3) for layout in ('out', 'in')}
    require(len(m['runs']) == 8 and {r['label'] for r in m['runs']} == expected_labels,
            'missing/duplicate runs')
    refs, candidate = collections.defaultdict(list), collections.defaultdict(list)
    for run in m['runs']:
        label, command = run['label'], run['command']
        require(run['exit_code'] == 0, label + ': failed process')
        text = (directory / (label + '.log')).read_text()
        rows = collections.defaultdict(dict)
        if label.startswith('reference'):
            require(command == [m['reference']] + reference_args(op, maximum), label + ': reference command changed')
            for line in text.splitlines():
                if not line.startswith('# sample,'): continue
                _, operation, size, variant, repeat, index, latency = line.split(',')
                require(operation == longop, label + ': wrong operation')
                key = int(size), variant, int(repeat)
                require(int(index) not in rows[key], label + ': duplicate sample')
                rows[key][int(index)] = float(latency)
            expected = {(size, variant, repeat) for size in ref_sizes for variant in variants(op, size) for repeat in (1, 2, 3)}
            require(set(rows) == expected, label + ': missing reference variants/sizes/repeats')
            for key, values in rows.items():
                require(set(values) == set(range(1, 11)) and all(math.isfinite(x) and x > 0 for x in values.values()), label + ': invalid samples')
            for size in sizes:
                for variant in variants(op, size) - {'nccl'}:
                    refs[size, variant].append(statistics.median(
                        statistics.mean(rows[size, variant, repeat].values()) for repeat in (1, 2, 3)))
        else:
            layout = label.split('_')[1]
            placement.validate_run(directory, m.get('cpu_placement', 'unbound'), command, text)
            require(command[command.index('--library') + 1] == m['library'] and
                    command[command.index('--profile') + 1] == str((directory / 'profile.tsv').resolve()) and
                    command[command.index('-np') + 1] == '4' and command[command.index('--') + 1:] ==
                    [m['client'], op, layout, '1024', str(maximum), '20', '10'], label + ': candidate command changed')
            require('UCCL reference-protocol checks passed' in text and 'max-rank CUDA events including enqueue gaps' in text,
                    label + ': missing correctness/timing declaration')
            for line in text.splitlines():
                if not line.startswith('sample,'): continue
                _, operation, actual_layout, size, index, latency, error = line.split(',')
                size, index, latency, error = int(size), int(index), float(latency), float(error)
                require(operation == op and actual_layout == layout and index not in rows[size] and
                        math.isfinite(latency) and latency > 0 and math.isfinite(error) and 0 <= error <= 1e-5,
                        label + ': invalid sample')
                rows[size][index] = latency
            require(set(rows) == sizes, label + ': missing candidate sizes')
            for size, values in rows.items():
                require(set(values) == set(range(10)), label + ': missing sample indices')
                candidate[size, layout].append(statistics.mean(values.values()))
            loaded = re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0', text)
            fallback = re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)', text)
            routes = collections.Counter(tuple(map(int, x)) for x in re.findall(
                r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)', text))
            require(sorted(map(int, loaded)) == list(range(4)) and sorted(tuple(map(int, x)) for x in fallback) == [(r, 0) for r in range(4)], label + ': native/fallback')
            require(routes == collections.Counter((r, number, size, 30) for r in range(4) for size in sizes), label + ': routes')
    output = []
    for size in sorted(sizes):
        reference = {variant: min(refs[size, variant]) for variant in variants(op, size) - {'nccl'}}
        best = min(reference, key=lambda variant: (reference[variant], variant))
        for layout in ('out', 'in'):
            require(len(candidate[size, layout]) == 3, 'missing repeats')
            time = statistics.median(candidate[size, layout])
            output.append(dict(operation=op, bytes=size, layout=layout, reference_variant=best,
                               reference_us=reference[best], candidate_us=time, ratio=time/reference[best]))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--audit-only', action='store_true')
    mode.add_argument('--summarize-only', action='store_true', help='recompute completed raw runs without launching jobs')
    p.add_argument('--operation', choices=('ag', 'rs'))
    p.add_argument('--reference', type=Path)
    p.add_argument('--library', type=Path)
    p.add_argument('--client', type=Path)
    p.add_argument('--real-nccl', type=Path)
    p.add_argument('--mpirun')
    p.add_argument('--cpu-placement', choices=('unbound', 'numa', 'core', 'core2'), default='unbound')
    p.add_argument('--max-bytes', type=int, default=1 << 30)
    args = p.parse_args()
    if not (args.audit_only or args.summarize_only):
        require(all((args.operation, args.reference, args.library, args.client, args.real_nccl, args.mpirun)), 'missing launch arguments')
        require(1024 <= args.max_bytes <= 1 << 30 and not args.max_bytes & (args.max_bytes - 1), 'invalid maximum')
        args.out.mkdir(parents=True, exist_ok=False)
        os.environ['UCCL_LITE_MSCCLPP_CONTROL_ROOT'] = '127.0.0.1'
        library, client, real = args.library.resolve(), args.client.resolve(), args.real_nccl.resolve()
        reference = args.reference.resolve() / ('build/uccl_lite_peer_ag_rs_packet' if args.operation == 'ag' else 'build/uccl_lite_peer_ag_rs')
        record = placement.make_record(args.cpu_placement, real, library, args.out)
        (args.out / 'identity.json').write_text(json.dumps(record, indent=2) + '\n')
        candidate_profile = (args.out / 'profile.tsv').resolve()
        op_number = 1 if args.operation == 'ag' else 2
        profile.write_candidate(candidate_profile, [record], [(op_number, (1 << n)//16) for n in range(10, args.max_bytes.bit_length())])
        m = dict(operation=args.operation, max_bytes=args.max_bytes, warmups=20, iterations=10,
                 original_repeats=3, candidate_repeats=3, reference_brackets=2,
                 scope='FP32 four L4s; max-GPU CUDA events; original 1 process/4 GPUs, candidate 4 processes',
                 cpu_placement=args.cpu_placement,
                 runs=[])
        for key, binary in [('library', library), ('client', client), ('reference', reference)]:
            m[key] = str(binary); m[key + '_sha256'] = profile.common.sha256(binary)
            m[key + '_ldd'] = subprocess.check_output(['ldd', str(binary)], text=True)
        launcher = [args.mpirun, '-np', '4', '--bind-to', 'none', '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp',
                    sys.executable, '-B', str(placement.launcher_path(args.cpu_placement)), 'launch',
                    '--library', str(library), '--real-nccl', str(real), '--profile', str(candidate_profile), '--', str(client)]
        order = ['reference_before', '0_out', '0_in', '1_in', '1_out', '2_out', '2_in', 'reference_after']
        for label in order:
            command = ([str(reference)] + reference_args(args.operation, args.max_bytes) if label.startswith('reference')
                       else launcher + [args.operation, label.split('_')[1], '1024', str(args.max_bytes), '20', '10'])
            print('running', label, flush=True)
            m['runs'].append(dict(label=label, command=command))
            (args.out / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
            with (args.out / (label + '.log')).open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
            m['runs'][-1]['exit_code'] = result.returncode
            (args.out / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
            require(result.returncode == 0, label + ': process failed')
    rows = audit(args.out)
    failures = [r for r in rows if r['ratio'] > 1]
    if args.audit_only:
        recorded = list(csv.DictReader((args.out / 'comparison.csv').open()))
        require(len(recorded) == len(rows) and all(all(str(row[k]) == recorded[i][k] for k in row) for i, row in enumerate(rows)), 'stored comparison mismatch')
    else:
        with (args.out / 'comparison.csv').open('w') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (args.out / 'result.json').write_text(json.dumps(dict(rows=len(rows), slower_rows=len(failures), failures=failures, full_goal_complete=False), indent=2) + '\n')
    print(f'{len(failures)}/{len(rows)} rows slower than original UCCL')
    return 2 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
