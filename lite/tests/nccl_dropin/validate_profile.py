"""Independent ABBA validation of an existing local deployment profile."""

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
from dropin_profile import HERE, identity, launch_environment, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real-nccl', type=Path, required=True)
    p.add_argument('--library', type=Path)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--iterations', type=int, default=50)
    args = p.parse_args()
    if args.library:
        import dropin_profile
        dropin_profile.LIBRARY = args.library.resolve(strict=True)
    if args.repeats < 3 or args.iterations < 1:
        p.error('requires at least three repeats and positive iterations')
    args.out.mkdir(parents=True, exist_ok=False)
    ident, details = identity(args.real_nccl)
    ranks = len(details['gpus'].splitlines())
    selected = {(int(f[0]), int(f[1])) for line in args.profile.read_text().splitlines()
                if len(f := line.split()) == 5}
    if not selected or not selected.issubset({tuple(c) for c in json.loads(args.cases.read_text())}):
        p.error('cases must cover every selected shape in a nonempty profile')
    shutil.copyfile(args.profile, args.out / 'profile.tsv')
    shutil.copyfile(args.cases, args.out / 'cases.json')
    manifest = {'identity': ident, 'details': details, 'trials': [],
                'profile_sha256': sha256(args.profile),
                'benchmark_sha256': sha256(HERE / 'bench_dropin.py')}
    trials = []
    # Share the calibrator's advisory lock; timing runs must not overlap it.
    lock = open('/tmp/uccl-lite-integration-wxz-l4.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    for i, backend in enumerate(('nccl', 'dropin', 'dropin', 'nccl')):
        env = launch_environment(args.real_nccl, args.profile if backend == 'dropin' else None)
        if backend == 'nccl':
            env['LD_PRELOAD'] = str(args.real_nccl.resolve())
            env['SGLANG_NCCL_SO_PATH'] = str(args.real_nccl.resolve())
        elif 'UCCL_LITE_DROPIN_PROFILE' not in env:
            raise RuntimeError('deployment profile was rejected by the launcher')
        target = args.out / f'{i}_{backend}.jsonl'
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc-per-node={ranks}', str(HERE / 'bench_dropin.py'),
                   '--cases', str(args.cases), '--out', str(target), '--warmup', '10',
                   '--iterations', str(args.iterations), '--repeats', str(args.repeats)]
        log = args.out / f'{i}_{backend}.log'
        print('Starting', backend, i, flush=True)
        with log.open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
        if backend == 'dropin':
            import re
            counters = re.findall(r'rank=(\d+) AG=(\d+) RS=(\d+) AR=(\d+)', log.read_text())
            cases = json.loads(args.cases.read_text())
            for rank in range(ranks):
                for op in (1, 2, 3):
                    minimum = sum(o == op and (o, n) in selected for o, n in cases)
                    minimum *= 8 * args.repeats * (10 + args.iterations)
                    actual = sum(int(row[op]) for row in counters if int(row[0]) == rank)
                    if actual < minimum:
                        raise RuntimeError('deployment did not execute all selected calls')
        manifest['trials'].append({'backend': backend, 'command': command})
        (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        trials.append({(r['op'], r['count'], r['layout'], r['alignment']): r
                       for line in target.read_text().splitlines() if (r := json.loads(line))})
    if any(set(t) != set(trials[0]) for t in trials):
        raise RuntimeError('different cases across independent jobs')
    summary = []
    for key in trials[0]:
        jobs = [statistics.median(t[key]['repeat_ms']) for t in trials]
        if not all(t[key]['correct'] for t in trials):
            raise RuntimeError('numerical comparison failed')
        ratio = max(jobs[1:3]) / min(jobs[0], jobs[3])
        summary.append({'op': key[0], 'count': key[1], 'layout': key[2],
                        'alignment': key[3], 'jobs_ms': jobs,
                        'worst_dropin_over_best_nccl': ratio,
                        'within_5_percent': ratio <= 1.05})
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    failures = sum(not r['within_5_percent'] for r in summary)
    print('Independent 5% regression failures:', failures, '/', len(summary))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
