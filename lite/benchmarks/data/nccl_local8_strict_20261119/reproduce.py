#!/usr/bin/env python3
"""Run unchanged nccl-tests using the qualified libnccl.so.2 loader alias."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tests/nccl_mscclpp'))
import local8_process
from run import checked_strict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-bytes', type=int, default=1 << 30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if not 1024 <= args.max_bytes <= 1 << 30 or args.max_bytes & (args.max_bytes - 1):
        p.error('max-bytes must be a power of two in 1 KiB–1 GiB')
    if args.warmup < 0 or args.iterations < 1:
        p.error('invalid iteration counts')
    lock = open('/tmp/uccl-lite-integration-wxz-l4.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if local8_process.gpu_users():
        raise RuntimeError('GPU occupied')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    out = args.out or ROOT / 'build' / ('local8_reproduce_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('NCCL_', 'CUDA_', 'UCCL_LITE_', 'MSCCLPP_'))
           and k != 'LD_PRELOAD'}
    env.update(json.loads((HERE / 'launch_environment.json').read_text()))
    prefix = ['mpirun', '--timeout', '900', '-np', '8', '--bind-to', 'none',
              '--map-by', 'slot', '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp',
              sys.executable, '-B', str(ROOT / 'integration/nccl_lite/mscclpp_profile.py'),
              'launch', '--local8', '--library', str(ROOT / 'build/nccl_local8/libuccl_lite_nccl_local8.so'),
              '--real-nccl', str(ROOT / 'build/nccl-2.28.9/lib/libnccl.so.2'),
              '--profile', str(HERE / 'profile.tsv'), '--', 'env', '-u', 'LD_PRELOAD']
    results = {}
    for name, operation in [('all_gather', 1), ('reduce_scatter', 2), ('all_reduce', 3)]:
        cmd = prefix + [str(ROOT / 'build/nccl-tests-local8-src/build' / (name + '_perf')),
                        '-b', '1K', '-e', str(args.max_bytes), '-f', '2', '-g', '1',
                        '-d', 'bfloat16', '-o', 'sum', '-w', str(args.warmup),
                        '-n', str(args.iterations), '-c', '1', '-a', '3', '-m', '1', '-G', '0', '-R', '0']
        log_path = out / (name + '.log')
        with log_path.open('x') as log:
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            status = local8_process.wait(child, 930, log_path=log_path,
                                          affinity_path=out / (name + '.affinity.json'))
        if status:
            raise RuntimeError(f'{name} failed; see {log_path}')
        result = checked_strict(log_path.read_text(), operation)
        expected = {1 << n for n in range(10, args.max_bytes.bit_length())}
        if set(result['rows_us']) != expected or any(int(r[3]) != 2 * (args.warmup + args.iterations + 1)
                                                   for r in result['strict_routes']):
            raise RuntimeError('incomplete size/route coverage')
        results[name] = dict(argv=cmd, returncode=status, audit=result)
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print('PASS', name, 'through libnccl.so.2; native NCCL absent', flush=True)
    print('Logs:', out, flush=True)


if __name__ == '__main__':
    main()
