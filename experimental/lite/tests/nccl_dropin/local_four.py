"""Guarded four-rank packet stress: tails, reuse, streams, and mixed layouts."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
COUNTS = (1, 3, 7, 31, 32, 33, 1025, 65539, 131072, 131073)


def worker(out):
    import torch
    import torch.distributed as dist
    from bench_dropin import Nccl

    rank = int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo')
    if dist.get_world_size() != 4:
        raise RuntimeError('requires four local ranks')
    nccl = Nccl(rank)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    pending = []
    # Every rank uses a different alignment; odd ranks use legal in-place
    # layouts. Keep all buffers live while both streams have outstanding work.
    for step in range(3):
        for count in COUNTS:
            for op in (1, 2, 3):
                stream = streams[len(pending) % 2]
                ni = count * (4 if op == 2 else 1)
                no = count * (4 if op == 1 else 1)
                guard = 128 + (1, 2, 4, 8)[rank]
                with torch.cuda.stream(stream):
                    a = torch.full((max(ni, no) + guard * 2,), -123., device='cuda', dtype=torch.bfloat16)
                    b = a if rank % 2 else torch.full_like(a, -123.)
                    source_offset = rank * count if rank % 2 and op == 1 else 0
                    target_offset = rank * count if rank % 2 and op == 2 else 0
                    source = a[guard + source_offset:guard + source_offset + ni]
                    target = b[guard + target_offset:guard + target_offset + no]
                    values = ((torch.arange(ni, device='cuda') % 31).float() - 15) / 32
                    source.copy_(values + (rank + 1) / 16 + step / 32)
                    if rank == 3 and step == 1 and count == 31:
                        torch.cuda._sleep(100000)
                    nccl.call(op, source, target, count, stream)
                pending.append((a, b, target, op, count, guard, step))
    torch.cuda.synchronize()
    for a, b, target, op, count, guard, step in pending:
        assert torch.all(a[:guard] == -123) and torch.all(a[-guard:] == -123)
        assert torch.all(b[:guard] == -123) and torch.all(b[-guard:] == -123)
        if op == 1:
            base = ((torch.arange(count, device='cuda') % 31).float() - 15) / 32
            expected = torch.cat([(base + (r + 1) / 16 + step / 32).bfloat16() for r in range(4)])
        else:
            offset = rank * count if op == 2 else 0
            base = (((torch.arange(count, device='cuda') + offset) % 31).float() - 15) / 32
            expected = (base * 4 + 10 / 16 + step / 8).bfloat16()
        assert torch.equal(target, expected), (rank, op, count, step)
    nccl.check(nccl.lib.ncclCommDestroy(nccl.comm))
    dist.barrier()
    if rank == 0:
        out.write_text(json.dumps({'counts': COUNTS, 'operations': [1, 2, 3],
                                  'generations': 3, 'guards': 'pass',
                                  'mixed_streams_and_layouts': 'pass'}) + '\n')
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--real-nccl', type=Path)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--sanitizer', type=Path,
                        help='compute-sanitizer executable for per-rank memcheck')
    args = parser.parse_args()
    if args.library:
        import dropin_profile
        dropin_profile.LIBRARY = args.library.resolve(strict=True)
    if args.worker:
        return worker(args.out)
    from dropin_profile import identity, write_profile, launch_environment
    if not args.real_nccl:
        parser.error('--real-nccl is required')
    args.out.mkdir(parents=True, exist_ok=False)
    ident, details = identity(args.real_nccl)
    if len(details['gpus'].splitlines()) != 4:
        parser.error('requires four local L4 GPUs')
    profile = args.out / 'candidate.tsv'
    write_profile(profile, ident, details, args.real_nccl,
                  [[op, n, 1, 1, 3] for op in (1, 2, 3) for n in COUNTS], candidate=True)
    env = launch_environment(args.real_nccl, profile, calibrating=True)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node=4']
    if args.sanitizer:
        command += ['--no-python', str(args.sanitizer.resolve()), '--tool',
                    # NCCL probes unsupported kernel images during init and
                    # handles those API errors. Keep memory error reporting.
                    'memcheck', '--report-api-errors', 'no',
                    '--error-exitcode', '99', sys.executable]
    command += [str(Path(__file__).resolve()), '--worker',
               '--out', str(args.out / 'result.json')]
    (args.out / 'manifest.json').write_text(json.dumps({'command': command, 'identity': ident, 'details': details}, indent=2) + '\n')
    with (args.out / 'check.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=240)
    text = (args.out / 'check.log').read_text()
    if 'four-local-rank GPU path' not in text:
        raise RuntimeError('GPU path was not prepared')
    counters = re.findall(r'rank=(\d+) AG=(\d+) RS=(\d+) AR=(\d+)', text)
    expected = len(COUNTS) * 3
    if {int(row[0]) for row in counters if all(int(n) == expected for n in row[1:])} != set(range(4)):
        raise RuntimeError('missing selected-route counters on one or more ranks')
    if args.sanitizer and text.count('ERROR SUMMARY: 0 errors') != 4:
        raise RuntimeError('missing clean per-rank memcheck summaries')
    print('PASS: guarded four-rank packet stress')


if __name__ == '__main__':
    main()
