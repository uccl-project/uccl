"""Four-L4 strict ABI candidate smoke; timings are not performance admission."""
import argparse
import collections
import math
import re

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
import mscclpp_profile as profile
import local_cpu_profile as placement


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--real-nccl', type=Path, required=True)
    p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--mpirun', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cpu-placement', choices=('unbound', 'numa', 'core', 'core2'), default='unbound')
    p.add_argument('--max-bytes', type=int, default=64 << 20)
    p.add_argument('--abi', type=Path, help='run shared strict ABI stress instead of nccl-tests')
    p.add_argument('--unalign', type=int, default=0)
    args = p.parse_args()
    if args.max_bytes < 1024 or args.max_bytes > 1 << 30 or args.max_bytes & (args.max_bytes - 1):
        p.error('--max-bytes must be a power of two from 1 KiB to 1 GiB')
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['UCCL_LITE_MSCCLPP_CONTROL_ROOT'] = '127.0.0.1'
    library, real = args.library.resolve(), args.real_nccl.resolve()
    record = placement.make_record(args.cpu_placement, real, library, args.out)
    (args.out / 'identity.json').write_text(json.dumps(record, indent=2) + '\n')
    shapes = [(op, (1 << power) // (4 if op == 3 else 16))
              for op in (1, 2, 3) for power in range(10, args.max_bytes.bit_length())]
    candidate = (args.out / 'profile.tsv').resolve()
    profile.write_candidate(candidate, [record], shapes)
    if args.abi:
        command = [args.mpirun, '-np', '4', '--bind-to', 'none',
                   '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp',
                   sys.executable, '-B', str(placement.launcher_path(args.cpu_placement)),
                   'launch', '--library', str(library), '--real-nccl', str(real),
                   '--profile', str(candidate), '--', str(args.abi.resolve())]
        (args.out / 'commands.json').write_text(json.dumps(dict(
            command=command, test_sha256=profile.common.sha256(args.abi)), indent=2) + '\n')
        log = args.out / 'abi.log'
        with log.open('w') as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
        text = log.read_text()
        placement.validate_run(args.out, args.cpu_placement, command, text)
        if 'Strict ABI full-buffer checks passed:' not in text:
            raise RuntimeError('missing ABI correctness result')
        for rank in range(4):
            for marker in (f'rank={rank} native_nccl_loaded=0', f'rank={rank} fallback=0'):
                if text.count(marker) != 2:
                    raise RuntimeError(f'missing two-lifetime marker {marker}')
        (args.out / 'result.json').write_text(json.dumps(dict(
            correctness='pass', performance='unqualified', scope='shared strict ABI'), indent=2) + '\n')
        return
    commands = []
    for op, name in ((1, 'all_gather'), (2, 'reduce_scatter'), (3, 'all_reduce')):
        test = (args.tests / (name + '_perf')).resolve()
        command = [args.mpirun, '-np', '4', '--bind-to', 'none',
                   '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp',
                   sys.executable, '-B', str(placement.launcher_path(args.cpu_placement)),
                   'launch', '--library', str(library), '--real-nccl', str(real),
                   '--profile', str(candidate), '--', str(test),
                   '-b', '1024', '-e', str(args.max_bytes), '-f', '2', '-g', '1',
                   '-d', 'float', '-o', 'sum', '-w', '5', '-n', '10', '-c', '1',
                   '-a', '3', '-G', '0', '-R', '0', '-m', '1', '-u', str(args.unalign)]
        commands.append(dict(command=command, test_sha256=profile.common.sha256(test)))
        (args.out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        log = args.out / (name + '.log')
        with log.open('w') as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
        text = log.read_text()
        placement.validate_run(args.out, args.cpu_placement, command, text)
        if '# Out of bounds values : 0 OK' not in text:
            raise RuntimeError(f'{name}: missing correctness result')
        expected_sizes = {1 << power for power in range(10, args.max_bytes.bit_length())}
        ranks = re.findall(r'#\s+Rank\s+(\d+)\s+Group', text)
        if sorted(map(int, ranks)) != list(range(4)):
            raise RuntimeError(f'{name}: incorrect rank coverage')
        loaded = re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0', text)
        fallback = re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)', text)
        if sorted(map(int, loaded)) != list(range(4)) or sorted(
                tuple(map(int, item)) for item in fallback) != [(r, 0) for r in range(4)]:
            raise RuntimeError(f'{name}: native library or fallback assertion failed')
        routes = collections.Counter(tuple(map(int, item)) for item in re.findall(
            r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)', text))
        # Same route audit as the multi-node driver: five warmups, then ten
        # timed calls plus synchronization and correctness for each layout.
        expected = collections.Counter((r, op, size, 29)
                                       for r in range(4) for size in expected_sizes)
        if routes != expected:
            raise RuntimeError(f'{name}: route counts differ from the frozen nccl-tests protocol')
        rows = {}
        for line in text.splitlines():
            fields = line.split()
            if len(fields) != 13 or not fields[0].isdigit():
                continue
            size = int(fields[0])
            times = [float(fields[5]), float(fields[9])]
            if (size in rows or fields[2:4] != ['float', 'none' if op == 1 else 'sum']
                    or fields[8] != '0' or fields[12] != '0'
                    or not all(math.isfinite(x) and x > 0 for x in times)):
                raise RuntimeError(f'{name}: invalid numerical/timing row')
            rows[size] = times
        if set(rows) != expected_sizes:
            raise RuntimeError(f'{name}: incomplete size coverage')
        (args.out / (name + '.json')).write_text(json.dumps(rows, indent=2) + '\n')
    (args.out / 'result.json').write_text(json.dumps(dict(
        correctness='pass', performance='unqualified', dtype='float32', ranks=4,
        warmups=5, iterations=10, timing='nccl-tests batch, maximum rank',
        variant_selection=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
