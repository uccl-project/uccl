"""Exercise candidate routes with unchanged MPI nccl-tests; no deployment profile."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from run import checked_output

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
from dropin_profile import identity, write_profile, launch_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--real-nccl', type=Path, required=True)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--mpirun', default='mpirun')
    args = parser.parse_args()
    if args.library:
        import dropin_profile
        dropin_profile.LIBRARY = args.library.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    real = args.real_nccl.resolve(strict=True)
    ident, details = identity(real)
    ranks = len(details['gpus'].splitlines())
    if ranks not in (4, 8):
        raise RuntimeError('requires four or eight local L4 GPUs')
    profile = args.out / 'candidate.tsv'
    write_profile(profile, ident, details, real,
                  [[op, n, 1, 1, 3] for op in (1, 2, 3) for n in (32, 131072)],
                  candidate=True)
    env = launch_environment(real, profile, calibrating=True)
    export = []
    for key in env:
        if key.startswith(('UCCL_LITE_', 'NCCL_', 'CUDA_')) or key in ('LD_PRELOAD', 'LD_LIBRARY_PATH'):
            export += ['-x', key]
    manifest = {'identity': ident, 'details': details, 'commands': []}
    for graph in (0, 3):
        for op, minimum in [('all_gather', 64), ('reduce_scatter', 64 * ranks), ('all_reduce', 64)]:
            binary = args.tests.resolve() / (op + '_perf')
            dependencies = subprocess.check_output(['ldd', str(binary)], env=env, text=True)
            if not re.search(r'libmpi\.so', dependencies):
                raise RuntimeError('requires MPI-enabled nccl-tests: ' + str(binary))
            maximum = 262144 if op == 'all_reduce' else 262144 * ranks
            command = [args.mpirun, '-np', str(ranks), '--bind-to', 'none', *export,
                       str(binary), '-b', str(minimum), '-e', str(maximum), '-f', '2',
                       '-g', '1', '-d', 'bfloat16', '-w', '2', '-n', '5', '-c', '1',
                       '-G', str(graph)]
            log = args.out / f'{op}_graph{graph}.log'
            with log.open('w') as stream:
                result = subprocess.run(command, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=240)
            result.check_returncode()
            text = log.read_text()
            checked_output(text, ranks)
            counters = re.findall(r'UCCL Lite drop-in: rank=(\d+) AG=(\d+) RS=(\d+) AR=(\d+)', text)
            slot = {'all_gather': 1, 'reduce_scatter': 2, 'all_reduce': 3}[op]
            observed = {int(row[0]) for row in counters if int(row[slot]) > 0}
            if observed != set(range(ranks)):
                raise RuntimeError(f'{log}: missing staged route counters on ranks {set(range(ranks)) - observed}')
            manifest['commands'].append(command)
            (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(op, graph, 'PASS: validated with staged route counters on all ranks', flush=True)


if __name__ == '__main__':
    main()
