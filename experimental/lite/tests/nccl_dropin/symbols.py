"""The ABI proxy must not interpose its private CUDA/STL/core symbols."""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('libraries', nargs='+', type=Path)
args = p.parse_args()
required = {'ncclAllGather', 'ncclAllReduce', 'ncclReduceScatter', 'ncclGroupStart',
            'ncclGroupEnd', 'ncclCommInitRank', 'ncclCommInitRankConfig', 'ncclCommDestroy'}
for library in args.libraries:
    output = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True)
    symbols = {line.split()[-1].split('@')[0] for line in output.splitlines() if line.split()}
    unexpected = {name for name in symbols if not name.startswith('nccl')}
    if unexpected or not required.issubset(symbols):
        raise SystemExit(f'{library}: unexpected={unexpected}, missing={required - symbols}')
    print(f'PASS: {library}: {len(symbols)} NCCL ABI exports; private symbols hidden')
