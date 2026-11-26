#!/usr/bin/env python3
"""Test NCCL memcpy tuning with complete ascending sweeps in every process."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['calibrate', 'evaluate', 'trace'])
args = parser.parse_args()
root = Path('/home/xinze/dev/pro6000-integrated-b0ac7b4a-20261115-022114-retry1/source/experimental/lite')
out = root / 'build/nccl_ascending_20261115'
out.mkdir(exist_ok=True)
binary = root / 'build/nccl_tuned_20261115/nccl_reference'
lock = open('/home/xinze/dev/uccl-eat-pro6000/eval.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
environment = {k: v for k, v in os.environ.items() if not k.startswith(('NCCL_', 'UCCL_'))}
environment.update(LD_LIBRARY_PATH='/opt/pytorch/cuda/lib', CUDA_VISIBLE_DEVICES='0,1', NCCL_DEBUG='WARN')
configs = {'default': {}, 'memcpy': {'NCCL_P2P_USE_CUDA_MEMCPY': '1'}}
for size in [8, 16, 32]:
    configs[f'memcpy_buffer{size}m_channels8'] = {
        'NCCL_P2P_USE_CUDA_MEMCPY': '1', 'NCCL_BUFFSIZE': str(size << 20),
        'NCCL_MIN_NCHANNELS': '8', 'NCCL_MAX_NCHANNELS': '8'}
ops = ['allreduce', 'allgather', 'reducescatter']
jobs = []
if args.stage == 'calibrate':
    (out / 'candidates.json').write_text(json.dumps(configs, indent=2) + '\n')
    for repeat in range(2):
        pairs = [(op, name) for op in ops for name in configs]
        random.Random(202611190 + repeat).shuffle(pairs)
        jobs += [dict(name=f'cal_{op}_{config}_{repeat}', op=op, config=config, repeat=repeat)
                 for op, config in pairs]
else:
    selection = json.loads((out / 'selection.json').read_text())
    for repeat in range(2 if args.stage == 'evaluate' else 1):
        for op in (ops if repeat == 0 else list(reversed(ops))):
            chosen = selection['bulk_configurations'][op]
            order = list(dict.fromkeys(['default', chosen]))
            if repeat == 1:
                order.reverse()
            jobs += [dict(name=f'{args.stage}_{op}_{config}_{repeat}', op=op, config=config, repeat=repeat)
                     for config in order]
paths = [binary, root / 'build/nccl_tuned_20261115/nccl_reference.cu',
         Path('/home/xinze/dev/uccl-lite-spin-pro6000-20261113/build/experiment/deps/nccl/lib/libnccl.so.2')]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(out / f'{args.stage}_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
runs = []
for index, job in enumerate(jobs, 1):
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                      '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU occupied: ' + active)
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == hashes[str(binary)]
    name = job['name']
    env = {**environment, **configs[job['config']]}
    if args.stage == 'trace':
        env.update(NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,TUNING,ENV,P2P,SHM',
                   NCCL_DEBUG_FILE=str(out / (name + '_debug.log')))
    warmup, repeats, iterations = {'calibrate': (3, 3, 3), 'evaluate': (100, 5, 20), 'trace': (1, 1, 1)}[args.stage]
    seed = (202611190 if args.stage == 'calibrate' else 202611200) + job['repeat']
    cmd = [str(binary), '--backend', 'nccl', '--gpus', '2', '--op', job['op'],
           '--begin', '1024', '--end', '1073741824', '--warmup', str(warmup),
           '--repeats', str(repeats), '--iters', str(iterations), '--seed', str(seed),
           '--samples', str(out / (name + '_samples.csv'))]
    # Every process starts with 1 KiB and keeps the communicator through 1 GiB.
    assert '--reverse' not in cmd
    started = datetime.now(timezone.utc).isoformat()
    print(f'{index}/{len(jobs)} {name}: starting', flush=True)
    with (out / (name + '.csv')).open('x') as stdout, (out / (name + '.log')).open('x') as stderr:
        try:
            result = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr,
                                    timeout=60 if args.stage != 'evaluate' else 300)
            rc = result.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    output = (out / (name + '.csv')).read_text() + (out / (name + '.log')).read_text()
    passed = rc == 0 and 'correctness: PASS' in output
    runs.append(dict(**job, command=shlex.join(cmd), size_order='ascending',
                     environment={k:v for k,v in env.items() if k.startswith(('NCCL_', 'CUDA_', 'LD_LIBRARY_PATH'))},
                     started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                     returncode=rc, correctness_pass=passed))
    (out / f'{args.stage}_runs.json').write_text(json.dumps(runs, indent=2) + '\n')
    print(f'{name}: {"PASS" if passed else "FAIL (retained)"}', flush=True)
    if not passed and args.stage != 'calibrate':
        raise RuntimeError(name)
