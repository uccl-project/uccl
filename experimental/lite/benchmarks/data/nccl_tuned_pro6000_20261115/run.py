#!/usr/bin/env python3
"""Calibrate then independently evaluate NCCL AR/AG/RS with the figure harness."""
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
parser.add_argument('stage', choices=['calibrate', 'supplement', 'evaluate', 'trace'])
args = parser.parse_args()
root = Path('/home/xinze/dev/pro6000-integrated-b0ac7b4a-20261115-022114-retry1/source/experimental/lite')
out = root / 'build/nccl_tuned_20261115'
binary = out / 'nccl_reference'
lock = open('/home/xinze/dev/uccl-eat-pro6000/eval.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
env = {k: v for k, v in os.environ.items() if not k.startswith(('NCCL_', 'UCCL_'))}
env.update(LD_LIBRARY_PATH='/opt/pytorch/cuda/lib', CUDA_VISIBLE_DEVICES='0,1', NCCL_DEBUG='WARN')
configs = {'default': {}}
for size in [1, 2, 8, 16]:
    configs[f'buffer{size}m'] = {'NCCL_BUFFSIZE': str(size << 20)}
for threads in [128, 256]:
    configs[f'threads{threads}'] = {'NCCL_NTHREADS': str(threads)}
for channels in [8, 16]:
    configs[f'channels{channels}'] = {'NCCL_MIN_NCHANNELS': str(channels), 'NCCL_MAX_NCHANNELS': str(channels)}
configs['buffer1m_channels8'] = {**configs['buffer1m'], **configs['channels8']}
configs['buffer8m_channels16'] = {**configs['buffer8m'], **configs['channels16']}
configs['peer_read'] = {'NCCL_P2P_READ_ENABLE': '1'}
configs['memcpy'] = {'NCCL_P2P_USE_CUDA_MEMCPY': '1'}
for buffer in [8, 16, 32]:
    configs[f'memcpy_buffer{buffer}m_channels8'] = {**configs['memcpy'], **configs['channels8'], 'NCCL_BUFFSIZE': str(buffer << 20)}
configs['registered'] = {'UCCL_NCCL_REGISTER': '1'}
configs['registered_buffer1m'] = {**configs['registered'], **configs['buffer1m']}
configs['registered_buffer1m_threads128'] = {**configs['registered_buffer1m'], **configs['threads128']}
configs['registered_peer_read'] = {**configs['registered'], **configs['peer_read']}
configs['registered_channels8'] = {**configs['registered'], **configs['channels8']}
configs['zero_cta'] = {**configs['registered'], 'NCCL_CTA_POLICY': '2'}
configs['simple'] = {'NCCL_PROTO': 'Simple'}

def idle():
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                      '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU occupied: ' + active)

idle()
jobs = []
if args.stage == 'supplement':
    configs = json.loads((out / 'candidates.json').read_text())
    previous = json.loads((out / 'calibrate_runs.json').read_text())
    failed = [r['config'] for r in previous if r['repeat'] == 1 and not r['correctness_pass']]
    # Reverse all-op order begins with RS. Test AR/AG independently so an RS
    # startup failure cannot hide otherwise valid operation-specific evidence.
    jobs = [dict(name=f'supp_{name}_{op}_1', config=name, op=op, begin=1024, end=1 << 30, repeat=1)
            for name in failed for op in ['allgather', 'allreduce']]
elif args.stage == 'calibrate':
    (out / 'candidates.json').write_text(json.dumps(configs, indent=2) + '\n')
    for repeat in range(2):
        order = [name for name in configs if name != 'default']
        random.Random(202611150 + repeat).shuffle(order)
        jobs.append(dict(name=f'cal_default_before_{repeat}', config='default', op='all', begin=1024, end=1 << 30, repeat=repeat))
        jobs += [dict(name=f'cal_{name}_{repeat}', config=name, op='all', begin=1024, end=1 << 30, repeat=repeat) for name in order]
        jobs.append(dict(name=f'cal_default_after_{repeat}', config='default', op='all', begin=1024, end=1 << 30, repeat=repeat))
else:
    configs = json.loads((out / 'candidates.json').read_text())
    schedule = json.loads((out / 'schedule.json').read_text())
    segments = []
    for op in ['allreduce', 'allgather', 'reducescatter']:
        selected = sorted([r for r in schedule if r['operation'] == op], key=lambda r: r['bytes'])
        for row in selected:
            if segments and segments[-1]['op'] == op and segments[-1]['config'] == row['configuration'] and segments[-1]['end'] * 2 == row['bytes']:
                segments[-1]['end'] = row['bytes']
            else:
                segments.append(dict(op=op, config=row['configuration'], begin=row['bytes'], end=row['bytes']))
    (out / 'segments.json').write_text(json.dumps(segments, indent=2) + '\n')
    if args.stage == 'evaluate':
        for repeat in range(2):
            baseline = dict(name=f'eval_default_{repeat}', config='default', op='all', begin=1024, end=1 << 30, repeat=repeat)
            if repeat == 0:
                jobs.append(baseline)
            ordered = segments if repeat == 0 else list(reversed(segments))
            jobs += [dict(segment, name=f'eval_tuned_{repeat}_{i:02d}', repeat=repeat) for i, segment in enumerate(ordered)]
            if repeat == 1:
                jobs.append(baseline)
    else:
        jobs = [dict(segment, name=f'trace_{i:02d}', repeat=0) for i, segment in enumerate(segments)]

paths = [binary, out / 'nccl_reference.cu',
         root / 'build/uccl_lite_joint_nccl_reference',
         Path('/home/xinze/dev/uccl-lite-spin-pro6000-20261113/build/experiment/deps/nccl/lib/libnccl.so.2')]
hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
(out / (args.stage + '_hashes.json')).write_text(json.dumps(hashes, indent=2) + '\n')
runs = []
for index, job in enumerate(jobs, 1):
    idle()
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == hashes[str(binary)]
    name = job['name']
    settings = {**env, **configs[job['config']]}
    if args.stage == 'trace':
        settings.update(NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,TUNING,ENV,P2P,SHM',
                        NCCL_DEBUG_FILE=str(out / (name + '_debug.log')))
    warmup, repeats, iters = {'calibrate': (3, 3, 3), 'supplement': (3, 3, 3), 'evaluate': (100, 5, 20), 'trace': (1, 1, 1)}[args.stage]
    seed = (202611150 if args.stage in ['calibrate', 'supplement'] else 202611170) + job['repeat']
    cmd = [str(binary), '--backend', 'nccl', '--gpus', '2', '--op', job['op'],
           '--begin', str(job['begin']), '--end', str(job['end']), '--warmup', str(warmup),
           '--repeats', str(repeats), '--iters', str(iters), '--seed', str(seed),
           '--samples', str(out / (name + '_samples.csv'))]
    if job['repeat'] == 1:
        cmd.append('--reverse')
    started = datetime.now(timezone.utc).isoformat()
    print(f'{index}/{len(jobs)} {name} {job["config"]}: starting', flush=True)
    with (out / (name + '.csv')).open('x') as stdout, (out / (name + '.log')).open('x') as stderr:
        try:
            process = subprocess.run(cmd, env=settings, stdout=stdout, stderr=stderr,
                                     timeout=30 if args.stage == 'supplement' else 600)
            rc = process.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    output = (out / (name + '.csv')).read_text() + (out / (name + '.log')).read_text()
    passed = rc == 0 and 'correctness: PASS' in output
    runs.append(dict(**job, command=shlex.join(cmd),
                     environment={k:v for k,v in settings.items() if k.startswith(('NCCL_', 'UCCL_', 'CUDA_', 'LD_LIBRARY_PATH'))},
                     started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                     returncode=rc, correctness_pass=passed))
    (out / (args.stage + '_runs.json')).write_text(json.dumps(runs, indent=2) + '\n')
    print(f'{name}: {"PASS" if passed else "FAIL (retained)"}', flush=True)
    if not passed and args.stage not in ['calibrate', 'supplement']:
        raise RuntimeError(name)
idle()
