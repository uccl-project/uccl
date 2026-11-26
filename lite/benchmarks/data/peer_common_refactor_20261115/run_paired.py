#!/usr/bin/env python3
"""Checked, ascending ABBA comparison; one evaluator holds the host GPU lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--before', required=True, type=Path)
p.add_argument('--after', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
p.add_argument('--lock', required=True)
p.add_argument('--gpus', type=int, default=2)
p.add_argument('--end', type=int, default=1073741824)
p.add_argument('--warmup', type=int, default=100)
p.add_argument('--repeats', type=int, default=5)
p.add_argument('--iters', type=int, default=20)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('NCCL_', 'UCCL_LITE_'))}
env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(a.gpus)))

def active():
    lines = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
        text=True).splitlines()
    return {int(x) for x in lines if x.strip().isdigit()
            and Path('/proc', x.strip()).exists()}

records = []
with open(a.lock, 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if active():
        raise RuntimeError('GPU host busy')
    for name, cmd in [
        ('gpu', ['nvidia-smi', '-q']),
        ('topology', ['nvidia-smi', 'topo', '-m']),
        ('cpu', ['lscpu']),
    ]:
        (a.out / (name + '.txt')).write_text(
            subprocess.check_output(cmd, text=True))
    binaries = {key: str(path.resolve()) for key, path in
                [('before', a.before), ('after', a.after)]}
    hashes = {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for key, path in binaries.items()}
    (a.out / 'config.json').write_text(json.dumps({
        'args': {k: str(v) for k, v in vars(a).items()},
        'binaries': binaries, 'binary_sha256': hashes,
        'baseline_revision': 'b968a8011fb6979398a12ee2b5273bcf6693b94f',
        'order': ['before', 'after', 'after', 'before'],
        'scope': 'BF16 out-of-place; eager CUDA events, max over ranks; '
                 'every warmup/timed invocation independently checked; '
                 'fixed peer backend, no best-per-size selection',
    }, indent=2) + '\n')
    common = ['--gpus', str(a.gpus), '--source-identity', 'peer-common-paired']
    jobs = [('selftest', 'after', ['--selftest', '--end', '4096'])]
    for i, label in enumerate(['before', 'after', 'after', 'before']):
        name = f'{i}-{label}'
        jobs.append((name, label, [
            '--backend', 'peer', '--begin', '1024', '--end', str(a.end),
            '--warmup', str(a.warmup), '--repeats', str(a.repeats),
            '--iters', str(a.iters), '--seed', '719',
            '--samples', str((a.out / (name + '.csv')).resolve())]))
    for name, label, args in jobs:
        if active():
            raise RuntimeError('External GPU work detected')
        cmd = [binaries[label]] + common + args
        begin = time.monotonic()
        print('START', name, flush=True)
        with (a.out / (name + '.log')).open('w') as out:
            child = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                     env=env)
            try:
                while child.poll() is None:
                    if active() - {child.pid}:
                        raise RuntimeError('External GPU work; exclude run')
                    if time.monotonic() - begin > 900:
                        raise RuntimeError('Benchmark exceeded 900s')
                    time.sleep(1)
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        records.append({'name': name, 'argv': cmd,
                        'returncode': child.returncode,
                        'seconds': time.monotonic() - begin})
        (a.out / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
        print('DONE', name, child.returncode, flush=True)
        if child.returncode:
            raise RuntimeError('Failed: ' + name)
    (a.out / 'COMPLETE').write_text('PASS\n')
