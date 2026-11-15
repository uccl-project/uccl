#!/usr/bin/env python3
"""Run the recorded argv manifest serially under a shared GPU evaluation lock."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

manifest, destination, lock_path = sys.argv[1:]
out = Path(destination).resolve()
out.mkdir(parents=True, exist_ok=False)
jobs = json.loads(Path(manifest).read_text())
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('NCCL_', 'UCCL_LITE_'))}

def active():
    return {int(x) for x in subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
        text=True).splitlines() if x.strip().isdigit()
        and Path('/proc', x.strip()).exists()}

records = []
with open(lock_path, 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if active():
        raise RuntimeError('GPU host busy')
    for name, args in [('gpu', ['nvidia-smi', '-q']),
                       ('topology', ['nvidia-smi', 'topo', '-m']),
                       ('cpu', ['lscpu'])]:
        (out / (name + '.txt')).write_text(
            subprocess.check_output(args, text=True))
    (out / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    for job in jobs:
        name, argv = job['name'], job['argv']
        if active():
            raise RuntimeError('External GPU work detected')
        begin = time.monotonic()
        print('START', name, flush=True)
        with (out / (name + '.log')).open('w') as log:
            child = subprocess.Popen(argv, env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    if active() - {child.pid}:
                        raise RuntimeError('External GPU work; exclude run')
                    if time.monotonic() - begin > job.get('timeout', 900):
                        raise RuntimeError('Test timed out')
                    time.sleep(1)
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        records.append(dict(name=name, argv=argv, returncode=child.returncode,
                            seconds=time.monotonic() - begin,
                            binary_sha256=hashlib.sha256(
                                Path(argv[0]).read_bytes()).hexdigest()))
        (out / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
        print('DONE', name, child.returncode, flush=True)
        if child.returncode:
            raise RuntimeError('Failed: ' + name)
    (out / 'COMPLETE').write_text('PASS\n')
