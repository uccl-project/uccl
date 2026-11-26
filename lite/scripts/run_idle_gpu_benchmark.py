#!/usr/bin/env python3
"""Run a single-process GPU benchmark only in an observed idle window.
Abort only our child if another GPU process appears. No outside process is touched.
This monitor supplements, and does not replace, a shared resource reservation.
"""
import argparse
import datetime
import json
import os
import signal
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--log', required=True, type=Path)
parser.add_argument('--wait-seconds', type=float, default=120)
parser.add_argument('command', nargs=argparse.REMAINDER)
a = parser.parse_args()
command = a.command[1:] if a.command[:1] == ['--'] else a.command
if not command: parser.error('missing command')
if a.log.exists(): parser.error('refusing to overwrite an existing run')
monitor = a.log.with_suffix(a.log.suffix + '.monitor.jsonl')

def processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                             '--format=csv,noheader,nounits'], text=True,
                            capture_output=True, check=True)
    return sorted({int(x.strip()) for x in result.stdout.splitlines() if x.strip()})
def record(event, **fields):
    with monitor.open('a') as f:
        f.write(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               event=event, **fields))+'\n')

deadline = time.monotonic() + a.wait_seconds
while True:
    pids = processes()
    record('availability', pids=pids)
    if not pids:
        time.sleep(.5)
        if not processes(): break
    if time.monotonic() >= deadline:
        record('not_started_busy'); raise SystemExit(75)
    time.sleep(1)

record('start', command=command)
with a.log.open('w') as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    contaminated = False
    try:
        while True:
            status = child.poll()
            pids = processes()
            foreign = [pid for pid in pids if pid != child.pid]
            record('sample', child=child.pid, pids=pids, foreign=foreign)
            if foreign:
                contaminated = True
                if status is None:
                    os.killpg(child.pid, signal.SIGTERM)
                break
            if status is not None: break
            time.sleep(.5)
        try: status = child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); status = child.wait()
    except BaseException:
        if child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=10)
        raise
record('finish', status=status, contaminated=contaminated)
raise SystemExit(76 if contaminated else status)
