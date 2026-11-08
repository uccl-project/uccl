"""Bounded child processes; terminate only the process group owned by this run."""

import os
import json
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


def run(command, log, timeout, env=None, watch_stdin=False, test_id=None):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with open(log, "wb") as output:
        child = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        reason = None
        try:
            while child.poll() is None:
                if time.monotonic() - start >= timeout:
                    reason = "timeout"
                    break
                if watch_stdin and select.select([sys.stdin], [], [], 0)[0]:
                    # Only cancellation/EOF is sent while a child is running.
                    message = sys.stdin.readline()
                    if not message or json.loads(message).get("cancel") == test_id:
                        reason = "controller disconnected or cancelled"
                        break
                time.sleep(0.1)
        finally:
            # Also reap descendants left behind by an otherwise successful test.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
    return {
        "exit_code": (
            124 if reason == "timeout" else (125 if reason else child.returncode)
        ),
        "seconds": round(time.monotonic() - start, 3),
        "error": reason,
        "log": str(log),
    }
