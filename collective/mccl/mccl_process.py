"""Bounded local process-group shutdown for Linux/macOS test launchers."""

import os
import signal
import subprocess


def run_command(command, env, timeout):
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    def terminate(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Darwin may return EPERM for an already-exited launcher's group.
            # Never hide a genuine inability to signal a live launcher.
            if process.poll() is None:
                raise

    try:
        try:
            output, _ = process.communicate(timeout=timeout)
            return process.returncode, output, False
        except subprocess.TimeoutExpired:
            terminate(signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                terminate(signal.SIGKILL)
                try:
                    output, _ = process.communicate(timeout=2)
                except subprocess.TimeoutExpired as error:
                    # Escaped descendants can keep stdout open after launcher exit.
                    output = error.output or b""
                    if isinstance(output, bytes):
                        output = output.decode(errors="replace")
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            # Children may ignore TERM and close their pipes, so launcher EOF
            # alone is not proof that the rest of its group has stopped.
            terminate(signal.SIGKILL)
            return process.returncode, output, True
    finally:
        process.stdout.close()
