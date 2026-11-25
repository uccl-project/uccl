"""Run the existing large-notification fixture with its dynamically assigned port."""

import os
import re
import subprocess

env = {**os.environ, "UCCL_P2P_TRANSPORT": "rdma", "UCCL_P2P_DISABLE_IPC": "1"}
server = subprocess.Popen(
    ["p2p/notif_e2e", "server"], stdout=subprocess.PIPE, text=True, env=env
)
try:
    for line in server.stdout:
        print(line, end="", flush=True)
        if line.startswith("META "):
            metadata = line[5:].strip()
            match = re.search(r"(\d+\.\d+\.\d+\.\d+):(\d+)", metadata)
            if not match:
                raise RuntimeError(f"Unrecognized endpoint metadata: {metadata}")
            subprocess.run(
                ["p2p/notif_e2e", "client", *match.groups()],
                check=True,
                env=env,
                timeout=60,
            )
            break
    else:
        raise RuntimeError("Server did not advertise metadata")
    for line in server.stdout:
        print(line, end="", flush=True)
    if server.wait(timeout=60):
        raise RuntimeError("Notification server failed")
finally:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
