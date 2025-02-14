from shared import *
import os
from pathlib import Path

UCCL_HOME = os.getenv("UCCL_HOME")
uccl_parent_dir = Path(UCCL_HOME).parent if UCCL_HOME else None

def rsync(local_client, nodes):
    wait_handler_vec = []
    for node in nodes:
        wait_handler = exec_command_no_wait(
            local_client,
            f"rsync -auv -e 'ssh -o StrictHostKeyChecking=no' {UCCL_HOME} {node}:{uccl_parent_dir}",
        )
        wait_handler_vec.append(wait_handler)
    for wait_handler in wait_handler_vec:
        _ = wait_handler.wait()


nodes = get_nodes()
print(f"Nodes: {nodes}")

local_client = paramiko.SSHClient()
local_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
local_client.connect("localhost")

rsync(local_client, nodes)
