from common import *


def rsync(local_client, nodes):
    wait_handler_vec = []
    for node in nodes:
        wait_handler = exec_command_no_wait(
            local_client,
            f"rsync -auv -e 'ssh -o StrictHostKeyChecking=no' /opt/uccl/ {node}:/opt/uccl/",
        )
        wait_handler_vec.append(wait_handler)
    for wait_handler in wait_handler_vec:
        _ = wait_handler.wait()


nodes = read_nodes()
print(f"Nodes: {nodes}")

local_client = paramiko.SSHClient()
local_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
local_client.connect("localhost")

rsync(local_client, nodes)
