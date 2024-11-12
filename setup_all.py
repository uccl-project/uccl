import sys
sys.path.append("../")
from common import *
import signal

platform = "cloudlab"
nodes = ["192.168.6.1", "192.168.6.2"]

config_nic_cmd_mapping = {
    "aws": "./config_nic.sh ens6 1 3498 afxdp aws",
    "cloudlab": "./config_nic.sh ens1f1np1 1 1500 afxdp cloudlab",
}

make_macro_mapping = {
    "aws": "AWS",
    "cloudlab": "CLOUDLAB",
}

node_clients = [paramiko.SSHClient() for _ in nodes]
for node, node_client in zip(nodes, node_clients):
    node_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    node_client.connect(node)
    
_ = exec_command_and_wait(
    node_clients[0],
    f'cd /opt/uccl/afxdp; make -j "CXXFLAGS=-D{make_macro_mapping[platform]}"'
)

node_str = " ".join(nodes)
_ = exec_command_and_wait(
    node_clients[0],
    f'cd /opt/uccl; ./sync.sh {node_str}'
)

wait_handler_vec = []
for node_client in node_clients:
    wait_handler = exec_command_no_wait(
        node_client,
        f'cd /opt/uccl; {config_nic_cmd_mapping[platform]}'
    )
    wait_handler_vec.append(wait_handler)
for wait_handler in wait_handler_vec:
    _ = wait_handler.wait()

wait_handler_vec.clear()
for node_client in node_clients:
    wait_handler = exec_command_no_wait(
        node_client,
        f'cd /opt/uccl/afxdp; sudo pkill afxdp_daemon; sudo ./afxdp_daemon_main --logtostderr=1'
    )
    wait_handler_vec.append(wait_handler)

def signal_handler(sig, frame):
    for wait_handler in wait_handler_vec:
        wait_handler.kill()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

for wait_handler in wait_handler_vec:
    _ = wait_handler.wait()
