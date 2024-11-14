import sys

sys.path.append("../")
from common import *
import signal
import argparse
import os

core_count = os.cpu_count()
num_queues = 2

make_macro_mapping = {
    "aws_afxdp": "AWS_ENA",
    "cloudlab_afxdp": "CLOUDLAB_MLX5",
    "aws_tcp": "AWS_ENA",
    "cloudlab_tcp": "CLOUDLAB_MLX5",
}

config_nic_cmd_mapping = {
    "aws_afxdp": f"./config_nic.sh ens6 {num_queues} 3498 afxdp aws",
    "cloudlab_afxdp": f"./config_nic.sh ens1f1np1 {num_queues} 1500 afxdp cloudlab",
    "aws_tcp": f"./config_nic.sh ens6 {core_count} 9001 tcp aws",
    "cloudlab_tcp": f"./config_nic.sh ens1f1np1 {core_count} 1500 tcp cloudlab",
}


def read_nodes():
    with open("nodes.txt", "r") as file:
        return [
            line.strip()
            for line in file
            if not line.strip().startswith("#") and line.strip()
        ]


# Usage: python setup_all.py --target aws_afxdp|cloudlab_afxdp|aws_tcp|cloudlab_tcp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parsing setup_all arguments.")

    parser.add_argument(
        "--target",
        type=str,
        default="cloudlab_afxdp",
        help="aws_afxdp, cloudlab_afxdp, aws_tcp, cloudlab_tcp",
    )

    args = parser.parse_args()
    target = args.target

    nodes = read_nodes()
    print(f"Nodes: {nodes}")

    node_clients = [paramiko.SSHClient() for _ in nodes]
    for node, node_client in zip(nodes, node_clients):
        node_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        node_client.connect(node)

    _ = exec_command_and_wait(
        node_clients[0],
        f'cd /opt/uccl/afxdp; make -j "CXXFLAGS=-D{make_macro_mapping[target]}"',
    )

    _ = exec_command_and_wait(node_clients[0], f"cd /opt/uccl; ./sync.sh")

    wait_handler_vec = []
    for node_client in node_clients:
        wait_handler = exec_command_no_wait(
            node_client, f"cd /opt/uccl; {config_nic_cmd_mapping[target]}"
        )
        wait_handler_vec.append(wait_handler)
    for wait_handler in wait_handler_vec:
        _ = wait_handler.wait()

    if target == "aws_tcp" or target == "cloudlab_tcp":
        exit(0)

    wait_handler_vec.clear()
    for node_client in node_clients:
        wait_handler = exec_command_no_wait(
            node_client,
            f"cd /opt/uccl/afxdp; sudo pkill afxdp_daemon; sudo ./afxdp_daemon_main --logtostderr=1",
        )
        wait_handler_vec.append(wait_handler)

    def signal_handler(sig, frame):
        for wait_handler in wait_handler_vec:
            wait_handler.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for wait_handler in wait_handler_vec:
        _ = wait_handler.wait()
