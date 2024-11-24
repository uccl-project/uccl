import sys

sys.path.append("../")
from common import *
import signal
import argparse
import os

core_count = os.cpu_count()
num_queues = 1
num_irqcores = int(num_queues)

config_mapping = {
    "aws_afxdp_c5": ["AWS_C5", "ens6", 3498],
    "aws_afxdp_g4": ["AWS_G4", "ens6", 3498],
    "aws_afxdp_g4_metal": ["AWS_G4_METAL", "enp199s0", 3498],
    "cloudlab_afxdp_xl170": ["CLOUDLAB_XL170", "ens1f1np1", 1500],
    "cloudlab_afxdp_d6515": ["CLOUDLAB_D6515", "enp65s0f0np0", 3498],
    #
    "aws_tcp_c5": ["AWS_C5", "ens6", 9001],
    "aws_tcp_g4": ["AWS_G4", "ens6", 9001],
    "aws_tcp_g4_metal": ["AWS_G4_METAL", "enp199s0", 9001],
    "cloudlab_tcp_xl170": ["CLOUDLAB_XL170", "ens1f1np1", 1500],
    "cloudlab_tcp_d6515": ["CLOUDLAB_D6515", "enp65s0f0np0", 9000],
}
PYTHON = "source /opt/anaconda3/bin/activate; conda run -n base python"

# Usage: python setup_all.py --target=cloudlab_afxdp_xl170

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parsing setup_all arguments.")

    parser.add_argument(
        "--target",
        type=str,
        default="cloudlab_afxdp_xl170",
        help=f'{", ".join(list(config_mapping.keys()))}',
    )

    args = parser.parse_args()
    target = args.target

    if target not in config_mapping:
        print("target not found!")
        exit(0)

    make_macro = config_mapping[target][0]
    net_dev = config_mapping[target][1]
    mtu = config_mapping[target][2]

    nodes = read_nodes()
    print(f"Nodes: {nodes}")

    node_clients = [paramiko.SSHClient() for _ in nodes]
    for node, node_client in zip(nodes, node_clients):
        node_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        node_client.connect(node)

    _ = exec_command_and_wait(
        node_clients[0],
        f'cd /opt/uccl/afxdp; make -j "CXXFLAGS=-D{make_macro}"',
    )

    _ = exec_command_and_wait(
        node_clients[0],
        f"cd /opt/uccl; {PYTHON} rsync.py",
    )

    afxdp_or_tcp = "afxdp" if "afxdp" in target else "tcp"
    aws_or_cloudlab = "aws" if "aws" in target else "cloudlab"
    if target == "aws_tcp_g4_metal":
        core_count = 32
        num_irqcores = 32
    elif target == "cloudlab_tcp_d6515":
        core_count = 63
        num_irqcores = 63

    if afxdp_or_tcp == "afxdp":
        nic_cmd = f"./config_nic.sh {net_dev} {num_queues} {num_irqcores} {mtu} {afxdp_or_tcp} {aws_or_cloudlab}"
    else:
        nic_cmd = f"./config_nic.sh {net_dev} {core_count} {core_count} {mtu} {afxdp_or_tcp} {aws_or_cloudlab}"

    wait_handler_vec = []
    for node_client in node_clients:
        wait_handler = exec_command_no_wait(
            node_client, f"cd /opt/uccl; {nic_cmd}"
        )
        wait_handler_vec.append(wait_handler)
    for wait_handler in wait_handler_vec:
        _ = wait_handler.wait()

    if afxdp_or_tcp == "tcp":
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
