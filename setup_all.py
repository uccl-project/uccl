import sys
from shared import *
import signal
import argparse
import os

num_queues = 12
core_count = os.cpu_count()
num_irqcores = int(num_queues)

config_mapping = {
    "aws_c5_afxdp": ["AWS_C5", "ens6", 3498],
    "aws_g4_afxdp": ["AWS_G4", "ens6", 3498],
    "aws_g4metal_afxdp": ["AWS_G4METAL", "enp199s0", 3498],
    "cloudlab_xl170_afxdp": ["CLOUDLAB_XL170", "ens1f1np1", 1500],
    "cloudlab_d6515_afxdp": ["CLOUDLAB_D6515", "enp65s0f0np0", 3498],
    #
    "aws_c5_tcp": ["AWS_C5", "ens6", 9001],
    "aws_g4_tcp": ["AWS_G4", "ens6", 9001],
    "aws_g4metal_tcp": ["AWS_G4METAL", "enp199s0", 9001],
    "cloudlab_xl170_tcp": ["CLOUDLAB_XL170", "ens1f1np1", 1500],
    "cloudlab_d6515_tcp": ["CLOUDLAB_D6515", "enp65s0f0np0", 9000],
    #
    "setup_extra": ["", "", 0],
}
PYTHON = "source /opt/anaconda3/bin/activate; conda run -n base python"

# Usage:
#   python setup_all.py --target=setup_extra
#   python setup_all.py --target=cloudlab_xl170_afxdp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parsing setup_all arguments.")

    parser.add_argument(
        "--target",
        type=str,
        default="cloudlab_xl170_afxdp",
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

    nodes = get_nodes()
    print(f"Nodes: {nodes}")

    node_clients = [paramiko.SSHClient() for _ in nodes]
    for node, node_client in zip(nodes, node_clients):
        node_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        node_client.connect(node)

    if target == "setup_extra":
        _ = exec_command_and_wait(
            node_clients[0],
            f"cd /opt/uccl; {PYTHON} rsync.py",
        )
        wait_handler_vec = []
        for node_client in node_clients:
            wait_handler = exec_command_no_wait(
                node_client, f"cd /opt/uccl; ./setup_extra.sh"
            )
            wait_handler_vec.append(wait_handler)
        for wait_handler in wait_handler_vec:
            _ = wait_handler.wait()
        exit(0)

    _ = exec_command_and_wait(
        node_clients[0],
        f'cd /opt/uccl/afxdp; make -j "CXXFLAGS=-D{make_macro}"; cd playground; make -j "CXXFLAGS=-D{make_macro}"',
    )

    _ = exec_command_and_wait(
        node_clients[0],
        f"cd /opt/uccl; {PYTHON} rsync.py",
    )

    ### Setup NIC
    afxdp_or_tcp = "afxdp" if "afxdp" in target else "tcp"
    aws_or_cloudlab = "aws" if "aws" in target else "cloudlab"
    if target == "aws_g4metal_tcp":
        core_count = 32
        num_irqcores = 32
    elif target == "aws_c5_tcp":
        core_count = 32
        num_irqcores = 32
    elif target == "cloudlab_d6515_tcp":
        core_count = 63
        num_irqcores = 63

    if afxdp_or_tcp == "afxdp":
        nic_cmd = f"./setup_nic.sh {net_dev} {num_queues} {num_irqcores} {mtu} {afxdp_or_tcp} {aws_or_cloudlab}"
    else:
        nic_cmd = f"./setup_nic.sh {net_dev} {core_count} {core_count} {mtu} {afxdp_or_tcp} {aws_or_cloudlab}"

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

    wait_handler_vec = []
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
