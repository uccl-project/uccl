import sys
from shared import *
import signal
import argparse
import os

config_mapping = {
    "aws_c5_afxdp": ["AWS_C5", "ens6", 3498],
    "aws_g4_afxdp": ["AWS_G4", "ens6", 3498],
    "aws_g4metal_afxdp": ["AWS_G4METAL", "enp199s0", 3498],
    "clab_xl170_afxdp": ["CLAB_XL170", "ens1f1np1", 1500],
    "clab_d6515_afxdp": ["CLAB_D6515", "enp65s0f0np0", 3498],
    #
    "aws_c5_tcp": ["AWS_C5", "ens6", 9001],
    "aws_g4_tcp": ["AWS_G4", "ens6", 9001],
    "aws_g4metal_tcp": ["AWS_G4METAL", "enp199s0", 9001],
    "clab_xl170_tcp": ["CLAB_XL170", "ens1f1np1", 1500],
    "clab_d6515_tcp": ["CLAB_D6515", "enp65s0f0np0", 9000],
    #
    "aws_c5_tcp_3kmtu": ["AWS_C5", "ens6", 3498],
    "clab_d6515_tcp_3kmtu": ["CLAB_D6515", "enp65s0f0np0", 3498],
    #
    "ibm_gx3_afxdp": ["IBM_GX3", "ens3", 1500],
    "ibm_gx3_tcp": ["IBM_GX3", "ens3", 1500],
    #
    "tpu_v6e8_tcp": ["TPU_V6E8", "ens8", 1460],
    "tpu_v6e8_afxdp": ["TPU_V6E8", "ens8", 1460],
    #
    "setup_extra": ["", "", 0],
}

UCCL_HOME = os.getenv("UCCL_HOME")
if not UCCL_HOME:
    raise ValueError("UCCL_HOME environment variable is not set.")


async def sync_repo(node_file):
    await run_command(
        f"cd {UCCL_HOME}/scripts; UCCL_HOME={UCCL_HOME} python rsync.py --node_file {node_file}",
    )


async def make_afxdp(make_macro, target):
    make_cmd = (
        f'cd {UCCL_HOME}/afxdp; make -j "CXXFLAGS=-D{make_macro}" transport_test afxdp_daemon_main'
        if "tpu" in target
        else f'cd {UCCL_HOME}/afxdp; make -j "CXXFLAGS=-D{make_macro}"'
    )
    await run_command(make_cmd)


# Usage:
#   python setup_all.py --target=setup_extra
#   python setup_all.py --target=clab_xl170_afxdp
#   python setup_all.py --target=tpu_v6e8_afxdp --node_file=node_ips/tpu_2.txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parsing setup_all arguments.")

    parser.add_argument(
        "--target",
        type=str,
        default="default",
        help=f'{", ".join(list(config_mapping.keys()))}',
    )
    parser.add_argument(
        "--node_file",
        type=str,
        default="node_ips/default.txt",
        help=f"Path to file containing list of nodes",
    )

    args = parser.parse_args()
    target = args.target
    node_file = args.node_file

    if target not in config_mapping:
        print("target not found!")
        exit(0)

    nodes = get_nodes(node_file)
    print(f"Nodes: {nodes}")
    node_clients = [paramiko.SSHClient() for _ in nodes]
    for node, node_client in zip(nodes, node_clients):
        node_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        node_client.connect(node)

    if target == "setup_extra":
        asyncio.run(sync_repo(node_file))
        wait_handler_vec = []
        for node_client in node_clients:
            wait_handler = exec_command_no_wait(
                node_client, f"cd {UCCL_HOME}/scripts; ./setup_extra.sh"
            )
            wait_handler_vec.append(wait_handler)
        for wait_handler in wait_handler_vec:
            _ = wait_handler.wait()
        exit(0)

    make_macro = config_mapping[target][0]
    net_dev = config_mapping[target][1]
    mtu = config_mapping[target][2]

    asyncio.run(make_afxdp(make_macro, target))
    asyncio.run(sync_repo(node_file))

    ### Setup NIC
    num_queues = parse_num_queues(
        make_macro, f"{UCCL_HOME}/afxdp/transport_config.h"
    )
    if num_queues is None:
        print("NUM_QUEUES not found!")
        exit(0)

    core_count = os.cpu_count()
    num_irqcores = int(num_queues)
    afxdp_or_tcp = "afxdp" if "afxdp" in target else "tcp"
    platform = target.split("_", 1)[0]

    if "aws_g4metal_tcp" in target:
        core_count = 32
        num_irqcores = 32
    elif "aws_c5_tcp" in target:
        core_count = 32
        num_irqcores = 32
    elif "clab_d6515_tcp" in target:
        core_count = 63
        num_irqcores = 63
    elif "ibm_gx3_tcp" in target:
        num_irqcores = 16
    elif "tpu_v6e8_tcp" in target:
        num_irqcores = 16

    if afxdp_or_tcp == "afxdp":
        nic_cmd = f"./setup_nic.sh {net_dev} {num_queues} {num_irqcores} {mtu} {afxdp_or_tcp} {platform}"
    else:
        nic_cmd = f"./setup_nic.sh {net_dev} {core_count} {core_count} {mtu} {afxdp_or_tcp} {platform}"

    wait_handler_vec = []
    for node_client in node_clients:
        wait_handler = exec_command_no_wait(
            node_client, f"cd {UCCL_HOME}/scripts; {nic_cmd}"
        )
        wait_handler_vec.append(wait_handler)
    for wait_handler in wait_handler_vec:
        _ = wait_handler.wait()

    if afxdp_or_tcp == "tcp":
        exit(0)

    wait_handler_vec = []
    for node_client in node_clients:
        print(f"Running AFXDP daemon")
        wait_handler = exec_command_no_wait(
            node_client,
            f"cd {UCCL_HOME}/afxdp; sudo pkill afxdp_daemon; sudo ./afxdp_daemon_main --logtostderr=1",
        )
        wait_handler_vec.append(wait_handler)

    def signal_handler(sig, frame):
        for wait_handler in wait_handler_vec:
            wait_handler.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for wait_handler in wait_handler_vec:
        _ = wait_handler.wait()
