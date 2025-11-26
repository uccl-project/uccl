import os
import time
import threading
import zmq
import argparse

from benchmark_nixl import (
    create_dataset,
    create_nixl_agent_mc,
    init_transfer_metadata_mc,
    do_transfer_mc,
    cleanup_transfer,
    cleanup_agent,
)

GPU_PAIRS = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
]

UCCL_BASE_PORT = 28000
ZMQ_BASE_PORT = 38000
BARRIER_PORT = 29000

MESSAGE_SIZE = 67108864
NUM_KV = 1
BACKEND = "uccl_p2p"
OP = "WRITE"


def barrier_wait(is_server, peer_ip):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    if is_server:
        sock.bind(f"tcp://*:{BARRIER_PORT}")
        print("[Server] Waiting for client READY...")
        sock.recv_string()
        print("[Server] Broadcasting START.")
        sock.send_string("START")
    else:
        sock.connect(f"tcp://{peer_ip}:{BARRIER_PORT}")
        print("[Client] Sending READY...")
        sock.send_string("READY")
        sock.recv_string()
        print("[Client] Received START.")
    sock.close()
    ctx.term()


def run_one_pair(is_server, peer_ip, local_gpu, pair_idx):
    uccl_port = UCCL_BASE_PORT + pair_idx
    zmq_port = ZMQ_BASE_PORT + pair_idx

    role = "server" if is_server else "client"
    print(
        f"[PAIR {pair_idx}] {role} GPU={local_gpu} UCCL_PORT={uccl_port} ZMQ_PORT={zmq_port}"
    )

    os.environ["UCCL_TCPX_OOB_PORT"] = str(uccl_port)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    if is_server:
        sock.bind(f"tcp://*:{zmq_port}")
    else:
        sock.connect(f"tcp://{peer_ip}:{zmq_port}")

    dataset = create_dataset(role, MESSAGE_SIZE, NUM_KV, "gpu", local_gpu)
    agent, register_descs = create_nixl_agent_mc(
        role, dataset, sock, local_gpu, BACKEND
    )

    barrier_wait(is_server, peer_ip)

    local_desc = register_descs.trim()

    if is_server:
        msg = sock.recv_string()
        if msg != "START":
            raise RuntimeError("Invalid metadata START signal")
        sock.send(agent.get_serialized_descs(local_desc))
        sock.recv()
    else:
        sock.send_string("START")
        remote_desc = agent.deserialize_descs(sock.recv())
        handle = agent.initialize_xfer(OP, local_desc, remote_desc, "server")
        do_transfer_mc(role, agent, handle, sock)

    if is_server:
        cleanup_transfer(agent, None, register_descs)
    else:
        cleanup_transfer(agent, handle, register_descs)

    cleanup_agent(agent)

    sock.close()
    ctx.term()

    print(f"[PAIR {pair_idx}] DONE.")


def run_all_pairs(is_server, peer_ip):
    threads = []
    for idx, (gpu_s, gpu_c) in enumerate(GPU_PAIRS):
        local_gpu = gpu_s if is_server else gpu_c
        t = threading.Thread(
            target=run_one_pair, args=(is_server, peer_ip, local_gpu, idx)
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("All 8 GPU pairs finished.")


def run_server():
    run_all_pairs(is_server=True, peer_ip=None)


def run_client(server_ip):
    run_all_pairs(is_server=False, peer_ip=server_ip)


if __name__ == "__main__":
    """
    python benchmark_nixl_8gpu.py --role server
    python benchmark_nixl_8gpu.py --role client --server-ip <SERVER_IP>
    """
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["server", "client"])
    p.add_argument("--server-ip", default=None)
    args = p.parse_args()

    if args.role == "server":
        run_server()
    else:
        if not args.server_ip:
            raise RuntimeError("client requires --server-ip")
        run_client(args.server_ip)
