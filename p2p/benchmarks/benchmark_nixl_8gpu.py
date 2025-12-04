import os
import time
import zmq
import argparse
import multiprocessing as mp

from benchmark_nixl import (
    create_dataset,
    create_nixl_agent_mc,
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
BARRIER_BASE_PORT = 29000
SIZE_BARRIER_BASE_PORT = 30000

DEFAULT_SIZES = [
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    10485760,
    16777216,
    104857600,
    104857600 * 2,
]
# DEFAULT_SIZES = [104857600 * 2] * 10

NUM_KV = 1
BACKEND = "uccl_p2p"
OP = "WRITE"
ITERS = 10


def barrier_wait(is_server, peer_ip, pair_idx):
    port = BARRIER_BASE_PORT + pair_idx
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    if is_server:
        sock.bind(f"tcp://*:{port}")
        sock.recv_string()
        sock.send_string("START")
    else:
        sock.connect(f"tcp://{peer_ip}:{port}")
        sock.send_string("READY")
        sock.recv_string()
    sock.close()
    ctx.term()


def size_barrier_wait(is_server, peer_ip, pair_idx, size_idx):
    port = SIZE_BARRIER_BASE_PORT + size_idx + pair_idx * 100

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)

    if is_server:
        sock.bind(f"tcp://*:{port}")
        sock.recv_string()
        sock.send_string("GO")
    else:
        sock.connect(f"tcp://{peer_ip}:{port}")
        sock.send_string("READY")
        sock.recv_string()

    sock.close()
    ctx.term()


def run_one_size(role, sock, agent, register_descs, size, pair_idx, warmup=2):
    local_desc = register_descs.trim()

    # warmup
    for _ in range(warmup):
        if role == "server":
            msg = sock.recv_string()
            if msg != "START":
                raise RuntimeError("Invalid START (warmup)")
            sock.send(agent.get_serialized_descs(local_desc))
            sock.recv()  # wait client done
        else:
            sock.send_string("START")
            remote_desc = agent.deserialize_descs(sock.recv())
            handle = agent.initialize_xfer(OP, local_desc, remote_desc, "server")
            do_transfer_mc(role, agent, handle, sock)

    if role == "server":
        msg = sock.recv_string()
        if msg != "START":
            raise RuntimeError("Invalid START")
        sock.send(agent.get_serialized_descs(local_desc))
        t0 = time.perf_counter()
        sock.recv()
        t1 = time.perf_counter()
    else:
        sock.send_string("START")
        remote_desc = agent.deserialize_descs(sock.recv())
        handle = agent.initialize_xfer(OP, local_desc, remote_desc, "server")
        t0 = time.perf_counter()
        do_transfer_mc(role, agent, handle, sock)
        t1 = time.perf_counter()

    elapsed = t1 - t0
    gbps = (size * 8 / elapsed) / 1e9
    gbs = (size / elapsed) / 1e9

    print(
        f"[{role} {pair_idx}] {size/1024/1024:8.1f} MB : {gbps:6.2f} Gbps | {gbs:6.2f} GB/s | {elapsed:.6f} s"
    )
    return gbps, gbs, elapsed


def run_one_pair(is_server, peer_ip, local_gpu, pair_idx, sizes, result_queue):
    uccl_port = UCCL_BASE_PORT + pair_idx
    zmq_port = ZMQ_BASE_PORT + pair_idx
    role = "server" if is_server else "client"
    os.environ["UCCL_TCPX_OOB_PORT"] = str(uccl_port)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    if is_server:
        sock.bind(f"tcp://*:{zmq_port}")
    else:
        sock.connect(f"tcp://{peer_ip}:{zmq_port}")

    max_size = max(sizes)
    dataset = create_dataset(role, max_size, NUM_KV, "gpu", local_gpu)
    agent, register_descs = create_nixl_agent_mc(
        role, dataset, sock, local_gpu, BACKEND
    )

    barrier_wait(is_server, peer_ip, pair_idx)

    for size_idx, size in enumerate(sizes):
        size_barrier_wait(is_server, peer_ip, pair_idx, size_idx)
        gbps, gbs, elapsed = run_one_size(
            role, sock, agent, register_descs, size, pair_idx, warmup=2
        )
        result_queue.put((pair_idx, size_idx, size, gbps, gbs, elapsed))

    cleanup_transfer(agent, None, register_descs)
    cleanup_agent(agent)
    sock.close()
    ctx.term()


def run_all_pairs(is_server, peer_ip, sizes):
    result_queue = mp.Queue()
    procs = []

    for idx, (gpu_s, gpu_c) in enumerate(GPU_PAIRS):
        local_gpu = gpu_s if is_server else gpu_c
        p = mp.Process(
            target=run_one_pair,
            args=(is_server, peer_ip, local_gpu, idx, sizes, result_queue),
        )
        p.start()
        procs.append(p)

    if not is_server:
        total_pairs = len(GPU_PAIRS)
        total_sizes = len(sizes)
        total_results = total_pairs * total_sizes

        results = []
        for _ in range(total_results):
            results.append(result_queue.get())

        from collections import defaultdict

        agg = defaultdict(list)

        for pair_idx, size_idx, size, gbps, gbs, elapsed in results:
            agg[size_idx].append((size, gbps, gbs))

        print("\n=== Aggregated Results (CLIENT SIDE) ===")
        for size_idx in range(len(sizes)):
            size = sizes[size_idx]
            gbps_list = [x[1] for x in agg[size_idx]]
            gbs_list = [x[2] for x in agg[size_idx]]

            avg_gbps = sum(gbps_list) / len(gbps_list)
            avg_gbs = sum(gbs_list) / len(gbs_list)

            print(
                f"[#{size_idx:02d}] SIZE {size/1024/1024:8.1f} MB | "
                f"Avg Gbps: {avg_gbps:6.2f} | Avg GB/s: {avg_gbs:6.2f}"
            )

    for p in procs:
        p.join()


def run_server(sizes):
    run_all_pairs(True, None, sizes)


def run_client(server_ip, sizes):
    run_all_pairs(False, server_ip, sizes)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["server", "client"])
    p.add_argument("--server-ip", default=None)
    p.add_argument(
        "--sizes",
        default=",".join(str(s) for s in DEFAULT_SIZES),
    )
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    if args.role == "server":
        run_server(sizes)
    else:
        if not args.server_ip:
            raise RuntimeError("client requires --server-ip")
        run_client(args.server_ip, sizes)
