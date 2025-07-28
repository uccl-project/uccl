from __future__ import annotations

import argparse
import sys
import time
from typing import List
import traceback
import zmq
import io
import sys

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError as exc:
    sys.stderr.write("Failed to import NIXL\n")
    raise

try:
    import torch
except ImportError as exc:
    sys.stderr.write("Failed to import torch\n")
    raise

listen_port = 9000


def create_dataset(role, size, device, gpu_idx=0):
    """
    Create a dataset of tensors whose total size is at least size in bytes.
    """
    dtype = torch.float32
    num_blocks = 1
    value = 0 if "server" in role else 1

    element_size = torch.tensor([], dtype=dtype).element_size()
    n_elems_per_block = size // (element_size * num_blocks)
    if n_elems_per_block == 0:
        n_elems_per_block = 1

    dataset = []
    if device == "gpu":
        dev = f"cuda:{gpu_idx}"
    else:
        dev = "cpu"
    for _ in range(num_blocks):
        block = torch.full((n_elems_per_block,), value, device=dev, dtype=dtype)
        dataset.append(block)

    # If total size is less than requested, add more elements to the last block
    total_bytes = sum(t.numel() * t.element_size() for t in dataset)
    if total_bytes < size:
        extra_elems = (size - total_bytes) // element_size
        if extra_elems > 0:
            extra_block = torch.full((extra_elems,), value, device=device, dtype=dtype)
            dataset.append(extra_block)

    return dataset


def init_zmq(host, port, role):
    """
    Initialize the ZMQ socket for communication.
    """
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PAIR)
    if "server" in role:
        zmq_socket.bind(f"tcp://{host}:{port}")
    else:
        zmq_socket.connect(f"tcp://{host}:{port}")
        # Ensure the socket is ready to receive messages
        zmq_socket.setsockopt(zmq.LINGER, 0)

    return zmq_socket


def initialize_xfer_metadata_mc(
    role: str, operation: str, agent: nixl_agent, register_descs, zmq_socket
):
    """
    Initialize transfer metadata with zmq sockets for Mooncake
    """
    local_xfer_descs = register_descs.trim()
    remote_xfer_descs = None
    transfer_handle = None

    if "server" in role:
        # Wait until there is a message from the creator
        msg = zmq_socket.recv().decode("utf-8")
        if msg == "START":
            pass
        else:
            print(f"{role} received unexpected message: {msg}")
            zmq_socket.close()
            exit(0)

        # send the xfer descs to the peer
        zmq_socket.send(agent.get_serialized_descs(local_xfer_descs))

    elif "client" in role:
        zmq_socket.send("START".encode("utf-8"))

        # Wait until there is a message from the peer
        msg = zmq_socket.recv()
        remote_xfer_descs = agent.deserialize_descs(msg)

        uid = "TRANSFER"
        transfer_handle = agent.initialize_xfer(
            operation, local_xfer_descs, remote_xfer_descs, "server"
        )

    return transfer_handle


def initialize_xfer_metadata_ucx(
    role: str,
    operation: str,
    agent: nixl_agent,
    register_descs,
    server_ip,
    server_port,
):
    """
    Initialize transfer metadata.
    """

    local_xfer_descs = register_descs.trim()
    remote_xfer_descs = None
    transfer_handle = None

    if "server" in role:
        # Wait until there is a message from the creator
        while not agent.check_remote_metadata("client"):
            continue

        # send the xfer descs to the peer
        desc = agent.get_serialized_descs(local_xfer_descs)
        agent.send_notif("client", desc)

    elif "client" in role:
        agent.fetch_remote_metadata("server", server_ip, server_port)
        agent.send_local_metadata(server_ip, server_port)
        # Wait until there is a message from the peer
        notifs = agent.get_new_notifs()
        while len(notifs) == 0:
            notifs = agent.get_new_notifs()

        remote_xfer_descs = agent.deserialize_descs(notifs["server"][0])
        while not agent.check_remote_metadata("server"):
            continue
        uid = "TRANSFER"
        transfer_handle = agent.initialize_xfer(
            operation, local_xfer_descs, remote_xfer_descs, "server", uid
        )

    return transfer_handle


def create_nixl_agent_mc(role: str, dataset, zmq_socket):
    """
    Create Nixl agents based on the role with Mooncake backend
    """
    config = nixl_agent_config(backends=["Mooncake"])
    agent = nixl_agent(role, config)
    descs = agent.get_reg_descs(dataset)
    register_descs = agent.register_memory(descs)
    local_meta = agent.get_agent_metadata()

    if "client" in role:
        zmq_socket.send(local_meta)
        remote_meta = zmq_socket.recv()
        agent.add_remote_agent(remote_meta).decode("utf-8")
    elif "server" in role:
        remote_meta = zmq_socket.recv()
        agent.add_remote_agent(remote_meta).decode("utf-8")
        zmq_socket.send(local_meta)

    return agent, register_descs


def create_nixl_agent_ucx(role: str, dataset):
    """
    Create Nixl agents based on the role.
    """
    port = listen_port
    if role == "client":
        port = 0
    config = nixl_agent_config(True, True, port)
    agent = nixl_agent(role, config)
    descs = agent.get_reg_descs(dataset)
    register_descs = agent.register_memory(descs)
    return agent, register_descs


def start_transfer_mc(
    role: str, agent: nixl_agent, transfer_handle, zmq_socket, uid="TRANSFER"
):
    if "client" in role:
        state = agent.transfer(transfer_handle)
        if state == "ERR":
            print("Error in transfer")
        while True:
            state = agent.check_xfer_state(transfer_handle)
            if state == "DONE":
                zmq_socket.send(uid.encode("utf-8"))
                break
            elif state == "ERR":
                print("Error in transfer")
                break
    else:
        while True:
            transfer_done = zmq_socket.recv()
            if transfer_done.decode("utf-8") == uid:
                break


def start_transfer_ucx(role: str, agent: nixl_agent, transfer_handle, uid="TRANSFER"):
    if "client" in role:
        state = agent.transfer(transfer_handle)
        if state == "ERR":
            print("Error in transfer")
        while True:
            state = agent.check_xfer_state(transfer_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                print("Error in transfer")
                break
    else:
        uid = "TRANSFER"
        while not agent.check_remote_xfer_done("client", uid.encode("utf-8")):
            continue


def cleanup_transfer(
    agent: nixl_agent,
    transfer_handle,
    register_descs,
):
    # Cleanup the transfer handle and registered descriptors
    if transfer_handle is not None:
        agent.release_xfer_handle(transfer_handle)
    agent.deregister_memory(register_descs)


def cleanup_agent(
    agent: nixl_agent,
):
    agent.remove_remote_agent(agent.name)


def start_agent_pair(size, args):
    op = "WRITE"
    zmq_socket = None

    if args.backend == "mooncake":
        zmq_socket = init_zmq(args.remote_ip, listen_port, args.role)
    try:
        dataset = create_dataset(args.role, size, args.device, args.local_gpu_idx)

        # Suppress stdout for better output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        if args.backend == "mooncake":
            agent, register_descs = create_nixl_agent_mc(args.role, dataset, zmq_socket)
            transfer_handle = initialize_xfer_metadata_mc(
                args.role, op, agent, register_descs, zmq_socket
            )
        else:
            agent, register_descs = create_nixl_agent_ucx(args.role, dataset)
            transfer_handle = initialize_xfer_metadata_ucx(
                args.role,
                op,
                agent,
                register_descs,
                args.remote_ip,
                listen_port,
            )
        sys.stdout = old_stdout

        total_size = 0
        start = time.perf_counter()

        if args.backend == "mooncake":
            for n in range(args.iters):
                start_transfer_mc(args.role, agent, transfer_handle, zmq_socket)
                total_size += size
        else:
            for n in range(args.iters):
                start_transfer_ucx(
                    args.role,
                    agent,
                    transfer_handle,
                )
                total_size += size

        end = time.perf_counter()

        transfer_time = end - start
        gbps = (total_size * 8) / transfer_time / 1e9  # bits per second → Gbps
        gb_sec = total_size / transfer_time / 1e9  # bytes per second → GB/s
        lat = transfer_time / args.iters
        print(
            f"[{args.role}] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s | {lat:6.6f} s"
        )
        if "server" in args.role:
            for i, block in enumerate(dataset):
                assert torch.mean(block) - 1 < 1e-8, f"Block {i} not equal to 1"

    except KeyboardInterrupt:
        return 0.0
    except Exception as e:
        print(f"Error in agent pair {args.role}: {traceback.format_exc()}")
        return 0.0
    finally:
        cleanup_transfer(
            agent,
            transfer_handle,
            register_descs,
        )
        cleanup_agent(agent)
        if args.backend == "mooncake":
            zmq_socket.close()


def _pretty_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    val = float(num_bytes)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024
    return f"{num_bytes} B"  # fallback


def parse_size_list(val: str) -> List[int]:
    try:
        return [int(s) for s in val.split(",") if s]
    except ValueError:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers")


def main():
    p = argparse.ArgumentParser(description="Benchmark NIXL/UCX bandwidth")
    p.add_argument(
        "--role",
        choices=["server", "client"],
        required=True,
        help="Run as server (receiver) or client (sender)",
    )
    p.add_argument(
        "--remote-ip",
        default="0.0.0.0",
        help="Server IP address (client only)",
    )
    p.add_argument(
        "--local-gpu-idx",
        type=int,
        default=0,
        help="Local GPU index to bind buffers",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Buffer location (cpu or gpu)",
    )
    p.add_argument(
        "--sizes",
        type=parse_size_list,
        default=[
            256,
            1024,
            4096,
            16384,
            65536,
            262144,
            1048576,
            10485760,
            104857600,
        ],
        help="Comma separated list of message sizes in bytes",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="Iterations per message size (excluding 1 warm-up)",
    )
    p.add_argument(
        "--num-kvblocks",
        type=int,
        default=1,
        help="Number of key-value blocks to send/recv in a single call",
    )
    p.add_argument(
        "--backend",
        choices=["ucx", "mooncake"],
        default="ucx",
        help="Backend that nixl will use for the data transfer",
    )
    args = p.parse_args()

    print("NIXL P2P Benchmark — role:", args.role)
    print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    print(
        f"Device: {args.device} | Local GPU idx: {args.local_gpu_idx} | Iterations: {args.iters}"
    )
    for size in args.sizes:
        start_agent_pair(size, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
