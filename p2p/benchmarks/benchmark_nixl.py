from __future__ import annotations

import argparse
import sys
import time
from typing import List
import traceback
import zmq
import io
import sys
import torch

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError as exc:
    sys.stderr.write("Failed to import NIXL\n")
    raise

listen_port = 9000


def create_dataset(role, size, num_kvblocks, device, gpu_idx=0):
    """
    Create a dataset of tensors whose total size is at least size in bytes.
    """
    dtype = torch.float32 if size >= 4 else torch.uint8
    value = 0 if "server" in role else 1

    element_size = torch.tensor([], dtype=dtype).element_size()
    n_elems_per_block = size // (element_size * num_kvblocks)
    if n_elems_per_block == 0:
        n_elems_per_block = 1

    dataset = []
    if device == "gpu":
        dev = f"cuda:{gpu_idx}"
    else:
        dev = "cpu"
    for _ in range(num_kvblocks):
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


def cleanup_agent(
    agent: nixl_agent,
):
    if agent is None:
        return
    peer = "server" if agent.name == "client" else "client"
    agent.remove_remote_agent(peer)


def cleanup_transfer(
    agent: nixl_agent,
    transfer_handle,
    register_descs,
):
    if agent is None:
        return
    # Cleanup the transfer handle and registered descriptors
    if transfer_handle is not None:
        agent.release_xfer_handle(transfer_handle)
    if register_descs is not None:
        agent.deregister_memory(register_descs)


def create_nixl_agent_mc(role: str, dataset, zmq_socket, device_idx, backend):
    """
    Create Nixl agents based on the role with Mooncake/UCCL backend
    """
    backend_name = (
        "Mooncake"
        if backend == "mooncake"
        else ("UCCL_P2P" if backend == "uccl" else "TCPX")
    )
    config = nixl_agent_config(backends=[backend_name])
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
        zmq_socket.send(local_meta)
        agent.add_remote_agent(remote_meta).decode("utf-8")

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


def init_transfer_metadata_mc(
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


def init_transfer_metadata_ucx(
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


def do_transfer_mc(
    role: str, agent: nixl_agent, transfer_handle, zmq_socket, uid="TRANSFER"
):
    if "client" in role:
        state = agent.transfer(transfer_handle)
        assert state != "ERR", "Error in transfer"
        while True:
            state = agent.check_xfer_state(transfer_handle)
            assert state != "ERR", "Error in transfer"
            if state == "DONE":
                zmq_socket.send(uid.encode("utf-8"))
                break
    else:
        while True:
            transfer_done = zmq_socket.recv()
            if transfer_done.decode("utf-8") == uid:
                break


def do_transfer_ucx(role: str, agent: nixl_agent, transfer_handle, uid="TRANSFER"):
    if "client" in role:
        state = agent.transfer(transfer_handle)
        assert state != "ERR", "Error in transfer"
        while True:
            state = agent.check_xfer_state(transfer_handle)
            assert state != "ERR", "Error in transfer"
            if state == "DONE":
                break
    else:
        uid = "TRANSFER"
        while not agent.check_remote_xfer_done("client", uid.encode("utf-8")):
            continue


def start_transfer(size, num_kvblocks, args):
    op = "WRITE" if args.op_type == "write" else "READ"
    zmq_socket = None

    if args.backend == "mooncake" or args.backend == "uccl" or args.backend == "tcpx":
        zmq_socket = init_zmq(args.remote_ip, listen_port, args.role)
    try:
        dataset = create_dataset(
            args.role, size, num_kvblocks, args.device, args.local_gpu_idx
        )

        agent = None
        transfer_handle = None
        register_descs = None

        # Suppress stdout for better output during agent setup
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        if (
            args.backend == "mooncake"
            or args.backend == "uccl"
            or args.backend == "tcpx"
        ):
            agent, register_descs = create_nixl_agent_mc(
                args.role, dataset, zmq_socket, args.local_gpu_idx, args.backend
            )
        else:
            agent, register_descs = create_nixl_agent_ucx(args.role, dataset)
        sys.stdout = old_stdout

        total_size = 0
        total_transfer_time = 0.0

        for _ in range(args.iters):
            if (
                args.backend == "mooncake"
                or args.backend == "uccl"
                or args.backend == "tcpx"
            ):
                transfer_handle = init_transfer_metadata_mc(
                    args.role, op, agent, register_descs, zmq_socket
                )
            else:
                transfer_handle = init_transfer_metadata_ucx(
                    args.role,
                    op,
                    agent,
                    register_descs,
                    args.remote_ip,
                    listen_port,
                )
            start = time.perf_counter()
            if (
                args.backend == "mooncake"
                or args.backend == "uccl"
                or args.backend == "tcpx"
            ):
                do_transfer_mc(args.role, agent, transfer_handle, zmq_socket)
                total_size += size
            else:
                do_transfer_ucx(args.role, agent, transfer_handle)
                total_size += size

            end = time.perf_counter()
            transfer_time = end - start
            total_transfer_time += transfer_time

        avg_transfer_time = total_transfer_time / args.iters
        gbps = (total_size * 8) / total_transfer_time / 1e9  # bits per second → Gbps
        gb_sec = total_size / total_transfer_time / 1e9  # bytes per second → GB/s
        lat = avg_transfer_time  # Average latency per transfer

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
        if (
            args.backend == "mooncake"
            or args.backend == "uccl"
            or args.backend == "tcpx"
        ):
            zmq_socket.close()


def create_nixl_agent_ucx_dual(role: str, send_dataset, recv_dataset):
    """
    Create Nixl agents based on the role.
    """
    port = listen_port
    config = nixl_agent_config(True, True, port)
    agent = nixl_agent(role, config)
    send_descs = agent.get_reg_descs(send_dataset)
    recv_descs = agent.get_reg_descs(recv_dataset)
    send_register_descs = agent.register_memory(send_descs)
    recv_register_descs = agent.register_memory(recv_descs)
    return agent, send_register_descs, recv_register_descs


def init_transfer_metadata_ucx_dual(
    role: str,
    agent: nixl_agent,
    send_register_descs,
    recv_register_descs,
    remote_ip,
    remote_port,
):
    """
    Initialize transfer metadata for dual WRITE operations.
    Both endpoints will write to each other simultaneously.
    Both endpoints do symmetric operations: establish connection, exchange descriptors, create transfer handles.
    """
    send_local_xfer_descs = send_register_descs.trim()
    recv_local_xfer_descs = recv_register_descs.trim()
    remote_peer = "server" if "client" in role else "client"

    # Both endpoints establish metadata connection (client connects, server waits)
    if "client" in role:
        agent.send_local_metadata(remote_ip, remote_port)
        agent.fetch_remote_metadata(remote_peer, remote_ip, remote_port)
        while not agent.check_remote_metadata(remote_peer):
            continue
    else:
        while not agent.check_remote_metadata(remote_peer):
            continue
        agent.send_local_metadata(remote_ip, remote_port)
        agent.fetch_remote_metadata(remote_peer, remote_ip, remote_port)

    # Both endpoints send their descriptors to peer
    recv_desc = agent.get_serialized_descs(recv_local_xfer_descs)
    agent.send_notif(remote_peer, recv_desc)

    # Both endpoints wait for peer descriptors
    notifs = agent.get_new_notifs()
    while len(notifs) == 0:
        notifs = agent.get_new_notifs()

    # Both endpoints parse remote descriptors
    remote_recv_xfer_descs = agent.deserialize_descs(
        notifs[remote_peer][0]
    )  # What we write to

    # Why twice? TBH, I do not know.
    while not agent.check_remote_metadata(remote_peer):
        continue

    # Both endpoints initialize transfer handles
    transfer_handle = agent.initialize_xfer(
        "WRITE",
        send_local_xfer_descs,
        remote_recv_xfer_descs,
        remote_peer,
        "TRANSFER",
    )

    return transfer_handle


def do_transfer_ucx_dual(role: str, agent: nixl_agent, transfer_handle):
    """
    Execute dual WRITE transfers where both endpoints simultaneously write to each other.
    Both endpoints do the same operations: initiate transfers and wait for remote completion.
    """
    # Both endpoints initiate their outbound transfers
    state = agent.transfer(transfer_handle)
    assert state != "ERR", "Error in transfer initiation"

    # Poll our outbound transfer and wait for remote inbound transfer to complete
    done = False
    remote_peer = "server" if "client" in role else "client"
    recv_uid = "TRANSFER"

    while not done or not agent.check_remote_xfer_done(
        remote_peer, recv_uid.encode("utf-8")
    ):
        # Check our outbound transfer
        if not done:
            state = agent.check_xfer_state(transfer_handle)
            assert state != "ERR", "Error in send transfer"
            done = state == "DONE"


def start_transfer_dual(size, num_kvblocks, args):
    """
    Dual direction transfer where both client and server simultaneously perform WRITE operations.
    Each endpoint writes data to the other endpoint concurrently.
    """
    assert (
        args.backend == "ucx"
    ), "Dual direction transfer only supported with UCX backend"

    try:
        # Create datasets for both sending and receiving
        # Dataset for sending (writing to remote)
        send_dataset = create_dataset(
            args.role, size, num_kvblocks, args.device, args.local_gpu_idx
        )
        # Dataset for receiving (where remote will write to)
        recv_dataset = create_dataset(
            (
                "server" if "client" in args.role else "client"
            ),  # opposite role for recv buffer
            size,
            num_kvblocks,
            args.device,
            args.local_gpu_idx,
        )

        agent = None
        transfer_handle = None
        send_register_descs = None
        recv_register_descs = None

        # Suppress stdout for better output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Create agents and register memory for both send and receive datasets
        agent, send_register_descs, recv_register_descs = create_nixl_agent_ucx_dual(
            args.role, send_dataset, recv_dataset
        )

        # Initialize dual transfer handles for bidirectional WRITE operations
        transfer_handle = init_transfer_metadata_ucx_dual(
            args.role,
            agent,
            send_register_descs,
            recv_register_descs,
            args.remote_ip,
            listen_port,
        )

        sys.stdout = old_stdout

        total_size = 0
        start = time.perf_counter()

        # Perform dual direction WRITE transfers for specified iterations
        for _ in range(args.iters):
            # Both endpoints perform WRITE operations simultaneously using dual function
            do_transfer_ucx_dual(args.role, agent, transfer_handle)
            total_size += size

        end = time.perf_counter()

        transfer_time = end - start
        gbps = (total_size * 8) / transfer_time / 1e9  # bits per second → Gbps
        gb_sec = total_size / transfer_time / 1e9  # bytes per second → GB/s
        lat = transfer_time / args.iters

        print(
            f"[{args.role}] DUAL-WRITE {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s | {lat:6.6f} s"
        )

        # Verify received data (should be opposite of what we sent)
        expected_value = 0 if "client" in args.role else 1
        for i, block in enumerate(recv_dataset):
            block_mean = torch.mean(block).item()
            assert (
                abs(block_mean - expected_value) < 1e-6
            ), f"Block {i} received value {block_mean}, expected {expected_value}"

    except KeyboardInterrupt:
        return 0.0
    except Exception as e:
        print(f"Error in dual write transfer {args.role}: {traceback.format_exc()}")
        return 0.0
    finally:
        # Cleanup transfer handles and registered descriptors
        cleanup_transfer(agent, transfer_handle, send_register_descs)
        cleanup_transfer(agent, None, recv_register_descs)
        cleanup_agent(agent)


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
        default="gpu",
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
            16777216,
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
        choices=["ucx", "mooncake", "uccl", "tcpx"],
        default="ucx",
        help="Backend that nixl will use for the data transfer",
    )
    p.add_argument(
        "--op-type",
        choices=["write", "read"],
        default="write",
        help="Operation that nixl will use for the data transfer",
    )
    p.add_argument(
        "--dual",
        action="store_true",
        help="Run the benchmark on two directions",
    )
    args = p.parse_args()

    assert not (
        args.dual and args.backend == "mooncake"
    ), "We do not support dual direction with Mooncake backend"
    assert not (
        args.dual and args.remote_ip == "0.0.0.0"
    ), "Remote IP must be set for dual direction transfer"

    print("NIXL P2P Benchmark — role:", args.role)
    print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    print("Number of key-value blocks per message:", args.num_kvblocks)
    print(
        f"Device: {args.device} | Local GPU idx: {args.local_gpu_idx} | Iterations: {args.iters}"
    )
    if args.dual:
        for size in args.sizes:
            start_transfer_dual(size, args.num_kvblocks, args)
    else:
        for size in args.sizes:
            start_transfer(size, args.num_kvblocks, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
