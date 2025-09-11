from __future__ import annotations
import argparse, sys, time, socket, struct
from typing import List
import torch.distributed as dist
import torch
import numpy as np
import os
from uccl import p2p
from uccl.transfer import TransferManager

# UCCL P2P read requires RC mode, as RDMA UC does not support one-sided read.
os.environ["UCCL_RCMODE"] = "1"

# parse_metadata is now provided by the C++ layer via p2p.Endpoint.parse_metadata()


def _make_buffer(n_bytes: int, device: str, gpu: int):
    n = n_bytes // 4
    if device == "gpu":
        buf = torch.ones(n, dtype=torch.float32, device=f"cuda:{gpu}")
        ptr = buf.data_ptr()
    else:
        buf = torch.ones(n, dtype=torch.float32, pin_memory=True)
        ptr = buf.data_ptr()
    return buf, ptr


def _pretty(num: int):
    units, val = ["B", "KB", "MB", "GB"], float(num)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024


def _run_server(args, manager, remote_oob_ip, local_rank):
    """Run as receiver (rank 1) - receives data from sender."""
    print("[Server] Waiting for connection...")
    conn_id = manager.accept()

    print("[Server] Connection established")

    for sz in args.sizes:
        recv_buf, recv_ptr = _make_buffer(sz, args.device, local_rank)

        # Register transfer
        transfer_id = manager.register_transfer(conn_id, recv_buf)

        # Warm-up receive
        manager.post_transfer_metadata(transfer_id)

        # Benchmark receives
        for _ in range(args.iters):
            manager.post_transfer_metadata(transfer_id)

        # Check if tensor received data correctly
        expected_value = float(sz)
        if not recv_buf.allclose(
            torch.tensor(expected_value, dtype=torch.float32).cuda()
        ):
            print(
                f"[Server] WARNING: Tensor not filled correctly for size {sz}: {recv_buf}"
            )

        # Cleanup transfer
        manager.deregister_transfer(transfer_id)

    print("[Server] Benchmark complete")


def _run_client(args, manager, remote_oob_ip, local_rank):
    """Run as sender (rank 0) - sends data to receiver."""

    # Parse remote OOB IP to get connection info
    remote_ip = remote_oob_ip
    receiver_port = args.listen_port  # Use same port as server

    print(f"[Client] Connecting to receiver at {remote_ip}:{receiver_port}...")
    conn_id = manager.connect(remote_ip, receiver_port)

    print("[Client] Connection established")

    for sz in args.sizes:
        # Create send buffer with known pattern
        send_buf, send_ptr = _make_buffer(sz, args.device, local_rank)
        send_buf.fill_(float(sz))  # Fill with size value for verification

        # Register transfer
        transfer_id = manager.register_transfer(conn_id, send_buf)

        # Warm-up send
        transfer_metadata = manager.fetch_transfer_metadata(transfer_id)
        poll_id = manager.do_transfer_async(transfer_id, transfer_metadata)

        # Wait for warmup to complete
        while not manager.check_transfer_done(transfer_id, poll_id):
            time.sleep(0.001)

        # Benchmark starts
        start = time.perf_counter()
        total = 0

        for _ in range(args.iters):
            poll_id = manager.do_transfer_async(transfer_id, transfer_metadata)

            # Wait for transfer to complete
            while not manager.check_transfer_done(transfer_id, poll_id):
                pass

            total += sz

        elapsed = time.perf_counter() - start

        print(
            f"[Client] {_pretty(sz):>8} : "
            f"{(total*8)/elapsed/1e9:6.2f} Gbps | "
            f"{total/elapsed/1e9:6.2f} GB/s | "
            f"{elapsed/args.iters:6.6f} s"
        )

        # Cleanup transfer
        manager.deregister_transfer(transfer_id)

    print("[Client] Benchmark complete")


def parse_sizes(v: str) -> List[int]:
    try:
        return [int(x) for x in v.split(",") if x]
    except ValueError:
        raise argparse.ArgumentTypeError("bad --sizes")


def main():
    p = argparse.ArgumentParser("UCCL transfer benchmark (one-sided)")
    p.add_argument("--num-cpus", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument(
        "--sizes",
        type=parse_sizes,
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
    )
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--listen-port", type=int, default=29999)
    args = p.parse_args()

    print("Sizes:", ", ".join(_pretty(s) for s in args.sizes))

    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"
    assert args.device == "gpu"
    local_rank = int(os.environ["LOCAL_RANK"])

    # Use different ports for sender and receiver
    if rank == 0:
        # Sender (client) - no need to listen
        manager = TransferManager(local_rank, args.num_cpus, args.listen_port + 1000)
    else:
        # Receiver (server) - listens on the specified port
        manager = TransferManager(local_rank, args.num_cpus, args.listen_port)

    local_metadata = manager.ep.get_metadata()

    if rank == 0:
        dist.send(torch.ByteTensor(list(local_metadata)), dst=1)
        remote_metadata_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_metadata_tensor, src=1)
        remote_metadata = bytes(remote_metadata_tensor.tolist())
    else:
        remote_metadata_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_metadata_tensor, src=0)
        dist.send(torch.ByteTensor(list(local_metadata)), dst=0)
        remote_metadata = bytes(remote_metadata_tensor.tolist())

    remote_oob_ip = p2p.Endpoint.parse_metadata(remote_metadata)[0]

    if rank == 0:
        _run_client(args, manager, remote_oob_ip, local_rank)
    elif rank == 1:
        _run_server(args, manager, remote_oob_ip, local_rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Ctrl-C] Aborted.")
        sys.exit(1)
