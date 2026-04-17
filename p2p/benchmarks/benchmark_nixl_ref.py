from __future__ import annotations

import argparse
import sys
import time
from typing import List
import traceback
from datetime import datetime

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

import numpy as np


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


def setup_device(device, gpu_idx=0):
    """
    Setup the device for tensor operations.
    """
    if device == "gpu":
        try:
            if torch.cuda.is_available():
                torch.set_default_device(f"cuda:{gpu_idx}")
            else:
                torch.set_default_device("cpu")
        except Exception as e:
            torch.set_default_device("cpu")
    else:
        torch.set_default_device("cpu")


def create_nixl_agent(role, port, device, gpu_idx=0):
    """
    Create and configure NIXL agent.
    """
    setup_device(device, gpu_idx)

    # Configure agent
    listen_port = port if role == "server" else 0
    config = nixl_agent_config(True, True, listen_port)
    agent = nixl_agent(role, config)

    return agent


def initialize_transfer(role, agent, dataset, peer_ip, peer_port, transfer_id):
    """
    Initialize transfer metadata and setup communication.
    """
    # Register memory
    descs = agent.get_reg_descs(dataset)
    register_descs = agent.register_memory(descs)

    if not register_descs:
        print("Memory registration failed.")
        return None, None, None

    local_descs = register_descs.trim()
    transfer_handle = None

    if role == "target":
        # Target waits for initiator to fetch metadata
        while not agent.check_remote_metadata("initiator"):
            continue

        # Send descriptors to initiator
        target_desc_str = agent.get_serialized_descs(local_descs)
        agent.send_notif("initiator", target_desc_str)

    else:  # initiator
        # Fetch remote metadata and send local metadata
        agent.fetch_remote_metadata("target", peer_ip, peer_port)
        agent.send_local_metadata(peer_ip, peer_port)

        # Wait for target's descriptors
        notifs = agent.get_new_notifs()
        while len(notifs) == 0:
            notifs = agent.get_new_notifs()

        target_descs = agent.deserialize_descs(notifs["target"][0])

        # Wait for remote metadata to be ready
        while not agent.check_remote_metadata("target"):
            continue

        # Initialize transfer
        transfer_handle = agent.initialize_xfer(
            "WRITE", local_descs, target_descs, "target", transfer_id
        )

    return agent, register_descs, transfer_handle


def execute_transfer(role, agent, transfer_handle, transfer_id):
    """
    Execute the transfer based on role.
    """
    if role == "initiator":
        # Start transfer
        state = agent.transfer(transfer_handle)
        if state == "ERR":
            print("Posting transfer failed.")
            return False

        # Wait for completion
        while True:
            state = agent.check_xfer_state(transfer_handle)
            if state == "ERR":
                print("Transfer got to Error state.")
                return False
            elif state == "DONE":
                break
    else:  # target
        # Wait for transfer to complete
        while not agent.check_remote_xfer_done("initiator", transfer_id):
            continue

    return True


def cleanup_transfer(agent, transfer_handle, register_descs, role, peer_ip, peer_port):
    """
    Cleanup transfer resources.
    """
    if transfer_handle is not None:
        agent.release_xfer_handle(transfer_handle)

    if register_descs is not None:
        agent.deregister_memory(register_descs)

    if role == "initiator":
        agent.remove_remote_agent("target")
        agent.invalidate_local_metadata(peer_ip, peer_port)


def run_benchmark(size, args):
    """
    Run a single benchmark iteration.
    """
    transfer_id = b"BENCHMARK_TRANSFER"

    try:
        # Create dataset
        dataset = create_dataset(args.role, size, args.device, args.local_gpu_idx)

        # Create agent
        agent = create_nixl_agent(args.role, args.port, args.device, args.local_gpu_idx)

        # Initialize transfer
        agent, register_descs, transfer_handle = initialize_transfer(
            args.role, agent, dataset, args.remote_ip, args.port, transfer_id
        )

        if agent is None:
            return None

        # Execute transfer
        success = execute_transfer(args.role, agent, transfer_handle, transfer_id)

        if not success:
            cleanup_transfer(
                agent,
                transfer_handle,
                register_descs,
                args.role,
                args.remote_ip,
                args.port,
            )
            return None

        # Verify data for target
        if args.role == "target":
            for i, block in enumerate(dataset):
                if not torch.allclose(block, torch.ones_like(block)):
                    print(f"Data verification failed for block {i}.")
                    cleanup_transfer(
                        agent,
                        transfer_handle,
                        register_descs,
                        args.role,
                        args.remote_ip,
                        args.port,
                    )
                    return None

        # Cleanup
        cleanup_transfer(
            agent, transfer_handle, register_descs, args.role, args.remote_ip, args.port
        )

        return size

    except Exception as e:
        print(f"Error in benchmark {args.role}: {traceback.format_exc()}")
        return None


def start_agent_pair(size, args):
    """
    Run multiple iterations of the benchmark.
    """
    total_size = 0
    successful_transfers = 0

    start = time.perf_counter()

    for n in range(args.iters):
        result = run_benchmark(size, args)
        if result is not None:
            total_size += result
            successful_transfers += 1

    end = time.perf_counter()

    if successful_transfers > 0:
        transfer_time = end - start
        gbps = (total_size * 8) / transfer_time / 1e9  # bits per second → Gbps
        gb_sec = total_size / transfer_time / 1e9  # bytes per second → GB/s
        lat = transfer_time / successful_transfers

        print(
            f"[{args.role}] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s | {lat:6.6f} s | {successful_transfers}/{args.iters} successful"
        )
    else:
        print(f"[{args.role}] {_pretty_size(size):>8} : All transfers failed")


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
        choices=["target", "initiator"],
        required=True,
        help="Run as target (receiver) or initiator (sender)",
    )
    p.add_argument(
        "--remote-ip",
        default="0.0.0.0",
        help="Target IP address (initiator only)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Port for communication",
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
    args = p.parse_args()

    print("NIXL P2P Benchmark — role:", args.role)
    print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    print(
        f"Device: {args.device} | Local GPU idx: {args.local_gpu_idx} | Iterations: {args.iters}"
    )
    if args.role == "initiator":
        print(f"Target IP: {args.remote_ip} | Port: {args.port}")

    for size in args.sizes:
        start_agent_pair(size, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
