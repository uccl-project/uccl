from __future__ import annotations

import argparse
import sys
import time
from typing import List
import traceback

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


def create_datasets(role, sizes, device, gpu_idx=0):
    """
    Create a dataset of tensors whose total size is at least size in bytes.
    """
    datasets = []
    for size in sizes:
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
            block = torch.full(
                (n_elems_per_block,), value, device=dev, dtype=dtype
            )
            dataset.append(block)

        # If total size is less than requested, add more elements to the last block
        total_bytes = sum(t.numel() * t.element_size() for t in dataset)
        if total_bytes < size:
            extra_elems = (size - total_bytes) // element_size
            if extra_elems > 0:
                extra_block = torch.full(
                    (extra_elems,), value, device=device, dtype=dtype
                )
                dataset.append(extra_block)
        datasets.append(dataset)
    return datasets


def initialize_xfer_metadata(
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
    register_descs = [desc.trim() for desc in register_descs]
    remote_xfer_descs = None
    transfer_handles = []

    if "server" in role:
        # Wait until there is a message from the creator
        while not agent.check_remote_metadata("client"):
            continue

        # send the xfer descs to the peer
        descs = agent.get_serialized_descs(register_descs)
        agent.send_notif("client", descs)

        transfer_handles = [None] * len(register_descs)

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
        transfer_handles = [
            agent.initialize_xfer(
                operation,
                register_descs[i],
                remote_xfer_descs[i],
                "server",
                f"TRANSFER_{i}".encode("utf-8"),
            )
            for i in range(len(register_descs))
        ]

    return transfer_handles


def create_nixl_agent(role: str, datasets):
    """
    Create Nixl agents based on the role.
    """
    port = 9000
    listen_port = port if role == "server" else 0
    config = nixl_agent_config(True, True, listen_port)
    agent = nixl_agent(role, config)
    register_descs = []
    for dataset in datasets:
        descs = agent.get_reg_descs(dataset)
        register_descs.append(agent.register_memory(descs))
    return agent, register_descs


def start_transfer(
    role: str,
    agent: nixl_agent,
    transfer_handle,
    uid,
):
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
        while not agent.check_remote_xfer_done("client", uid):
            continue


def cleanup_transfer(
    agent: nixl_agent,
    transfer_handles,
    register_descs,
):
    # Cleanup the transfer handle and registered descriptors
    for transfer_handle in transfer_handles:
        if transfer_handle is not None:
            agent.release_xfer_handle(transfer_handle)
    for register_desc in register_descs:
        agent.deregister_memory(register_desc)


def cleanup_agent(
    agent: nixl_agent,
):
    agent.remove_remote_agent(agent.name)


def start_agent_pair(sizes, args):
    op = "WRITE"
    port = 9000

    datasets = create_datasets(
        args.role, sizes, args.device, args.local_gpu_idx
    )

    agent, register_descs = create_nixl_agent(args.role, datasets)

    transfer_handles = initialize_xfer_metadata(
        args.role, op, agent, register_descs, args.remote_ip, port
    )

    try:
        for i, size in enumerate(sizes):
            total_size = 0
            start = time.perf_counter()
            for _ in range(args.iters):
                start_transfer(
                    args.role,
                    agent,
                    transfer_handles[i],
                    f"TRANSFER_{i}".encode("utf-8"),
                )
                total_size += size
            end = time.perf_counter()
            transfer_time = end - start
            gbps = (
                (total_size * 8) / transfer_time / 1e9
            )  # bits per second → Gbps
            gb_sec = total_size / transfer_time / 1e9  # bytes per second → GB/s
            lat = transfer_time / args.iters
            print(
                f"[{args.role}] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s | {lat:6.6f} s"
            )
            if "server" in args.role:
                for i, block in enumerate(datasets[i]):
                    assert (
                        torch.mean(block) - 1 < 1e-8
                    ), f"Block {i} not equal to 1"

    except KeyboardInterrupt:
        return 0.0
    except Exception as e:
        print(f"Error in agent pair {args.role}: {traceback.format_exc()}")
        return 0.0
    finally:
        cleanup_transfer(
            agent,
            transfer_handles,
            register_descs,
        )
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
        raise argparse.ArgumentTypeError(
            "sizes must be comma-separated integers"
        )


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
    args = p.parse_args()

    print("NIXL P2P Benchmark — role:", args.role)
    print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    print(
        f"Device: {args.device} | Local GPU idx: {args.local_gpu_idx} | Iterations: {args.iters}"
    )
    start_agent_pair(args.sizes, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
