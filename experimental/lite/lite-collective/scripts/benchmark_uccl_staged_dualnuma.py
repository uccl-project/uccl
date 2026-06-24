from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

try:
    from uccl import p2p
except ImportError:
    sys.stderr.write("Failed to import p2p\n")
    raise


class _NumaBitmask(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_ulong),
        ("maskp", ctypes.POINTER(ctypes.c_ulong)),
    ]


def _load_libnuma():
    libnuma_path = ctypes.util.find_library("numa")
    if not libnuma_path:
        raise RuntimeError("libnuma not found; cannot enforce NUMA-local staging")

    libnuma = ctypes.CDLL(libnuma_path)
    libnuma.numa_available.restype = ctypes.c_int
    libnuma.numa_num_configured_nodes.restype = ctypes.c_int
    libnuma.numa_allocate_nodemask.restype = ctypes.POINTER(_NumaBitmask)
    libnuma.numa_bitmask_setbit.argtypes = [ctypes.POINTER(_NumaBitmask), ctypes.c_uint]
    libnuma.numa_bind.argtypes = [ctypes.POINTER(_NumaBitmask)]
    libnuma.numa_run_on_node.argtypes = [ctypes.c_int]
    libnuma.numa_run_on_node.restype = ctypes.c_int
    libnuma.numa_set_strict.argtypes = [ctypes.c_int]
    free_bitmask = getattr(libnuma, "numa_bitmask_free", None)
    if free_bitmask is None:
        free_bitmask = getattr(libnuma, "numa_free_nodemask", None)
    if free_bitmask is not None:
        free_bitmask.argtypes = [ctypes.POINTER(_NumaBitmask)]
    libnuma._free_bitmask = free_bitmask
    return libnuma


def _bind_process_to_numa(node: int):
    libnuma = _load_libnuma()
    if libnuma.numa_available() < 0:
        raise RuntimeError("NUMA APIs are unavailable on this host")

    total_nodes = libnuma.numa_num_configured_nodes()
    if node < 0 or node >= total_nodes:
        raise ValueError(
            f"Invalid NUMA node {node}; host reports {total_nodes} configured nodes"
        )

    mask = libnuma.numa_allocate_nodemask()
    if not mask:
        raise RuntimeError("numa_allocate_nodemask failed")
    try:
        libnuma.numa_set_strict(1)
        libnuma.numa_bitmask_setbit(mask, node)
        rc = libnuma.numa_run_on_node(node)
        if rc != 0:
            raise OSError(f"numa_run_on_node({node}) failed with rc={rc}")
        libnuma.numa_bind(mask)
    finally:
        if libnuma._free_bitmask is not None:
            libnuma._free_bitmask(mask)


def _get_gpu_bus_id(device_idx: int) -> str:
    cudart_path = ctypes.util.find_library("cudart")
    if not cudart_path:
        raise RuntimeError("libcudart not found; cannot auto-detect GPU NUMA node")

    cudart = ctypes.CDLL(cudart_path)
    cudart.cudaDeviceGetPCIBusId.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    cudart.cudaDeviceGetPCIBusId.restype = ctypes.c_int

    buf = ctypes.create_string_buffer(32)
    rc = cudart.cudaDeviceGetPCIBusId(buf, len(buf), device_idx)
    if rc != 0:
        raise RuntimeError(
            f"cudaDeviceGetPCIBusId failed for GPU {device_idx} with rc={rc}"
        )
    return buf.value.decode("utf-8").lower()


def _get_gpu_numa_node(device_idx: int) -> int:
    bus_id = _get_gpu_bus_id(device_idx)
    path = os.path.join("/sys/bus/pci/devices", bus_id, "numa_node")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except OSError as exc:
        raise RuntimeError(
            f"Failed to read GPU {device_idx} NUMA node from {path}"
        ) from exc


def _parse_int_csv(value: str) -> List[int]:
    try:
        parsed = [int(x) for x in value.split(",") if x]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad integer csv: {value}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("integer csv must not be empty")
    return parsed


def _resolve_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))


def _resolve_local_world_size(default_value: int) -> int:
    if "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", str(default_value)))


def _resolve_numa_nodes(local_gpus: List[int], explicit_numa_nodes: Optional[List[int]]):
    if explicit_numa_nodes is not None:
        if len(explicit_numa_nodes) != len(local_gpus):
            raise ValueError("--numa-nodes must have the same length as --local-gpus")
        return explicit_numa_nodes

    numa_nodes = [_get_gpu_numa_node(gpu_idx) for gpu_idx in local_gpus]
    if any(node < 0 for node in numa_nodes):
        raise RuntimeError(
            "Auto-detected at least one GPU NUMA node as -1; pass --numa-nodes explicitly"
        )
    return numa_nodes


def _make_tensor(size_bytes: int, device: str, gpu_idx: int, pinned: bool = False):
    assert size_bytes > 0, "size_bytes must be > 0"
    dtype = torch.float32 if size_bytes >= 4 and size_bytes % 4 == 0 else torch.uint8
    n_elems = size_bytes // torch.tensor([], dtype=dtype).element_size()
    if device == "gpu":
        return torch.ones(n_elems, dtype=dtype, device=f"cuda:{gpu_idx}")
    if device == "cpu" and pinned:
        return torch.ones(n_elems, dtype=dtype).pin_memory()
    return torch.ones(n_elems, dtype=dtype)


def _make_tensor_list(
    size_bytes: int, device: str, gpu_idx: int, count: int, pinned: bool = False
):
    return [_make_tensor(size_bytes, device, gpu_idx, pinned) for _ in range(count)]


def _pretty_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    val = float(num_bytes)
    for unit in units:
        if val < 1024 or unit == units[-1]:
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024
    return f"{num_bytes} B"


def _wait_transfer(ep, transfer_id: int):
    is_done = False
    while not is_done:
        ok, is_done = ep.poll_async(transfer_id)
        assert ok, f"poll_async failed for transfer {transfer_id}"


def _launch_copy(copy_streams, copy_events, src_slots, dst_slots, slot_idx: int):
    copy_stream = copy_streams[slot_idx % len(copy_streams)]
    with torch.cuda.stream(copy_stream):
        for src, dst in zip(src_slots[slot_idx], dst_slots[slot_idx]):
            dst.copy_(src, non_blocking=True)
        copy_events[slot_idx].record(copy_stream)


def _post_send(ep, conn_id: int, mr_ids, ptrs, sizes):
    if len(mr_ids) == 1:
        ok, transfer_id = ep.send_async(conn_id, mr_ids[0], ptrs[0], sizes[0])
    else:
        ok, transfer_id = ep.sendv_async(conn_id, mr_ids, ptrs, sizes, len(mr_ids))
    assert ok, "send_async/sendv_async failed"
    return transfer_id


def _post_recv(ep, conn_id: int, mr_ids, ptrs, sizes):
    if len(mr_ids) == 1:
        ok, transfer_id = ep.recv_async(conn_id, mr_ids[0], ptrs[0], sizes[0])
    else:
        ok, transfer_id = ep.recvv_async(conn_id, mr_ids, ptrs, sizes, len(mr_ids))
    assert ok, "recv_async/recvv_async failed"
    return transfer_id


def _register_slot_group(ep, tensor_slots, size_per_block: int):
    mr_id_slots = []
    ptr_slots = []
    size_slots = []
    for slot_tensors in tensor_slots:
        mr_ids = []
        ptrs = []
        sizes = []
        for tensor in slot_tensors:
            ptr = tensor.data_ptr()
            ok, mr_id = ep.reg(ptr, size_per_block)
            assert ok, "ep.reg failed"
            mr_ids.append(mr_id)
            ptrs.append(ptr)
            sizes.append(size_per_block)
        mr_id_slots.append(mr_ids)
        ptr_slots.append(ptrs)
        size_slots.append(sizes)
    return mr_id_slots, ptr_slots, size_slots


def _run_sender_pipeline(
    args,
    ep,
    conn_id: int,
    sender_gpu_slots,
    sender_cpu_slots,
    sender_mr_ids,
    sender_ptrs,
    sender_sizes,
    sender_copy_streams,
    sender_copy_events,
    num_iters: int,
):
    if num_iters <= 0:
        return

    send_transfer_ids = [None] * args.pipeline_depth

    for step in range(num_iters + 1):
        if step < num_iters:
            slot_idx = step % args.pipeline_depth
            if send_transfer_ids[slot_idx] is not None:
                _wait_transfer(ep, send_transfer_ids[slot_idx])
                send_transfer_ids[slot_idx] = None
            _launch_copy(
                sender_copy_streams,
                sender_copy_events,
                sender_gpu_slots,
                sender_cpu_slots,
                slot_idx,
            )

        if step > 0:
            slot_idx = (step - 1) % args.pipeline_depth
            sender_copy_events[slot_idx].synchronize()
            send_transfer_ids[slot_idx] = _post_send(
                ep,
                conn_id,
                sender_mr_ids[slot_idx],
                sender_ptrs[slot_idx],
                sender_sizes[slot_idx],
            )

    for transfer_id in send_transfer_ids:
        if transfer_id is not None:
            _wait_transfer(ep, transfer_id)


def _run_receiver_pipeline(
    args,
    ep,
    conn_id: int,
    receiver_cpu_slots,
    receiver_gpu_slots,
    receiver_mr_ids,
    receiver_ptrs,
    receiver_sizes,
    receiver_copy_streams,
    receiver_copy_events,
    num_iters: int,
):
    if num_iters <= 0:
        return

    recv_transfer_ids = [None] * args.pipeline_depth
    recv_copy_pending = [False] * args.pipeline_depth

    for step in range(num_iters + 1):
        if step < num_iters:
            slot_idx = step % args.pipeline_depth
            if recv_copy_pending[slot_idx]:
                receiver_copy_events[slot_idx].synchronize()
                recv_copy_pending[slot_idx] = False
            recv_transfer_ids[slot_idx] = _post_recv(
                ep,
                conn_id,
                receiver_mr_ids[slot_idx],
                receiver_ptrs[slot_idx],
                receiver_sizes[slot_idx],
            )

        if step > 0:
            slot_idx = (step - 1) % args.pipeline_depth
            _wait_transfer(ep, recv_transfer_ids[slot_idx])
            recv_transfer_ids[slot_idx] = None
            _launch_copy(
                receiver_copy_streams,
                receiver_copy_events,
                receiver_cpu_slots,
                receiver_gpu_slots,
                slot_idx,
            )
            recv_copy_pending[slot_idx] = True

    for slot_idx in range(args.pipeline_depth):
        if recv_copy_pending[slot_idx]:
            receiver_copy_events[slot_idx].synchronize()


def _run_server(args, ep, remote_metadata):
    ok, remote_ip, remote_gpu, conn_id = ep.accept()
    assert ok, f"[Rank {args.rank}] Failed to accept RDMA connection"
    print(
        f"[Rank {args.rank}] server accepted {remote_ip} (GPU {remote_gpu}) "
        f"conn_id={conn_id}",
        flush=True,
    )

    for size in args.sizes:
        size_per_block = size // args.num_iovs

        receiver_cpu_slots = [
            _make_tensor_list(
                size_per_block,
                "cpu",
                args.local_gpu_idx,
                args.num_iovs,
                pinned=True,
            )
            for _ in range(args.pipeline_depth)
        ]
        receiver_gpu_slots = [
            _make_tensor_list(
                size_per_block,
                args.receiver_device,
                args.local_gpu_idx,
                args.num_iovs,
            )
            for _ in range(args.pipeline_depth)
        ]
        receiver_mr_ids, receiver_ptrs, receiver_sizes = _register_slot_group(
            ep, receiver_cpu_slots, size_per_block
        )
        receiver_copy_streams = [
            torch.cuda.Stream(device=args.local_gpu_idx)
            for _ in range(args.receiver_copy_streams)
        ]
        receiver_copy_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(args.pipeline_depth)
        ]

        _run_receiver_pipeline(
            args,
            ep,
            conn_id,
            receiver_cpu_slots,
            receiver_gpu_slots,
            receiver_mr_ids,
            receiver_ptrs,
            receiver_sizes,
            receiver_copy_streams,
            receiver_copy_events,
            args.warmup_iters,
        )
        torch.cuda.synchronize(device=args.local_gpu_idx)
        dist.barrier()

        _run_receiver_pipeline(
            args,
            ep,
            conn_id,
            receiver_cpu_slots,
            receiver_gpu_slots,
            receiver_mr_ids,
            receiver_ptrs,
            receiver_sizes,
            receiver_copy_streams,
            receiver_copy_events,
            args.iters,
        )
        torch.cuda.synchronize(device=args.local_gpu_idx)
        dist.barrier()

        for slot_mr_ids in receiver_mr_ids:
            for mr_id in slot_mr_ids:
                ep.dereg(mr_id)


def _run_client(args, ep, remote_metadata) -> Dict[int, float]:
    remote_ip, remote_port, remote_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    ok, conn_id = ep.connect(remote_ip, remote_gpu, remote_port=remote_port)
    assert ok, f"[Rank {args.rank}] Failed to connect to server"
    print(
        f"[Rank {args.rank}] client connected to {remote_ip}:{remote_port} "
        f"(GPU {remote_gpu}) conn_id={conn_id}",
        flush=True,
    )

    elapsed_by_size: Dict[int, float] = {}

    for size in args.sizes:
        size_per_block = size // args.num_iovs

        sender_gpu_slots = [
            _make_tensor_list(
                size_per_block,
                args.sender_device,
                args.local_gpu_idx,
                args.num_iovs,
            )
            for _ in range(args.pipeline_depth)
        ]
        sender_cpu_slots = [
            _make_tensor_list(
                size_per_block,
                "cpu",
                args.local_gpu_idx,
                args.num_iovs,
                pinned=True,
            )
            for _ in range(args.pipeline_depth)
        ]
        sender_mr_ids, sender_ptrs, sender_sizes = _register_slot_group(
            ep, sender_cpu_slots, size_per_block
        )
        sender_copy_streams = [
            torch.cuda.Stream(device=args.local_gpu_idx)
            for _ in range(args.sender_copy_streams)
        ]
        sender_copy_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(args.pipeline_depth)
        ]

        _run_sender_pipeline(
            args,
            ep,
            conn_id,
            sender_gpu_slots,
            sender_cpu_slots,
            sender_mr_ids,
            sender_ptrs,
            sender_sizes,
            sender_copy_streams,
            sender_copy_events,
            args.warmup_iters,
        )
        torch.cuda.synchronize(device=args.local_gpu_idx)
        dist.barrier()

        start = time.perf_counter()
        _run_sender_pipeline(
            args,
            ep,
            conn_id,
            sender_gpu_slots,
            sender_cpu_slots,
            sender_mr_ids,
            sender_ptrs,
            sender_sizes,
            sender_copy_streams,
            sender_copy_events,
            args.iters,
        )
        torch.cuda.synchronize(device=args.local_gpu_idx)
        elapsed = time.perf_counter() - start
        dist.barrier()

        for slot_mr_ids in sender_mr_ids:
            for mr_id in slot_mr_ids:
                ep.dereg(mr_id)

        elapsed_by_size[size] = elapsed

    return elapsed_by_size


def _print_step_summary(rank: int, size: int, results: List[Dict[str, object]]):
    if rank != 0:
        return

    client_results = [
        result for result in results if result["role"] == "client" and result["size"] == size
    ]
    if not client_results:
        return

    client_results.sort(key=lambda item: int(item["rank"]))
    total_bytes = 0
    max_elapsed = 0.0
    sum_pair_gbps = 0.0

    print(f"[Summary] size={_pretty_size(size)}", flush=True)
    for result in client_results:
        elapsed = float(result["elapsed"])
        total = int(result["total_bytes"])
        total_bytes += total
        max_elapsed = max(max_elapsed, elapsed)
        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        sum_pair_gbps += gbps
        print(
            f"  pair rank {result['rank']} gpu{result['gpu']} numa{result['numa']} -> "
            f"peer rank {result['peer_rank']} gpu{result['peer_gpu']} numa{result['peer_numa']} : "
            f"{gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s | {elapsed / int(result['iters']):.6f} s/iter",
            flush=True,
        )

    aggregate_gbps = (total_bytes * 8) / max_elapsed / 1e9 if max_elapsed > 0 else 0.0
    aggregate_gb_sec = total_bytes / max_elapsed / 1e9 if max_elapsed > 0 else 0.0
    print(
        f"  aggregate effective: {aggregate_gbps:6.2f} Gbps | {aggregate_gb_sec:6.2f} GB/s "
        f"(max pair elapsed {max_elapsed:.6f} s)",
        flush=True,
    )
    print(
        f"  aggregate sum-of-pairs: {sum_pair_gbps:6.2f} Gbps",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        "UCCL staged send/recv benchmark for 2 GPUs on different NUMA nodes sharing one NIC"
    )
    parser.add_argument("--local-gpus", type=_parse_int_csv, default=[0, 3])
    parser.add_argument("--numa-nodes", type=_parse_int_csv, default=None)
    parser.add_argument("--sender-device", choices=["gpu"], default="gpu")
    parser.add_argument("--receiver-device", choices=["gpu"], default="gpu")
    parser.add_argument(
        "--sizes",
        type=_parse_int_csv,
        default=[4096, 65536, 1048576, 10485760, 67108864, 104857600],
        help="Comma-separated list of message sizes in bytes",
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=8)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--num-iovs", type=int, default=1)
    parser.add_argument("--sender-copy-streams", type=int, default=2)
    parser.add_argument("--receiver-copy-streams", type=int, default=2)
    args = parser.parse_args()

    if args.pipeline_depth < 1:
        raise ValueError("--pipeline-depth must be >= 1")
    if args.num_iovs < 1:
        raise ValueError("--num-iovs must be >= 1")
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be >= 0")
    if args.sender_copy_streams < 1:
        raise ValueError("--sender-copy-streams must be >= 1")
    if args.receiver_copy_streams < 1:
        raise ValueError("--receiver-copy-streams must be >= 1")
    if any(size <= 0 for size in args.sizes):
        raise ValueError("--sizes must all be > 0")
    if any(size % args.num_iovs != 0 for size in args.sizes):
        raise ValueError("each size in --sizes must be divisible by --num-iovs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_uccl_staged_dualnuma.py")

    args.numa_nodes = _resolve_numa_nodes(args.local_gpus, args.numa_nodes)

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = _resolve_local_rank()
    local_world_size = _resolve_local_world_size(len(args.local_gpus))

    if len(args.local_gpus) != len(args.numa_nodes):
        raise ValueError("--local-gpus and --numa-nodes must have the same length")
    if local_world_size != len(args.local_gpus):
        raise ValueError(
            "LOCAL_WORLD_SIZE must match the length of --local-gpus for this benchmark"
        )
    if world_size != 2 * local_world_size:
        raise ValueError(
            f"This benchmark expects exactly 2 nodes; got world_size={world_size}, "
            f"local_world_size={local_world_size}"
        )
    if local_rank < 0 or local_rank >= local_world_size:
        raise ValueError(
            f"Resolved invalid local rank {local_rank} for local world size {local_world_size}"
        )

    node_rank = rank // local_world_size
    if node_rank not in (0, 1):
        raise ValueError(f"Expected exactly 2 node groups; got node_rank={node_rank}")

    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    args.node_rank = node_rank
    args.local_gpu_idx = args.local_gpus[local_rank]
    args.local_numa_node = args.numa_nodes[local_rank]
    args.role = "client" if node_rank == 0 else "server"
    args.peer_rank = rank + local_world_size if node_rank == 0 else rank - local_world_size

    _bind_process_to_numa(args.local_numa_node)
    torch.cuda.set_device(f"cuda:{args.local_gpu_idx}")

    ep = p2p.Endpoint(args.local_gpu_idx)
    local_metadata = ep.get_metadata()
    all_metadata: List[Optional[bytes]] = [None] * world_size
    dist.all_gather_object(all_metadata, local_metadata)
    remote_metadata = all_metadata[args.peer_rank]
    assert remote_metadata is not None, f"Missing metadata for peer rank {args.peer_rank}"

    local_info = {
        "rank": rank,
        "local_rank": local_rank,
        "node_rank": node_rank,
        "role": args.role,
        "gpu": args.local_gpu_idx,
        "numa": args.local_numa_node,
        "peer_rank": args.peer_rank,
    }
    all_infos: List[Optional[Dict[str, int]]] = [None] * world_size
    dist.all_gather_object(all_infos, local_info)

    if rank == 0:
        socket_ifname = os.environ.get("MSCCLPP_SOCKET_IFNAME", "<auto>")
        hca_devices = os.environ.get("MSCCLPP_HCA_DEVICES", "<auto>")
        print("UCCL Staged send/recv Benchmark (dual NUMA, shared NIC)")
        print("=" * 72)
        print("Path: GPU -> CPU(pinned) -> send/recv -> CPU(pinned) -> GPU")
        print("Topology: 2 nodes x 2 ranks per node")
        print(f"Local GPUs per node: {args.local_gpus}")
        print(f"Local NUMA nodes per node: {args.numa_nodes}")
        print("Message sizes:", ", ".join(_pretty_size(size) for size in args.sizes))
        print(f"Iterations: {args.iters}")
        print(f"Warmup iterations: {args.warmup_iters}")
        print(f"Pipeline depth: {args.pipeline_depth}")
        print(f"Number of IOVs: {args.num_iovs}")
        print(f"Sender copy streams: {args.sender_copy_streams}")
        print(f"Receiver copy streams: {args.receiver_copy_streams}")
        print(f"MSCCLPP_SOCKET_IFNAME: {socket_ifname}")
        print(f"MSCCLPP_HCA_DEVICES: {hca_devices}")
        print("-" * 72)
        for info in sorted(all_infos, key=lambda item: int(item["rank"])):
            print(
                f"Rank {info['rank']}: node {info['node_rank']} | local_rank {info['local_rank']} | "
                f"{info['role']} | gpu{info['gpu']} | numa{info['numa']} | peer {info['peer_rank']}"
            )
        print("=" * 72, flush=True)

    dist.barrier()

    elapsed_by_size: Dict[int, float] = {}
    if args.role == "client":
        elapsed_by_size = _run_client(args, ep, remote_metadata)
    else:
        _run_server(args, ep, remote_metadata)

    for size in args.sizes:
        local_result = {
            "rank": rank,
            "peer_rank": args.peer_rank,
            "role": args.role,
            "gpu": args.local_gpu_idx,
            "numa": args.local_numa_node,
            "peer_gpu": all_infos[args.peer_rank]["gpu"],
            "peer_numa": all_infos[args.peer_rank]["numa"],
            "size": size,
            "iters": args.iters,
            "total_bytes": size * args.iters,
            "elapsed": elapsed_by_size.get(size, 0.0),
        }
        gathered_results: List[Optional[Dict[str, object]]] = [None] * world_size
        dist.all_gather_object(gathered_results, local_result)
        _print_step_summary(rank, size, gathered_results)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
