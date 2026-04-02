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


def _resolve_gpu_numa_node(local_gpu_idx: int, explicit_gpu_numa_node: Optional[int]):
    if explicit_gpu_numa_node is not None:
        return explicit_gpu_numa_node
    gpu_numa_node = _get_gpu_numa_node(local_gpu_idx)
    if gpu_numa_node < 0:
        raise RuntimeError(
            "Auto-detected GPU NUMA node is -1; pass --gpu-numa-node explicitly"
        )
    return gpu_numa_node


def _resolve_local_world_size() -> int:
    if "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", "1"))


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


def _allocate_cpu_slots_on_numa(
    size_per_block: int,
    gpu_idx: int,
    num_iovs: int,
    pipeline_depth: int,
    numa_node: int,
    restore_numa_node: int,
):
    _bind_process_to_numa(numa_node)
    try:
        return [
            _make_tensor_list(
                size_per_block,
                "cpu",
                gpu_idx,
                num_iovs,
                pinned=True,
            )
            for _ in range(pipeline_depth)
        ]
    finally:
        _bind_process_to_numa(restore_numa_node)


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


def _path_label(path, gpu_numa_node: int) -> str:
    if path["staging_numa"] == gpu_numa_node:
        return "local"
    return "remote"


def _connect_many(ep, remote_metadata: bytes, num_connections: int):
    remote_ip, remote_port, remote_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    conn_ids = []
    for conn_idx in range(num_connections):
        ok, conn_id = ep.connect(remote_ip, remote_gpu, remote_port=remote_port)
        assert ok, f"[Client] Failed to connect conn {conn_idx}"
        conn_ids.append(conn_id)
    print(
        f"[Client] Connected to {remote_ip}:{remote_port} (GPU {remote_gpu}) "
        f"with {len(conn_ids)} connections",
        flush=True,
    )
    return conn_ids


def _accept_many(ep, num_connections: int):
    conn_ids = []
    remote_ip = None
    remote_gpu = None
    for conn_idx in range(num_connections):
        ok, accepted_ip, accepted_gpu, conn_id = ep.accept()
        assert ok, f"[Server] Failed to accept conn {conn_idx}"
        conn_ids.append(conn_id)
        remote_ip = accepted_ip
        remote_gpu = accepted_gpu
    print(
        f"[Server] Accept from {remote_ip} (GPU {remote_gpu}) "
        f"with {len(conn_ids)} connections",
        flush=True,
    )
    return conn_ids


def _build_sender_paths(args, ep, conn_ids, size: int):
    size_per_block = size // args.num_iovs
    sender_paths = []
    for path_idx, staging_numa in enumerate(args.staging_numa_nodes):
        sender_gpu_slots = [
            _make_tensor_list(
                size_per_block,
                args.sender_device,
                args.local_gpu_idx,
                args.num_iovs,
            )
            for _ in range(args.pipeline_depth)
        ]
        sender_cpu_slots = _allocate_cpu_slots_on_numa(
            size_per_block,
            args.local_gpu_idx,
            args.num_iovs,
            args.pipeline_depth,
            staging_numa,
            args.gpu_numa_node,
        )
        sender_mr_ids, sender_ptrs, sender_sizes = _register_slot_group(
            ep, sender_cpu_slots, size_per_block
        )
        sender_paths.append(
            {
                "path_idx": path_idx,
                "staging_numa": staging_numa,
                "conn_id": conn_ids[path_idx],
                "gpu_slots": sender_gpu_slots,
                "cpu_slots": sender_cpu_slots,
                "mr_ids": sender_mr_ids,
                "ptrs": sender_ptrs,
                "sizes": sender_sizes,
                "copy_streams": [
                    torch.cuda.Stream(device=args.local_gpu_idx)
                    for _ in range(args.sender_copy_streams)
                ],
                "copy_events": [
                    torch.cuda.Event(enable_timing=False)
                    for _ in range(args.pipeline_depth)
                ],
            }
        )
    return sender_paths


def _build_receiver_paths(args, ep, conn_ids, size: int):
    size_per_block = size // args.num_iovs
    receiver_paths = []
    for path_idx, staging_numa in enumerate(args.staging_numa_nodes):
        receiver_cpu_slots = _allocate_cpu_slots_on_numa(
            size_per_block,
            args.local_gpu_idx,
            args.num_iovs,
            args.pipeline_depth,
            staging_numa,
            args.gpu_numa_node,
        )
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
        receiver_paths.append(
            {
                "path_idx": path_idx,
                "staging_numa": staging_numa,
                "conn_id": conn_ids[path_idx],
                "cpu_slots": receiver_cpu_slots,
                "gpu_slots": receiver_gpu_slots,
                "mr_ids": receiver_mr_ids,
                "ptrs": receiver_ptrs,
                "sizes": receiver_sizes,
                "copy_streams": [
                    torch.cuda.Stream(device=args.local_gpu_idx)
                    for _ in range(args.receiver_copy_streams)
                ],
                "copy_events": [
                    torch.cuda.Event(enable_timing=False)
                    for _ in range(args.pipeline_depth)
                ],
            }
        )
    return receiver_paths


def _cleanup_paths(ep, paths):
    for path in paths:
        for slot_mr_ids in path["mr_ids"]:
            for mr_id in slot_mr_ids:
                ep.dereg(mr_id)


def _run_sender_pipelines(args, ep, paths, num_iters: int):
    if num_iters <= 0:
        return

    send_transfer_ids = {
        path["path_idx"]: [None] * args.pipeline_depth for path in paths
    }

    for step in range(num_iters + 1):
        if step < num_iters:
            slot_idx = step % args.pipeline_depth
            for path in paths:
                if send_transfer_ids[path["path_idx"]][slot_idx] is not None:
                    _wait_transfer(ep, send_transfer_ids[path["path_idx"]][slot_idx])
                    send_transfer_ids[path["path_idx"]][slot_idx] = None
                _launch_copy(
                    path["copy_streams"],
                    path["copy_events"],
                    path["gpu_slots"],
                    path["cpu_slots"],
                    slot_idx,
                )

        if step > 0:
            slot_idx = (step - 1) % args.pipeline_depth
            for path in paths:
                path["copy_events"][slot_idx].synchronize()
                send_transfer_ids[path["path_idx"]][slot_idx] = _post_send(
                    ep,
                    path["conn_id"],
                    path["mr_ids"][slot_idx],
                    path["ptrs"][slot_idx],
                    path["sizes"][slot_idx],
                )

    for path in paths:
        for transfer_id in send_transfer_ids[path["path_idx"]]:
            if transfer_id is not None:
                _wait_transfer(ep, transfer_id)


def _run_receiver_pipelines(args, ep, paths, num_iters: int):
    if num_iters <= 0:
        return

    recv_transfer_ids = {
        path["path_idx"]: [None] * args.pipeline_depth for path in paths
    }
    recv_copy_pending = {
        path["path_idx"]: [False] * args.pipeline_depth for path in paths
    }

    for step in range(num_iters + 1):
        if step < num_iters:
            slot_idx = step % args.pipeline_depth
            for path in paths:
                if recv_copy_pending[path["path_idx"]][slot_idx]:
                    path["copy_events"][slot_idx].synchronize()
                    recv_copy_pending[path["path_idx"]][slot_idx] = False
                recv_transfer_ids[path["path_idx"]][slot_idx] = _post_recv(
                    ep,
                    path["conn_id"],
                    path["mr_ids"][slot_idx],
                    path["ptrs"][slot_idx],
                    path["sizes"][slot_idx],
                )

        if step > 0:
            slot_idx = (step - 1) % args.pipeline_depth
            for path in paths:
                _wait_transfer(ep, recv_transfer_ids[path["path_idx"]][slot_idx])
                recv_transfer_ids[path["path_idx"]][slot_idx] = None
                _launch_copy(
                    path["copy_streams"],
                    path["copy_events"],
                    path["cpu_slots"],
                    path["gpu_slots"],
                    slot_idx,
                )
                recv_copy_pending[path["path_idx"]][slot_idx] = True

    for path in paths:
        for slot_idx in range(args.pipeline_depth):
            if recv_copy_pending[path["path_idx"]][slot_idx]:
                path["copy_events"][slot_idx].synchronize()


def _run_server(args, ep, remote_metadata):
    conn_ids = _accept_many(ep, len(args.staging_numa_nodes))

    for size in args.sizes:
        receiver_paths = _build_receiver_paths(args, ep, conn_ids, size)

        _run_receiver_pipelines(args, ep, receiver_paths, args.warmup_iters)
        torch.cuda.synchronize(device=args.local_gpu_idx)
        dist.barrier()

        _run_receiver_pipelines(args, ep, receiver_paths, args.iters)
        torch.cuda.synchronize(device=args.local_gpu_idx)
        dist.barrier()

        _cleanup_paths(ep, receiver_paths)


def _run_client(args, ep, remote_metadata):
    conn_ids = _connect_many(ep, remote_metadata, len(args.staging_numa_nodes))
    elapsed_by_size: Dict[int, float] = {}

    for size in args.sizes:
        sender_paths = _build_sender_paths(args, ep, conn_ids, size)

        _run_sender_pipelines(args, ep, sender_paths, args.warmup_iters)
        torch.cuda.synchronize(device=args.local_gpu_idx)
        dist.barrier()

        start = time.perf_counter()
        _run_sender_pipelines(args, ep, sender_paths, args.iters)
        torch.cuda.synchronize(device=args.local_gpu_idx)
        elapsed = time.perf_counter() - start
        dist.barrier()

        _cleanup_paths(ep, sender_paths)
        elapsed_by_size[size] = elapsed

    return elapsed_by_size


def _print_summary(args, size: int, elapsed: float):
    total_bytes_per_path = size * args.iters
    aggregate_bytes = total_bytes_per_path * len(args.staging_numa_nodes)
    aggregate_gbps = (aggregate_bytes * 8) / elapsed / 1e9
    aggregate_gb_sec = aggregate_bytes / elapsed / 1e9

    print(f"[Summary] size={_pretty_size(size)}", flush=True)
    for path_idx, staging_numa in enumerate(args.staging_numa_nodes):
        label = "local" if staging_numa == args.gpu_numa_node else "remote"
        gbps = (total_bytes_per_path * 8) / elapsed / 1e9
        gb_sec = total_bytes_per_path / elapsed / 1e9
        print(
            f"  path {path_idx}: gpu{args.local_gpu_idx} numa{args.gpu_numa_node} -> "
            f"staging numa{staging_numa} ({label}) : "
            f"{gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s",
            flush=True,
        )
    print(
        f"  aggregate: {aggregate_gbps:6.2f} Gbps | {aggregate_gb_sec:6.2f} GB/s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        "UCCL staged send/recv benchmark with one GPU and multi-NUMA host staging paths"
    )
    parser.add_argument("--local-gpu-idx", type=int, default=0)
    parser.add_argument("--gpu-numa-node", type=int, default=None)
    parser.add_argument("--staging-numa-nodes", type=_parse_int_csv, default=[0, 1])
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
    if not args.staging_numa_nodes:
        raise ValueError("--staging-numa-nodes must not be empty")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_uccl_staged_singlegpu_dualnuma.py")

    args.gpu_numa_node = _resolve_gpu_numa_node(
        args.local_gpu_idx, args.gpu_numa_node
    )

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = _resolve_local_world_size()
    if world_size != 2:
        raise ValueError(
            "This benchmark expects exactly 1 process per node and 2 processes total; "
            f"got world_size={world_size}, local_world_size={local_world_size}. "
            "For this script, launch with --nproc_per_node=1."
        )

    # Keep endpoint/proxy threads tied to the GPU-local NUMA node.
    _bind_process_to_numa(args.gpu_numa_node)
    torch.cuda.set_device(f"cuda:{args.local_gpu_idx}")

    print(
        f"[Rank {rank}] gpu{args.local_gpu_idx} on numa{args.gpu_numa_node}, "
        f"staging paths={args.staging_numa_nodes}",
        flush=True,
    )

    ep = p2p.Endpoint(args.local_gpu_idx)
    local_metadata = ep.get_metadata()
    gathered_metadata: List[Optional[bytes]] = [None] * world_size
    dist.all_gather_object(gathered_metadata, local_metadata)
    remote_metadata = gathered_metadata[1 - rank]
    assert remote_metadata is not None

    if rank == 0:
        socket_ifname = os.environ.get("MSCCLPP_SOCKET_IFNAME", "<auto>")
        hca_devices = os.environ.get("MSCCLPP_HCA_DEVICES", "<auto>")
        print("UCCL Staged send/recv Benchmark (single GPU, dual NUMA staging)")
        print("=" * 72)
        print("Topology: 2 nodes x 1 GPU per node x multiple host staging NUMA paths")
        print(f"GPU: {args.local_gpu_idx}")
        print(f"GPU NUMA node: {args.gpu_numa_node}")
        print(f"Host staging NUMA nodes: {args.staging_numa_nodes}")
        print("Paths:")
        for path_idx, staging_numa in enumerate(args.staging_numa_nodes):
            label = "local" if staging_numa == args.gpu_numa_node else "remote"
            print(
                f"  path {path_idx}: gpu{args.local_gpu_idx} -> staging numa{staging_numa} ({label}) -> NIC"
            )
        print("Message sizes:", ", ".join(_pretty_size(size) for size in args.sizes))
        print(f"Iterations: {args.iters}")
        print(f"Warmup iterations: {args.warmup_iters}")
        print(f"Pipeline depth: {args.pipeline_depth}")
        print(f"Number of IOVs: {args.num_iovs}")
        print(f"Sender copy streams: {args.sender_copy_streams}")
        print(f"Receiver copy streams: {args.receiver_copy_streams}")
        print(f"MSCCLPP_SOCKET_IFNAME: {socket_ifname}")
        print(f"MSCCLPP_HCA_DEVICES: {hca_devices}")
        print("=" * 72, flush=True)

    dist.barrier()

    elapsed_by_size: Dict[int, float] = {}
    if rank == 0:
        elapsed_by_size = _run_client(args, ep, remote_metadata)
    else:
        _run_server(args, ep, remote_metadata)

    for size in args.sizes:
        elapsed_tensor = torch.tensor(
            [elapsed_by_size.get(size, 0.0)],
            dtype=torch.float64,
        )
        dist.broadcast(elapsed_tensor, src=0)
        if rank == 0:
            _print_summary(args, size, float(elapsed_tensor.item()))

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
