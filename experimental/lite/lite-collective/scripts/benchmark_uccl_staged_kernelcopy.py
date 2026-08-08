from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

try:
    from uccl import p2p
except ImportError:
    sys.stderr.write("Failed to import p2p\n")
    raise


_KERNEL_COPY_EXT = None


def _load_kernel_copy_extension():
    global _KERNEL_COPY_EXT
    if _KERNEL_COPY_EXT is not None:
        return _KERNEL_COPY_EXT

    build_dir = Path(__file__).resolve().parent.parent / ".tmp" / "torch_extensions"
    build_dir.mkdir(parents=True, exist_ok=True)

    cpp_source = r"""
    #include <torch/extension.h>

    torch::Tensor alloc_mapped_host_tensor(int64_t nbytes);
    void launch_copy(torch::Tensor src, torch::Tensor dst);

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("alloc_mapped_host_tensor", &alloc_mapped_host_tensor,
            "Allocate CUDA-mapped pinned host tensor");
      m.def("launch_copy", &launch_copy,
            "Launch byte-copy kernel between GPU tensor and mapped host tensor");
    }
    """

    cuda_source = r"""
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDAException.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <algorithm>
    #include <cstdint>

    namespace {

    __global__ void byte_copy_kernel(const uint8_t* src, uint8_t* dst, size_t nbytes) {
      size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
      size_t stride = blockDim.x * gridDim.x;
      for (size_t i = idx; i < nbytes; i += stride) {
        dst[i] = src[i];
      }
    }

    void* get_device_alias_for_host_ptr(void* host_ptr) {
      void* device_ptr = nullptr;
      auto err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
      TORCH_CHECK(
          err == cudaSuccess,
          "cudaHostGetDevicePointer failed for mapped host tensor: ",
          cudaGetErrorString(err),
          ". This benchmark requires CUDA-mapped pinned host memory."
      );
      return device_ptr;
    }

    }  // namespace

    torch::Tensor alloc_mapped_host_tensor(int64_t nbytes) {
      TORCH_CHECK(nbytes > 0, "nbytes must be > 0");
      void* host_ptr = nullptr;
      C10_CUDA_CHECK(cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped));

      auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8);
      auto deleter = [](void* ptr) {
        if (ptr != nullptr) {
          cudaFreeHost(ptr);
        }
      };
      return torch::from_blob(host_ptr, {nbytes}, deleter, options);
    }

    void launch_copy(torch::Tensor src, torch::Tensor dst) {
      TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
      TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
      TORCH_CHECK(src.scalar_type() == torch::kUInt8, "src must be uint8");
      TORCH_CHECK(dst.scalar_type() == torch::kUInt8, "dst must be uint8");
      TORCH_CHECK(src.numel() == dst.numel(), "src/dst size mismatch");
      TORCH_CHECK(src.is_cuda() != dst.is_cuda(),
                  "exactly one of src/dst must be CUDA and the other must be mapped host");

      const auto nbytes = static_cast<size_t>(src.numel());
      if (nbytes == 0) {
        return;
      }

      const uint8_t* src_ptr = nullptr;
      uint8_t* dst_ptr = nullptr;
      int device_idx = -1;

      if (src.is_cuda()) {
        device_idx = src.get_device();
        src_ptr = reinterpret_cast<const uint8_t*>(src.data_ptr());
        dst_ptr = reinterpret_cast<uint8_t*>(get_device_alias_for_host_ptr(dst.data_ptr()));
      } else {
        device_idx = dst.get_device();
        src_ptr = reinterpret_cast<const uint8_t*>(get_device_alias_for_host_ptr(src.data_ptr()));
        dst_ptr = reinterpret_cast<uint8_t*>(dst.data_ptr());
      }

      auto stream = at::cuda::getCurrentCUDAStream(device_idx);
      constexpr int threads = 256;
      int blocks = static_cast<int>((nbytes + threads - 1) / threads);
      blocks = std::max(1, std::min(blocks, 4096));
      byte_copy_kernel<<<blocks, threads, 0, stream>>>(
          src_ptr,
          dst_ptr,
          nbytes
      );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    """

    _KERNEL_COPY_EXT = load_inline(
        name="uccl_staged_kernelcopy_ext",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=None,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=False,
    )
    return _KERNEL_COPY_EXT


def _send_bytes_dist(payload: bytes, dst: int):
    n = len(payload)
    dist.send(torch.tensor([n], dtype=torch.int64), dst=dst)
    if n == 0:
        return
    buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    dist.send(buf, dst=dst)


def _recv_bytes_dist(src: int) -> bytes:
    n_tensor = torch.empty(1, dtype=torch.int64)
    dist.recv(n_tensor, src=src)
    n = int(n_tensor.item())
    if n == 0:
        return b""
    buf = torch.empty(n, dtype=torch.uint8)
    dist.recv(buf, src=src)
    return buf.numpy().tobytes()


def _make_tensor(size_bytes: int, device: str, gpu_idx: int, pinned: bool = False):
    assert size_bytes > 0, "size_bytes must be > 0"
    if device == "gpu":
        return torch.ones(size_bytes, dtype=torch.uint8, device=f"cuda:{gpu_idx}")
    if device == "cpu" and pinned:
        return _load_kernel_copy_extension().alloc_mapped_host_tensor(size_bytes)
    return torch.ones(size_bytes, dtype=torch.uint8)


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
    kernel_copy = _load_kernel_copy_extension()
    with torch.cuda.stream(copy_stream):
        for src, dst in zip(src_slots[slot_idx], dst_slots[slot_idx]):
            kernel_copy.launch_copy(src, dst)
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
    assert ok, "[Server] Failed to accept RDMA connection"
    print(f"[Server] Accept from {remote_ip} (GPU {remote_gpu}) conn_id={conn_id}")

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

    print("[Server] Benchmark complete")


def _run_client(args, ep, remote_metadata):
    remote_ip, remote_port, remote_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    ok, conn_id = ep.connect(remote_ip, remote_gpu, remote_port=remote_port)
    assert ok, "[Client] Failed to connect to server"
    print(
        f"[Client] Connected to {remote_ip}:{remote_port} (GPU {remote_gpu}) conn_id={conn_id}"
    )

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

        total = size * args.iters
        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        lat = elapsed / args.iters if args.iters > 0 else 0.0
        print(
            f"[Client] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | "
            f"{gb_sec:6.2f} GB/s | {lat:6.6f} s"
        )

    print("[Client] Benchmark complete")


def parse_sizes(v: str) -> List[int]:
    try:
        return [int(x) for x in v.split(",") if x]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bad --sizes") from exc


def main():
    p = argparse.ArgumentParser(
        "UCCL send/recv benchmark with GPU->CPU->RDMA->CPU->GPU staging pipeline using a CUDA copy kernel"
    )
    p.add_argument("--local-gpu-idx", type=int, default=0)
    p.add_argument("--intra", action="store_true")
    p.add_argument("--sender-device", choices=["gpu"], default="gpu")
    p.add_argument("--receiver-device", choices=["gpu"], default="gpu")
    p.add_argument(
        "--sizes",
        type=parse_sizes,
        default=[4096, 65536, 1048576, 10485760, 67108864, 104857600],
        help="Comma-separated list of message sizes in bytes",
    )
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup-iters", type=int, default=8)
    p.add_argument("--pipeline-depth", type=int, default=4)
    p.add_argument("--num-iovs", type=int, default=1)
    p.add_argument("--sender-copy-streams", type=int, default=2)
    p.add_argument("--receiver-copy-streams", type=int, default=2)
    args = p.parse_args()

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
        raise RuntimeError("CUDA is required for benchmark_uccl_staged_kernelcopy.py")

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"
    if args.intra:
        args.local_gpu_idx = rank
    torch.cuda.set_device(f"cuda:{args.local_gpu_idx}")

    # Build the extension once early so later timed loops only pay launch cost.
    _load_kernel_copy_extension()

    print("UCCL Staged send/recv Benchmark (kernel copy)")
    print("=" * 60)
    print("Path: GPU -> CPU(mapped pinned, kernel copy) -> send/recv -> CPU(mapped pinned, kernel copy) -> GPU")
    print("Copy engine: custom CUDA kernel with dst[i] = src[i]")
    print("Topology:", "intra-node" if args.intra else "inter-node")
    print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    print(f"Iterations: {args.iters}")
    print(f"Warmup iterations: {args.warmup_iters}")
    print(f"Pipeline depth: {args.pipeline_depth}")
    print(f"Number of IOVs: {args.num_iovs}")
    print(f"Sender copy streams: {args.sender_copy_streams}")
    print(f"Receiver copy streams: {args.receiver_copy_streams}")
    print(f"Role: {'client' if rank == 0 else 'server'}")
    print("=" * 60)

    ep = p2p.Endpoint(args.local_gpu_idx)
    local_metadata = ep.get_metadata()
    if rank == 0:
        _send_bytes_dist(local_metadata, dst=1)
        remote_metadata = _recv_bytes_dist(src=1)
    else:
        remote_metadata = _recv_bytes_dist(src=0)
        _send_bytes_dist(local_metadata, dst=0)

    if rank == 0:
        _run_client(args, ep, remote_metadata)
    else:
        _run_server(args, ep, remote_metadata)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
