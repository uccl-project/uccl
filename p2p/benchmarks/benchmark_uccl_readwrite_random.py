"""UCCL READ/WRITE benchmark — random [-1, 1) bf16 payloads.

Same harness as benchmark_uccl_readwrite.py, but the payload is uniformly
random in [-1, 1) (bf16) so dietgpu's ANS encoder sees realistic entropy.
No correctness verification — pure throughput measurement.
"""
from __future__ import annotations
import argparse, sys, time
from typing import List
import torch.distributed as dist
import torch

try:
    from uccl import p2p
except ImportError:
    sys.stderr.write("Failed to import p2p\n")
    raise

fifo_blob_size = 64  # bytes


_raw_cudamalloc_keepalive: list = []  # keep cudaMalloc allocations alive


_cudart_lib = None


def _get_cudart():
    """Locate libcudart.so on this system. Different CUDA installs use
    different version suffixes (libcudart.so.12, .11, ...), so we try them
    in order and cache the result."""
    global _cudart_lib
    if _cudart_lib is not None:
        return _cudart_lib
    import ctypes
    candidates = [
        "libcudart.so.13", "libcudart.so.12.9", "libcudart.so.12.8",
        "libcudart.so.12.4", "libcudart.so.12.2", "libcudart.so.12.1",
        "libcudart.so.12.0", "libcudart.so.12", "libcudart.so.11.8",
        "libcudart.so.11.7", "libcudart.so.11", "libcudart.so",
    ]
    last_err = None
    for name in candidates:
        try:
            _cudart_lib = ctypes.CDLL(name)
            return _cudart_lib
        except OSError as e:
            last_err = e
    raise RuntimeError(
        f"Could not locate libcudart.so (tried {candidates}); last error: {last_err}"
    )


def _make_raw_cudamalloc_buffer(n_bytes: int, gpu: int):
    """Allocate `n_bytes` via raw cudaMalloc (bypassing torch's caching
    allocator), then fill with the same uniform-bf16-in-[-1,1) bytes that
    torch.empty(...).uniform_(...) produces. Returns (keepalive_obj, ptr).

    Used to isolate "torch caching allocator vs raw cudaMalloc" as a variable
    for NIC PCIe DMA throughput.
    """
    import ctypes
    # Make sure torch initialized CUDA context on the right device, otherwise
    # cudaMalloc has no current context.
    torch.cuda.set_device(gpu)
    _ = torch.empty(1, device=f"cuda:{gpu}")  # touch device to init context
    torch.cuda.synchronize()

    cudart = _get_cudart()
    # Set return types so ctypes doesn't truncate 64-bit values.
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemcpy.restype = ctypes.c_int
    cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    cudart.cudaSetDevice.restype = ctypes.c_int

    rc = cudart.cudaSetDevice(gpu)
    if rc != 0:
        raise RuntimeError(f"cudaSetDevice({gpu}) failed rc={rc}")

    dev_ptr = ctypes.c_void_p()
    rc = cudart.cudaMalloc(ctypes.byref(dev_ptr), n_bytes)
    if rc != 0:
        raise RuntimeError(f"cudaMalloc({n_bytes}) failed rc={rc}")

    # Generate the same payload as the torch path.
    if n_bytes % 2 == 0:
        host_buf = torch.empty(n_bytes // 2, dtype=torch.bfloat16,
                               pin_memory=True).uniform_(-1.0, 1.0)
    else:
        host_buf = torch.randint(0, 256, (n_bytes,), dtype=torch.uint8,
                                 pin_memory=True)
    rc = cudart.cudaMemcpy(dev_ptr.value, host_buf.data_ptr(), n_bytes,
                           1)  # 1 = cudaMemcpyHostToDevice
    if rc != 0:
        raise RuntimeError(f"cudaMemcpy failed rc={rc}")

    keepalive = (dev_ptr, host_buf)
    _raw_cudamalloc_keepalive.append(keepalive)
    print(f"[raw_cudaMalloc] allocated {n_bytes} B at 0x{dev_ptr.value:x}",
          flush=True)
    return keepalive, dev_ptr.value


def _make_buffer(n_bytes: int, device: str, gpu: int,
                 use_raw_cudamalloc: bool = False):
    """Allocate a buffer of n_bytes and fill with random [-1, 1) values.

    bf16 path used when n_bytes is even (the usual case); falls back to uint8
    random bytes for odd sizes.
    """
    if n_bytes <= 0:
        raise ValueError(f"buffer size must be positive, got {n_bytes}")
    if use_raw_cudamalloc and device == "gpu":
        return _make_raw_cudamalloc_buffer(n_bytes, gpu)
    if n_bytes % 2 == 0:
        n_elems = n_bytes // 2
        if device == "gpu":
            buf = torch.empty(n_elems, dtype=torch.bfloat16, device=f"cuda:{gpu}")
        else:
            buf = torch.empty(n_elems, dtype=torch.bfloat16, pin_memory=True)
        buf.uniform_(-1.0, 1.0)
    else:
        if device == "gpu":
            buf = torch.randint(
                0, 256, (n_bytes,), dtype=torch.uint8, device=f"cuda:{gpu}"
            )
        else:
            buf = torch.randint(0, 256, (n_bytes,), dtype=torch.uint8, pin_memory=True)
    return buf, buf.data_ptr()


def _pretty(num: int):
    units, val = ["B", "KB", "MB", "GB"], float(num)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024


def _run_server(args, ep, remote_metadata):
    peer = 0
    pre_registered = {}
    if args.lazy:
        print("[Server] Pre-registering all memory...")
        for sz in args.sizes:
            size_per_block = sz // args.num_iovs
            buf_v, ptr_v, mr_id_v, size_v = [], [], [], []
            for _ in range(args.num_iovs):
                buf, ptr = _make_buffer(size_per_block, args.device, args.local_gpu_idx, args.use_raw_cudamalloc)
                ok, mr_id = ep.reg(
                    ptr, size_per_block, floatType=p2p.FloatType.kBFloat16
                )
                assert ok
                buf_v.append(buf)
                ptr_v.append(ptr)
                mr_id_v.append(mr_id)
                size_v.append(size_per_block)
            pre_registered[sz] = {
                "buf_v": buf_v,
                "ptr_v": ptr_v,
                "mr_id_v": mr_id_v,
                "size_v": size_v,
            }
        print("[Server] All memory pre-registered")

    print("[Server] Waiting for connection …")
    ok, r_ip, r_gpu, conn_id = ep.accept()
    assert ok
    print(f"[Server] Connected to {r_ip} (GPU {r_gpu}) id={conn_id}")

    for sz in args.sizes:
        if args.lazy:
            buf_v = pre_registered[sz]["buf_v"]
            ptr_v = pre_registered[sz]["ptr_v"]
            mr_id_v = pre_registered[sz]["mr_id_v"]
            size_v = pre_registered[sz]["size_v"]
        else:
            size_per_block = sz // args.num_iovs
            buf_v, ptr_v, mr_id_v, size_v = [], [], [], []
            for _ in range(args.num_iovs):
                buf, ptr = _make_buffer(size_per_block, args.device, args.local_gpu_idx, args.use_raw_cudamalloc)
                ok, mr_id = ep.reg(
                    ptr, size_per_block, floatType=p2p.FloatType.kBFloat16
                )
                assert ok
                buf_v.append(buf)
                ptr_v.append(ptr)
                mr_id_v.append(mr_id)
                size_v.append(size_per_block)
        ok, fifo_blob_v = ep.advertisev(conn_id, mr_id_v, ptr_v, size_v, args.num_iovs)
        assert ok
        assert all(len(fifo_blob) == fifo_blob_size for fifo_blob in fifo_blob_v)
        for fifo_blob in fifo_blob_v:
            dist.send(torch.ByteTensor(list(fifo_blob)), dst=peer)
        dist.barrier()
    print("[Server] Benchmark complete")


def _run_client(args, ep, remote_metadata):
    peer = 1
    pre_registered = {}
    if args.lazy:
        print("[Client] Pre-registering all memory...")
        for sz in args.sizes:
            size_per_block = sz // args.num_iovs
            buf_v, ptr_v, mr_id_v, size_v = [], [], [], []
            for _ in range(args.num_iovs):
                buf, ptr = _make_buffer(size_per_block, args.device, args.local_gpu_idx, args.use_raw_cudamalloc)
                ok, mr_id = ep.reg(
                    ptr, size_per_block, floatType=p2p.FloatType.kBFloat16
                )
                assert ok
                buf_v.append(buf)
                ptr_v.append(ptr)
                mr_id_v.append(mr_id)
                size_v.append(size_per_block)
            pre_registered[sz] = {
                "buf_v": buf_v,
                "ptr_v": ptr_v,
                "mr_id_v": mr_id_v,
                "size_v": size_v,
            }
        print("[Client] All memory pre-registered")

    ip, port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok
    print(f"[Client] Connected to {ip}:{port} (GPU {r_gpu}) id={conn_id}")

    for sz in args.sizes:
        if args.lazy:
            buf_v = pre_registered[sz]["buf_v"]
            ptr_v = pre_registered[sz]["ptr_v"]
            mr_id_v = pre_registered[sz]["mr_id_v"]
            size_v = pre_registered[sz]["size_v"]
        else:
            size_per_block = sz // args.num_iovs
            buf_v, ptr_v, mr_id_v, size_v = [], [], [], []
            for _ in range(args.num_iovs):
                buf, ptr = _make_buffer(size_per_block, args.device, args.local_gpu_idx, args.use_raw_cudamalloc)
                ok, mr_id = ep.reg(
                    ptr, size_per_block, floatType=p2p.FloatType.kBFloat16
                )
                assert ok
                buf_v.append(buf)
                ptr_v.append(ptr)
                mr_id_v.append(mr_id)
                size_v.append(size_per_block)

        fifo_blob_v = []
        for _ in range(args.num_iovs):
            fifo_blob = torch.zeros(fifo_blob_size, dtype=torch.uint8)
            dist.recv(fifo_blob, src=peer)
            fifo_blob_v.append(bytes(fifo_blob.tolist()))

        # Warmup — repeat a few times so dietgpu JIT compile, NIC SQ warmup,
        # cudaMalloc page touch, and any first-iter overhead don't pollute the
        # timed loop. Default 10 iters; tunable via --warmup-iters.
        for _ in range(args.warmup_iters):
            if args.async_api:
                transfer_ids = []
                if args.num_iovs == 1:
                    if args.mode == "write":
                        ok, transfer_id = ep.write_async(
                            conn_id, mr_id_v[0], ptr_v[0], size_v[0], fifo_blob_v[0]
                        )
                    else:
                        ok, transfer_id = ep.read_async(
                            conn_id, mr_id_v[0], ptr_v[0], size_v[0], fifo_blob_v[0]
                        )
                    assert ok
                    transfer_ids.append(transfer_id)
                else:
                    if args.mode == "write":
                        ok, transfer_id = ep.writev_async(
                            conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, args.num_iovs
                        )
                    else:
                        ok, transfer_id = ep.readv_async(
                            conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, args.num_iovs
                        )
                    assert ok
                    transfer_ids.append(transfer_id)
                while transfer_ids:
                    remaining_ids = []
                    for transfer_id in transfer_ids:
                        ok, is_done = ep.poll_async(transfer_id)
                        assert ok
                        if not is_done:
                            remaining_ids.append(transfer_id)
                    transfer_ids = remaining_ids
            else:
                if args.mode == "write":
                    ep.writev(conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, args.num_iovs)
                else:
                    ep.readv(conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, args.num_iovs)

        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            if args.async_api:
                if args.num_iovs == 1:
                    if args.mode == "write":
                        ok, transfer_id = ep.write_async(
                            conn_id, mr_id_v[0], ptr_v[0], size_v[0], fifo_blob_v[0]
                        )
                    else:
                        ok, transfer_id = ep.read_async(
                            conn_id, mr_id_v[0], ptr_v[0], size_v[0], fifo_blob_v[0]
                        )
                    assert ok
                    is_done = False
                    while not is_done:
                        ok, is_done = ep.poll_async(transfer_id)
                        assert ok
                    total += size_v[0]
                else:
                    if args.mode == "write":
                        ok, transfer_id = ep.writev_async(
                            conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, args.num_iovs
                        )
                    else:
                        ok, transfer_id = ep.readv_async(
                            conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, args.num_iovs
                        )
                    assert ok
                    is_done = False
                    while not is_done:
                        ok, is_done = ep.poll_async(transfer_id)
                        assert ok
                    total += sum(size_v)
            else:
                if args.mode == "write":
                    ep.writev(
                        conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, args.num_iovs
                    )
                else:
                    ep.readv(
                        conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, args.num_iovs
                    )
                total += sum(size_v)
        elapsed = time.perf_counter() - start
        print(
            f"[Client] {_pretty(sz):>8} : "
            f"{(total*8)/elapsed/1e9:6.2f} Gbps | "
            f"{total/elapsed/1e9:6.2f} GB/s | "
            f"{elapsed/args.iters:6.6f} s"
        )
        dist.barrier()
    print("[Client] Benchmark complete")


def parse_sizes(v: str) -> List[int]:
    try:
        return [int(x) for x in v.split(",") if x]
    except ValueError:
        raise argparse.ArgumentTypeError("bad --sizes")


def main():
    p = argparse.ArgumentParser(
        "UCCL READ/WRITE benchmark with random [-1, 1) bf16 payload"
    )
    p.add_argument("--mode", choices=["read", "write"], default="write")
    p.add_argument("--lazy", action="store_true")
    p.add_argument("--local-gpu-idx", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument(
        "--sizes",
        type=parse_sizes,
        default=[
            4096,
            16384,
            65536,
            262144,
            1048576,
            10485760,
            16*1048576,
            32*1048576,
            67108864,
            104857600,
            
            # 2*104857600,
            # 3*104857600,
            # 4*104857600,
        ],
    )
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup-iters", type=int, default=10,
                   help="warmup transfers before the timed loop")
    p.add_argument("--async-api", action="store_true")
    p.add_argument("--num-iovs", type=int, default=1)
    p.add_argument("--use-raw-cudamalloc", action="store_true",
                   help="bypass torch's caching allocator; use raw cudaMalloc "
                        "for the user payload buffers. Both nodes must enable "
                        "this together. Used to test whether NIC PCIe DMA "
                        "throughput on cudaMalloc buffers (which compress/"
                        "decompress_buffer also use) differs from torch's "
                        "caching-allocator buffers.")
    args = p.parse_args()

    print(f"Mode: {args.mode.upper()} (random bf16 payload)")
    print("Sizes:", ", ".join(_pretty(s) for s in args.sizes))
    if args.async_api:
        print("Async path enabled")

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"

    if args.lazy:
        ep = p2p.Endpoint()
    else:
        ep = p2p.Endpoint(args.local_gpu_idx)
    local_metadata = ep.get_metadata()

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

    if rank == 0:
        _run_client(args, ep, remote_metadata)
    elif rank == 1:
        _run_server(args, ep, remote_metadata)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Ctrl-C] Aborted.")
        sys.exit(1)
