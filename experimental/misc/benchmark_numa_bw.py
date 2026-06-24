from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import threading
import time
from contextlib import contextmanager

import torch


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


def _load_libc():
    return ctypes.CDLL(None)


def _load_libnuma():
    libnuma = ctypes.CDLL("libnuma.so.1")
    libnuma.numa_available.restype = ctypes.c_int
    libnuma.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
    libnuma.numa_alloc_onnode.restype = ctypes.c_void_p
    libnuma.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return libnuma


def _load_cudart():
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaHostRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    cudart.cudaHostRegister.restype = ctypes.c_int
    cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
    cudart.cudaHostUnregister.restype = ctypes.c_int
    cudart.cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    cudart.cudaMemcpyAsync.restype = ctypes.c_int
    cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamSynchronize.restype = ctypes.c_int
    cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
    cudart.cudaGetErrorString.restype = ctypes.c_char_p
    return cudart


LIBC = _load_libc()
LIBNUMA = _load_libnuma()
CUDART = _load_cudart()

LIBC.memmove.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
LIBC.memmove.restype = ctypes.c_void_p
LIBC.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
LIBC.memset.restype = ctypes.c_void_p


def _check_cuda(rc: int, op: str):
    if rc != 0:
        err = CUDART.cudaGetErrorString(rc)
        msg = err.decode() if err else f"cuda error {rc}"
        raise RuntimeError(f"{op} failed: {msg}")


def _parse_cpu_list(cpulist: str):
    cpus = []
    for part in cpulist.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _get_available_nodes():
    nodes = []
    sysfs_root = "/sys/devices/system/node"
    for entry in os.listdir(sysfs_root):
        if entry.startswith("node") and entry[4:].isdigit():
            nodes.append(int(entry[4:]))
    return sorted(nodes)


def _get_node_cpus(node: int):
    cpulist_path = f"/sys/devices/system/node/node{node}/cpulist"
    with open(cpulist_path, "r", encoding="utf-8") as f:
        return _parse_cpu_list(f.read())


def _normalize_bus_id(bus_id: str):
    domain, bus, device_func = bus_id.strip().split(":")
    device, func = device_func.split(".")
    return f"{int(domain, 16):04x}:{int(bus, 16):02x}:{int(device, 16):02x}.{int(func)}"


def _get_gpu_bus_id(gpu_index: int):
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id",
            "--format=csv,noheader",
        ],
        text=True,
    )
    for line in out.splitlines():
        index_str, bus_id = [part.strip() for part in line.split(",", 1)]
        if int(index_str) == gpu_index:
            return _normalize_bus_id(bus_id)
    raise ValueError(f"Failed to find PCI bus ID for GPU {gpu_index}")


def _get_gpu_numa_node(gpu_index: int):
    bus_id = _get_gpu_bus_id(gpu_index)
    path = f"/sys/bus/pci/devices/{bus_id}/numa_node"
    with open(path, "r", encoding="utf-8") as f:
        node = int(f.read().strip())
    if node < 0:
        raise RuntimeError(f"GPU {gpu_index} has no NUMA affinity in {path}")
    return node


@contextmanager
def _bind_to_node(node: int):
    old_affinity = os.sched_getaffinity(0)
    cpus = _get_node_cpus(node)
    if not cpus:
        raise RuntimeError(f"No CPUs found for NUMA node {node}")
    os.sched_setaffinity(0, set(cpus))
    try:
        yield
    finally:
        os.sched_setaffinity(0, old_affinity)


class NumaBuffer:
    def __init__(self, ptr: int, size_bytes: int):
        self.ptr = ptr
        self.size_bytes = size_bytes
        self.is_registered = False

    def memset(self, value: int):
        LIBC.memset(ctypes.c_void_p(self.ptr), value, self.size_bytes)

    def cuda_register(self):
        if not self.is_registered:
            _check_cuda(
                CUDART.cudaHostRegister(
                    ctypes.c_void_p(self.ptr), self.size_bytes, 0
                ),
                "cudaHostRegister",
            )
            self.is_registered = True

    def cuda_unregister(self):
        if self.is_registered:
            _check_cuda(
                CUDART.cudaHostUnregister(ctypes.c_void_p(self.ptr)),
                "cudaHostUnregister",
            )
            self.is_registered = False

    def free(self):
        self.cuda_unregister()
        LIBNUMA.numa_free(ctypes.c_void_p(self.ptr), self.size_bytes)


def _alloc_numa_buffer(size_bytes: int, node: int) -> NumaBuffer:
    ptr = LIBNUMA.numa_alloc_onnode(size_bytes, node)
    if not ptr:
        raise MemoryError(f"numa_alloc_onnode({size_bytes}, node={node}) failed")
    buf = NumaBuffer(int(ptr), size_bytes)
    buf.memset(0)
    return buf


def _measure_cpu_to_cpu(size_bytes: int, num_iters: int, run_node: int, src_node: int, dst_node: int):
    return _measure_cpu_to_cpu_parallel(
        size_bytes=size_bytes,
        num_iters=num_iters,
        run_node=run_node,
        src_node=src_node,
        dst_node=dst_node,
        num_threads=1,
    )


def _measure_cpu_to_cpu_parallel(
    size_bytes: int,
    num_iters: int,
    run_node: int,
    src_node: int,
    dst_node: int,
    num_threads: int,
):
    run_cpus = _get_node_cpus(run_node)
    if not run_cpus:
        raise RuntimeError(f"No CPUs found for NUMA node {run_node}")
    if num_threads < 1:
        raise ValueError("num_threads must be >= 1")
    if num_threads > len(run_cpus):
        raise ValueError(
            f"num_threads={num_threads} exceeds available CPUs on node {run_node}: {len(run_cpus)}"
        )

    src = _alloc_numa_buffer(size_bytes, src_node)
    dst = _alloc_numa_buffer(size_bytes, dst_node)
    ready = threading.Barrier(num_threads + 1)
    done = threading.Barrier(num_threads + 1)
    errors = []
    threads = []

    def worker(thread_idx: int, chunk_start: int, chunk_size: int, cpu: int):
        try:
            os.sched_setaffinity(threading.get_native_id(), {cpu})
            src_ptr = ctypes.c_void_p(src.ptr + chunk_start)
            dst_ptr = ctypes.c_void_p(dst.ptr + chunk_start)
            LIBC.memmove(dst_ptr, src_ptr, chunk_size)
            ready.wait()
            for _ in range(num_iters):
                LIBC.memmove(dst_ptr, src_ptr, chunk_size)
            done.wait()
        except BaseException as exc:
            errors.append((thread_idx, exc))
            try:
                ready.abort()
            except threading.BrokenBarrierError:
                pass
            try:
                done.abort()
            except threading.BrokenBarrierError:
                pass

    try:
        base = size_bytes // num_threads
        remainder = size_bytes % num_threads
        offset = 0
        for thread_idx in range(num_threads):
            chunk_size = base + (1 if thread_idx < remainder else 0)
            thread = threading.Thread(
                target=worker,
                args=(thread_idx, offset, chunk_size, run_cpus[thread_idx]),
                daemon=False,
            )
            offset += chunk_size
            thread.start()
            threads.append(thread)

        ready.wait()
        start = time.perf_counter()
        done.wait()
        end = time.perf_counter()
    finally:
        for thread in threads:
            thread.join()
        src.free()
        dst.free()

    if errors:
        thread_idx, exc = errors[0]
        raise RuntimeError(f"CPU memcpy worker {thread_idx} failed") from exc

    return (size_bytes * num_iters) / (end - start) / 1e9


def _measure_cpu_gpu(size_bytes: int, num_iters: int, run_node: int, host_node: int, gpu_index: int):
    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    stream_handle = ctypes.c_void_p(stream.cuda_stream)
    device_buffer = torch.empty(size_bytes, dtype=torch.uint8, device=device)

    with _bind_to_node(run_node):
        host = _alloc_numa_buffer(size_bytes, host_node)
        try:
            host.cuda_register()
            torch.cuda.synchronize(device)

            _check_cuda(
                CUDART.cudaMemcpyAsync(
                    ctypes.c_void_p(device_buffer.data_ptr()),
                    ctypes.c_void_p(host.ptr),
                    size_bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    stream_handle,
                ),
                "cudaMemcpyAsync(H2D warmup)",
            )
            _check_cuda(
                CUDART.cudaMemcpyAsync(
                    ctypes.c_void_p(host.ptr),
                    ctypes.c_void_p(device_buffer.data_ptr()),
                    size_bytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                    stream_handle,
                ),
                "cudaMemcpyAsync(D2H warmup)",
            )
            _check_cuda(
                CUDART.cudaStreamSynchronize(stream_handle),
                "cudaStreamSynchronize(warmup)",
            )

            start = time.perf_counter()
            for _ in range(num_iters):
                _check_cuda(
                    CUDART.cudaMemcpyAsync(
                        ctypes.c_void_p(device_buffer.data_ptr()),
                        ctypes.c_void_p(host.ptr),
                        size_bytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        stream_handle,
                    ),
                    "cudaMemcpyAsync(H2D)",
                )
            _check_cuda(
                CUDART.cudaStreamSynchronize(stream_handle),
                "cudaStreamSynchronize(H2D)",
            )
            mid = time.perf_counter()

            for _ in range(num_iters):
                _check_cuda(
                    CUDART.cudaMemcpyAsync(
                        ctypes.c_void_p(host.ptr),
                        ctypes.c_void_p(device_buffer.data_ptr()),
                        size_bytes,
                        CUDA_MEMCPY_DEVICE_TO_HOST,
                        stream_handle,
                    ),
                    "cudaMemcpyAsync(D2H)",
                )
            _check_cuda(
                CUDART.cudaStreamSynchronize(stream_handle),
                "cudaStreamSynchronize(D2H)",
            )
            end = time.perf_counter()
        finally:
            host.free()

    h2d_bw = (size_bytes * num_iters) / (mid - start) / 1e9
    d2h_bw = (size_bytes * num_iters) / (end - mid) / 1e9
    return h2d_bw, d2h_bw


def _pick_default_nodes():
    nodes = _get_available_nodes()
    if not nodes:
        raise RuntimeError("No NUMA nodes found under /sys/devices/system/node")
    if len(nodes) == 1:
        return nodes[0], nodes[0], nodes[0]
    return nodes[0], nodes[1], nodes[0]


def _pick_remote_node(local_node: int):
    nodes = _get_available_nodes()
    for node in nodes:
        if node != local_node:
            return node
    return local_node


def parse_args():
    default_run_node, default_src_node, default_dst_node = _pick_default_nodes()
    p = argparse.ArgumentParser(
        "Benchmark cross-NUMA CPU-CPU and CPU-GPU bandwidth"
    )
    p.add_argument("--buffer-size-mb", type=int, default=1024)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--run-node", type=int, default=default_run_node)
    p.add_argument("--src-node", type=int, default=default_src_node)
    p.add_argument("--dst-node", type=int, default=default_dst_node)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--cpu-gpu-host-node", type=int, default=None)
    p.add_argument("--cpu-threads", type=int, default=None)
    return p.parse_args()


def main():
    if LIBNUMA.numa_available() < 0:
        raise RuntimeError("libnuma reports NUMA is not available on this system")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args = parse_args()
    size_bytes = args.buffer_size_mb * 1024 * 1024

    nodes = _get_available_nodes()
    for node in (args.run_node, args.src_node, args.dst_node):
        if node not in nodes:
            raise ValueError(f"NUMA node {node} not found; available nodes: {nodes}")

    gpu_numa_node = _get_gpu_numa_node(args.gpu_index)
    cpu_gpu_host_node = args.cpu_gpu_host_node
    if cpu_gpu_host_node is None:
        cpu_gpu_host_node = _pick_remote_node(gpu_numa_node)
    if cpu_gpu_host_node not in nodes:
        raise ValueError(
            f"CPU-GPU host NUMA node {cpu_gpu_host_node} not found; available nodes: {nodes}"
        )

    cpu_threads = args.cpu_threads
    if cpu_threads is None:
        cpu_threads = len(_get_node_cpus(args.run_node))

    cpu_to_cpu_bw = _measure_cpu_to_cpu(
        size_bytes, args.iters, args.run_node, args.src_node, args.dst_node
    ) if cpu_threads == 1 else _measure_cpu_to_cpu_parallel(
        size_bytes=size_bytes,
        num_iters=args.iters,
        run_node=args.run_node,
        src_node=args.src_node,
        dst_node=args.dst_node,
        num_threads=cpu_threads,
    )
    h2d_bw, d2h_bw = _measure_cpu_gpu(
        size_bytes, args.iters, args.run_node, cpu_gpu_host_node, args.gpu_index
    )

    print(f"Buffer size: {args.buffer_size_mb} MB")
    print(f"Iterations: {args.iters}")
    print(f"Run NUMA node: {args.run_node}")
    print(f"CPU source NUMA node: {args.src_node}")
    print(f"CPU destination NUMA node: {args.dst_node}")
    print(f"CPU memcpy threads: {cpu_threads}")
    print(f"GPU index: {args.gpu_index}")
    print(f"GPU NUMA node: {gpu_numa_node}")
    print(f"CPU-GPU host NUMA node: {cpu_gpu_host_node}")
    print(f"CPU->CPU BW (node {args.src_node} -> node {args.dst_node}): {cpu_to_cpu_bw:.2f} GB/s")
    print(f"CPU(node {cpu_gpu_host_node})->GPU{args.gpu_index} H2D BW: {h2d_bw:.2f} GB/s")
    print(f"GPU{args.gpu_index}->CPU(node {cpu_gpu_host_node}) D2H BW: {d2h_bw:.2f} GB/s")


if __name__ == "__main__":
    main()
