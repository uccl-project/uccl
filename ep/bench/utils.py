import inspect
from typing import Any, Optional, Tuple, Union
import os
import torch
import torch.distributed as dist
from typing import Optional
import glob
import sys
from uccl.ep import EventHandle
import tempfile
import json
from pathlib import Path
import time
import numpy as np

# import deep_ep as ep
try:
    from uccl import ep
except ImportError as exc:
    import sys

    sys.stderr.write("Failed to import uccl.ep\n")
    raise

# import deep_ep as ep
try:
    from uccl import ep
except ImportError as exc:
    import sys

    sys.stderr.write("Failed to import uccl.ep\n")
    raise


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def hash_tensor(t: torch.Tensor):
    return t.view(torch.int64).sum().item()


def init_dist(local_rank: int, num_local_ranks: int):
    # Set device

    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": world_size,
        "rank": node_rank,
    }
    print(params)
    if "device_id" in sig.parameters:
        # noinspection PyTypeChecker
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    return (
        dist.get_rank(),
        dist.get_world_size(),
        dist.new_group(list(range(world_size))),
    )


def init_dist_under_torchrun(local_rank: int, num_local_ranks: int):
    # torchrun already sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    
    print(f"init_dist_under_torchrun: local_rank = {local_rank}, num_local_ranks = {num_local_ranks}")

    return (
        dist.get_rank(),
        dist.get_world_size(),
        dist.new_group(list(range(dist.get_world_size()))),
    )


def _gather_peer_ips(group):
    # Gather local IP strings across ranks
    world = dist.get_world_size(group)
    my_ip = ep.get_oob_ip()
    ips = [None] * world
    dist.all_gather_object(ips, my_ip, group=group)
    return ips


def get_peer_ip(rank: int, num_ranks: int, group: dist.ProcessGroup):

    if num_ranks == 1:
        # single-process local test: okay to leave blank (or 127.0.0.1)
        peer_ip = ""
    else:
        ips = _gather_peer_ips(group)
        # simple ring: next rank is your peer
        peer_ip = ips[(rank + 1) % num_ranks]
    return peer_ip if peer_ip else ""


def get_cpu_proxies_meta(proxies, rank, scratch_ptr, scratch_bytes, num_ranks, group):
    my_ip = ep.get_oob_ip()
    meta = {
        "rank": rank,
        "ptr": int(scratch_ptr),
        "nbytes": int(scratch_bytes),
        "ip": my_ip,
        "listen_ports": [proxy.get_listen_port() for proxy in proxies],
    }
    all_meta = [None] * num_ranks
    # Use current device or fallback to LOCAL_RANK or 0
    if "LOCAL_RANK" in os.environ:
        device_index = int(os.environ["LOCAL_RANK"])
    else:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    dist.all_gather_object(all_meta, meta, group=group)
    rank2meta = {m["rank"]: m for m in all_meta}

    # Debug: print IP distribution
    ip_counts = {}
    for m in all_meta:
        ip = m["ip"]
        ip_counts[ip] = ip_counts.get(ip, 0) + 1
    if rank == 0:
        print(f"[DEBUG] IP distribution across {num_ranks} ranks:", flush=True)
        for ip, count in ip_counts.items():
            print(f"[DEBUG]   {ip}: {count} ranks", flush=True)

    return rank2meta


def check_nvlink_connections(group: dist.ProcessGroup):
    """
    Check NVLink connection between every pair of GPUs.

    Arguments:
        group: the communication group.
    """
    # Check NVLink connection
    # NOTES: some A100 PCIE GPUs only have pairwise NVLink connection, so that we can only use EP2
    # TODO: check all cases, all local-node GPUs in the group should be connected via NVLink
    if "PCIE" in torch.cuda.get_device_name():
        assert group.size() <= 2, "PCIe GPUs only have pairwise NVLink connections"

        # noinspection PyUnresolvedReferences
        import pynvml

        pynvml.nvmlInit()

        # noinspection PyTypeChecker
        devices = (
            os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
            .strip(",")
            .split(",")
        )
        physical_device_idx = int(devices[torch.cuda.current_device()])
        physical_device_indices = [
            0,
        ] * group.size()
        dist.all_gather_object(physical_device_indices, physical_device_idx, group)

        # Check whether they are all connected via NVLink
        # Reference: https://github.com/vllm-project/vllm/blob/b8e809a057765c574726a6077fd124db5077ce1f/vllm/platforms/cuda.py#L438
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_indices
        ]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i >= j:
                    continue
                status = pynvml.nvmlDeviceGetP2PStatus(
                    handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                )
                assert (
                    status == pynvml.NVML_P2P_STATUS_OK
                ), f"GPU {physical_device_indices[i]} and GPU {physical_device_indices[j]} are not connected via NVLink"

        # Close NVML
        pynvml.nvmlShutdown()


class EventOverlap:
    """
    A wrapper class to manage CUDA events, also for better overlapping convenience.

    Attributes:
        event: the CUDA event captured.
        extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
    """

    def __init__(
        self,
        event: Optional[EventHandle] = None,
        extra_tensors: Optional[Tuple[torch.Tensor]] = None,
    ) -> None:
        """
        Initialize the class.

        Arguments:
            event: the CUDA event captured.
            extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
        """
        self.event = event

        # NOTES: we use extra tensors to achieve stream recording, otherwise,
        # stream recording will be incompatible with CUDA graph.
        self.extra_tensors = extra_tensors

    def current_stream_wait(self) -> None:
        """
        The current stream `torch.cuda.current_stream()` waits for the event to be finished.
        """
        assert self.event is not None
        self.event.current_stream_wait()

    def __enter__(self) -> Any:
        """
        Utility for overlapping and Python `with` syntax.

        You can overlap the kernels on the current stream with the following example:
        ```python
        event_overlap = event_after_all_to_all_kernels()
        with event_overlap():
            do_something_on_current_stream()
        # After exiting the `with` scope, the current stream with wait the event to be finished.
        ```
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Utility for overlapping and Python `with` syntax.

        Please follow the example in the `__enter__` function.
        """
        if self.event is not None:
            self.event.current_stream_wait()


def detect_ib_hca():
    """Detect InfiniBand HCA device.

    Returns the first mlx5 device name found, or None if no InfiniBand
    devices are available (e.g., on systems without IB).
    """
    try:
        devices = sorted(glob.glob("/sys/class/infiniband/*"))
    except (OSError, PermissionError):
        return None

    if not devices:
        # No InfiniBand devices found - this is okay on systems without RDMA
        return None

    ib_devs = [
        os.path.basename(d) for d in devices if os.path.basename(d).startswith("mlx5")
    ]
    if not ib_devs:
        return None
    return ib_devs[0]


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench(fn, num_warmups: int = 50, num_tests: int = 50, post_fn=None):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    current_device = torch.cuda.current_device()
    cache = torch.empty(
        int(256e6 // 4), dtype=torch.int, device=f"cuda:{current_device}"
    )

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]
    return np.average(times), np.min(times), np.max(times)


def bench_kineto(
    fn,
    kernel_names: Union[str, tuple],
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    barrier_comm_profiling: bool = False,
    num_kernels_per_period: int = 1,
):
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
        ) as prof:
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    current_device = torch.cuda.current_device()
                    lhs = torch.randn(
                        (8192, 8192),
                        dtype=torch.float,
                        device=f"cuda:{current_device}",
                    )
                    rhs = torch.randn(
                        (8192, 8192),
                        dtype=torch.float,
                        device=f"cuda:{current_device}",
                    )
                    lhs @ rhs
                    dist.all_reduce(
                        torch.ones(
                            1,
                            dtype=torch.float,
                            device=f"cuda:{current_device}",
                        )
                    )
                for _ in range(num_tests):
                    fn()
                torch.cuda.synchronize()
                dist.barrier()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = (
        prof.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        count = sum([name in line for line in prof_lines])
        if count != 1:
            print(f"\n[WARNING] Profiling table for kernel '{name}':")
            print("\n".join(prof_lines))
            print(
                f"[WARNING] Kernel '{name}' found {count} times in profiling table (expected 1)"
            )
            print(f"[WARNING] Continuing execution despite mismatch...\n")

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    # Return average kernel durations
    units = {"ms": 1e3, "us": 1e6}
    kernel_durations = []
    for name in kernel_names:
        found = False
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_durations.append(
                            float(time_str.replace(unit, "")) / scale
                        )
                        found = True
                        break
                break
        # NOTE(MaoZiming): in rare cases it misses certain events.
        if not found:
            print(
                f"[WARNING] Kernel '{name}' not found in profiling table, using 0.0 as placeholder"
            )
            kernel_durations.append(0.0)

    # Expand the kernels by periods
    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            prof.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())

        for i, kernel_name in enumerate(kernel_names):
            events = [
                event
                for event in profile_data["traceEvents"]
                if f"::{kernel_name}" in event["name"]
            ]
            events = sorted(events, key=lambda event: event["ts"])
            durations = [event["dur"] / 1e6 for event in events]

            # Handle incomplete periods gracefully (due to dropped samples)
            # NOTE(MaoZiming): This sometimes happen, suspect it is the profiler's issue.
            num_complete_periods = len(durations) // num_kernels_per_period
            if len(durations) % num_kernels_per_period != 0:
                dropped_samples = len(durations) % num_kernels_per_period
                if dist.get_rank() == 0:
                    print(
                        f"[WARNING] Kernel '{kernel_name}': {dropped_samples} samples dropped "
                        f"(got {len(durations)} samples, expected multiple of {num_kernels_per_period}). "
                        f"Using {num_complete_periods} complete periods.",
                        flush=True,
                    )
                # Truncate to only use complete periods
                durations = durations[: num_complete_periods * num_kernels_per_period]
            if num_complete_periods > 0:
                kernel_durations[i] = [
                    sum(durations[j::num_kernels_per_period]) / num_complete_periods
                    for j in range(num_kernels_per_period)
                ]
            else:
                # If we have no complete periods, fall back to returning zeros
                if dist.get_rank() == 0:
                    print(
                        f"[WARNING] Kernel '{kernel_name}': No complete periods found. "
                        f"Returning zeros.",
                        flush=True,
                    )
                kernel_durations[i] = [0.0] * num_kernels_per_period

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]


def initialize_uccl(
    scratch_ptr,
    scratch_nbytes,
    rank,
    num_ranks,
    group,
    num_experts=0,
    is_intranode=False,
    use_normal_mode=False,
):
    try:
        for shm_file in glob.glob("/dev/shm/uccl_barrier_*"):
            os.remove(shm_file)
    except Exception:
        pass

    # Try to get local_rank from environment or infer from current device
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Fallback: use current CUDA device as local_rank
        local_rank = torch.cuda.current_device()

    # Try to get nproc_per_node from environment
    if "LOCAL_WORLD_SIZE" in os.environ:
        nproc_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        # Fallback: infer from is_intranode and num_ranks
        if is_intranode:
            # All ranks are on the same node
            nproc_per_node = num_ranks
        else:
            # Assume uniform distribution across nodes
            # If we have N GPUs, assume each node has same number of GPUs
            num_gpus = torch.cuda.device_count()
            nproc_per_node = num_gpus if num_gpus > 0 else 1

    node_idx = rank // nproc_per_node if nproc_per_node > 0 else 0

    # Only check WORLD_SIZE consistency if it's defined
    if "WORLD_SIZE" in os.environ and nproc_per_node > 0:
        world_size = int(os.environ.get("WORLD_SIZE"))
        if world_size % nproc_per_node != 0:
            raise ValueError("WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE")

    proxies = []

    # Calculate num_nodes from num_ranks and nproc_per_node
    if nproc_per_node > 0:
        num_nodes = num_ranks // nproc_per_node
    else:
        num_nodes = num_ranks  # Fallback: assume each rank is on a different node

    for i in range(ep.get_num_proxy_threads()):
        proxy = ep.Proxy(
            thread_idx=i,
            gpu_buffer_addr=scratch_ptr,
            total_size=scratch_nbytes,
            rank=rank,
            node_idx=node_idx,
            local_rank=local_rank,
            num_experts=num_experts,
            num_ranks=num_ranks,
            num_nodes=num_nodes,
            use_normal_mode=use_normal_mode,
            is_intranode=is_intranode,
        )
        proxies.append(proxy)

    rank2meta = get_cpu_proxies_meta(
        proxies, rank, scratch_ptr, scratch_nbytes, num_ranks, group
    )
    peers_meta_list = [rank2meta[r] for r in range(num_ranks)]

    if not is_intranode:
        for proxy in proxies:
            proxy.set_peers_meta(peers_meta_list)

    ep.register_proxies(local_rank, proxies)

    # Set atomic buffer pointer for all proxies BEFORE starting them
    # This ensures the atomic buffer info is included in connection info exchange
    # Note: Only thread 0's proxy allocates the atomic buffer in its constructor
    if not is_intranode and len(proxies) > 0:
        # Get atomic buffer pointer from thread 0 proxy (only thread 0 allocates it)
        # This must be done before start_dual() so the atomic buffer info is included
        # in the connection info exchange during init_common()
        atomic_buffer_ptr = proxies[0].get_atomic_buffer_ptr()
        if atomic_buffer_ptr:
            for proxy in proxies:
                proxy.set_atomic_buffer_ptr(atomic_buffer_ptr)

    dist.barrier(group)
    if not is_intranode:
        for proxy in proxies:
            proxy.start_dual()

    workers = None
    # if hasattr(ep, "PeerCopyManager"):
    #     try:
    #         workers = ep.PeerCopyManager(src_device=local_rank)
    #         workers.start_for_proxies(proxies)
    #         if rank == 0:
    #             print("✓ PeerCopyManager started", flush=True)
    #     except Exception as e:
    #         if rank == 0:
    #             print(f"PeerCopyManager unavailable: {e}", flush=True)

    time.sleep(3)
    return proxies, workers


def destroy_uccl(proxies, workers):
    # Use current device or fallback to LOCAL_RANK
    if "LOCAL_RANK" in os.environ:
        device_index = int(os.environ["LOCAL_RANK"])
    else:
        device_index = torch.cuda.current_device()

    if workers is not None:
        try:
            workers.stop()
        except Exception:
            pass

    try:
        for p in proxies:
            p.stop()
    except Exception:
        pass
    try:
        ep.unregister_proxy(device_index)
    except Exception:
        pass
    try:
        for shm_file in glob.glob("/dev/shm/uccl_barrier_*"):
            os.remove(shm_file)
    except Exception:
        pass


def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n
    ), (x_amax / 448.0).view(m, -1)


def create_grouped_scores(
    scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int
):
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)


def inplace_unique(x: torch.Tensor, num_slots: int):
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def hash_tensor(t: torch.Tensor):
    return t.view(torch.int).sum().item()
