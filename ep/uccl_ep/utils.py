"""
Utility functions and classes for UCCL EP.
"""

import os
import glob
import time
import torch
import torch.distributed as dist
from typing import Any, Optional, Tuple

try:
    import ep
    from ep import EventHandle
except ImportError as exc:
    import sys

    sys.stderr.write("Failed to import C++ extension 'ep'\n")
    raise


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
        with event_overlap:
            do_something_on_current_stream()
        # After exiting the `with` scope, the current stream will wait the event to be finished.
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


def _discover_local_ip():
    """
    Try to infer the IP that can reach MASTER_ADDR (works in most clusters).
    """
    import socket

    master = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = int(os.environ.get("MASTER_PORT", "29500"))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP connect doesn't send packets; just selects a route/interface
        s.connect((master, port))
        return s.getsockname()[0]
    finally:
        s.close()


def _gather_peer_ips(group):
    """
    Gather local IP strings across ranks.
    """
    world = dist.get_world_size(group)
    my_ip = _discover_local_ip()
    ips = [None] * world
    dist.all_gather_object(ips, my_ip, group=group)
    return ips


def get_cpu_proxies_meta(rank, scratch_ptr, scratch_bytes, num_ranks, group):
    """
    Gather metadata for CPU proxies across all ranks.
    """
    meta = {
        "rank": rank,
        "ptr": int(scratch_ptr),
        "nbytes": int(scratch_bytes),
        "ip": _discover_local_ip(),
    }
    all_meta = [None] * num_ranks
    device_index = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device_index)
    dist.all_gather_object(all_meta, meta, group=group)
    rank2meta = {m["rank"]: m for m in all_meta}
    return rank2meta


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
    """
    Initialize UCCL proxy threads for communication.

    Arguments:
        scratch_ptr: pointer to the RDMA buffer.
        scratch_nbytes: size of the RDMA buffer in bytes.
        rank: the rank of the current process.
        num_ranks: the total number of ranks.
        group: the communication group.
        num_experts: the number of experts (optional).
        is_intranode: whether this is intranode communication only.
        use_normal_mode: whether to use normal mode (vs low-latency mode).

    Returns:
        proxies: list of proxy objects.
        workers: worker manager (currently None).
    """
    try:
        for shm_file in glob.glob("/dev/shm/uccl_barrier_*"):
            os.remove(shm_file)
    except Exception:
        pass

    local_rank = int(os.environ["LOCAL_RANK"])
    nproc_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    node_idx = rank // nproc_per_node

    if int(os.environ.get("WORLD_SIZE")) % nproc_per_node != 0:
        raise ValueError("WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE")

    proxies = []
    rank2meta = get_cpu_proxies_meta(
        rank, scratch_ptr, scratch_nbytes, num_ranks, group
    )
    peers_meta_list = [rank2meta[r] for r in range(num_ranks)]
    peer_ip = rank2meta[(rank + 1) % num_ranks]["ip"]

    for i in range(ep.get_num_proxy_threads()):
        proxy = ep.Proxy(
            thread_idx=i,
            gpu_buffer_addr=scratch_ptr,
            total_size=scratch_nbytes,
            rank=rank,
            node_idx=node_idx,
            local_rank=local_rank,
            peer_ip="" if is_intranode else peer_ip,
            num_experts=num_experts,
            num_ranks=num_ranks,
            num_nodes=int(os.environ.get("WORLD_SIZE")) // nproc_per_node,
            use_normal_mode=use_normal_mode,
        )
        if not is_intranode:
            proxy.set_peers_meta(peers_meta_list)
        proxies.append(proxy)
    ep.register_proxies(local_rank, proxies)

    dist.barrier(group)
    if not is_intranode:
        for proxy in proxies:
            proxy.start_dual()

    workers = None
    time.sleep(3)
    return proxies, workers


def destroy_uccl(proxies, workers):
    """
    Destroy UCCL proxy threads and clean up resources.

    Arguments:
        proxies: list of proxy objects to destroy.
        workers: worker manager to stop (optional).
    """
    device_index = int(os.environ["LOCAL_RANK"])
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
