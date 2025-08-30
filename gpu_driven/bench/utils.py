import inspect
from typing import Any, Optional, Tuple
import os
import torch
import torch.distributed as dist
from typing import Optional
import glob

from uccl.uccl_ep import EventHandle


def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": num_nodes * num_local_ranks,
        "rank": node_rank * num_local_ranks + local_rank,
    }
    if "device_id" in sig.parameters:
        # noinspection PyTypeChecker
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    return (
        dist.get_rank(),
        dist.get_world_size(),
        dist.new_group(list(range(num_local_ranks * num_nodes))),
    )


def _discover_local_ip():
    # Try to infer the IP that can reach MASTER_ADDR (works in most clusters)
    import socket, os

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
    # Gather local IP strings across ranks
    world = dist.get_world_size(group)
    my_ip = _discover_local_ip()
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


def get_cpu_proxies_meta(rank, scratch_ptr, scratch_bytes, num_ranks, group):
    meta = {
        "rank": rank,
        "ptr": int(scratch_ptr),
        "nbytes": int(scratch_bytes),
        "ip": _discover_local_ip(),
    }
    all_meta = [None] * num_ranks
    dist.all_gather_object(all_meta, meta)
    dist.barrier(group)
    rank2meta = {m["rank"]: m for m in all_meta}
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
    # List all mlx5 devices under /sys/class/infiniband
    devices = sorted(glob.glob("/sys/class/infiniband/mlx5_*"))
    if not devices:
        raise RuntimeError("No mlx5 devices found under /sys/class/infiniband")
    # Just take the first one for now
    dev_name = os.path.basename(devices[0])
    return dev_name
