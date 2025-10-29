from uccl.ep import Config, EventHandle

from .utils import EventOverlap, check_nvlink_connections, initialize_uccl, destroy_uccl
from .buffer import Buffer as _Buffer
import torch.distributed as dist

# Wrapper to match DeepEP API (without rdma_buffer_ptr parameter)
class Buffer(_Buffer):
    def __init__(
        self,
        group: dist.ProcessGroup,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 24,
        allow_nvlink_for_low_latency_mode: bool = True,
        allow_mnnvl: bool = False,
        explicitly_destroy: bool = False,
    ):
        # Call parent with rdma_buffer_ptr=None (will use shared buffer)
        super().__init__(
            group=group,
            rdma_buffer_ptr=None,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=low_latency_mode,
            num_qps_per_rank=num_qps_per_rank,
            allow_nvlink_for_low_latency_mode=allow_nvlink_for_low_latency_mode,
            allow_mnnvl=allow_mnnvl,
            explicitly_destroy=explicitly_destroy,
        )

__all__ = [
    'Config',
    'EventHandle',
    'Buffer',
    'EventOverlap',
    'check_nvlink_connections',
    'initialize_uccl',
    'destroy_uccl',
]
