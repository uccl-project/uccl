# 从 uccl.ep 导入需要的类
from uccl.ep import Config, EventHandle

# 从当前包的模块导入
from .utils import EventOverlap, check_nvlink_connections, initialize_uccl, destroy_uccl
from .buffer import Buffer as _OriginalBuffer

# 定义 __all__ 来明确指定包的公共接口
__all__ = [
    'Config',
    'EventHandle',
    'Buffer',
    'EventOverlap',
    'check_nvlink_connections',
    'initialize_uccl',
    'destroy_uccl',
]

# Monkeypatch Buffer to automatically initialize UCCL proxies
import os
import torch
import torch.distributed as dist


class Buffer(_OriginalBuffer):
    """
    Wrapper around the original Buffer class that automatically initializes UCCL proxies.
    """
    # Track initialized devices to avoid re-initialization
    _initialized_devices = set()
    _proxies_dict = {}
    _workers_dict = {}
    _scratch_buffers = {}

    def __init__(self, *args, **kwargs):
        # Get device index before calling super().__init__
        device_index = torch.cuda.current_device()

        # Extract group from args or kwargs to get rank info
        if len(args) > 0 and isinstance(args[0], dist.ProcessGroup):
            group = args[0]
        elif 'group' in kwargs:
            group = kwargs['group']
        else:
            raise ValueError("ProcessGroup 'group' must be provided as first argument")

        rank = group.rank()
        num_ranks = group.size()

        # Extract buffer sizes from args/kwargs
        # Signature: __init__(self, group, rdma_buffer_ptr=None, num_nvl_bytes=0, num_rdma_bytes=0, ...)
        if len(args) > 2:
            num_nvl_bytes = args[2] if len(args) > 2 else kwargs.get('num_nvl_bytes', 0)
            num_rdma_bytes = args[3] if len(args) > 3 else kwargs.get('num_rdma_bytes', 0)
        else:
            num_nvl_bytes = kwargs.get('num_nvl_bytes', 0)
            num_rdma_bytes = kwargs.get('num_rdma_bytes', 0)

        # Initialize UCCL proxies if not already done for this device
        if device_index not in Buffer._initialized_devices:
            # Create a scratch buffer for UCCL
            scratch_nbytes = max(num_nvl_bytes, num_rdma_bytes, 1024 * 1024 * 256)  # At least 256MB
            scratch = torch.empty(scratch_nbytes, dtype=torch.uint8, device=f"cuda:{device_index}")

            # Determine if this is intranode
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
            is_intranode = (num_ranks <= local_world_size)

            # Initialize UCCL
            proxies, workers = initialize_uccl(
                scratch=scratch,
                scratch_nbytes=scratch_nbytes,
                rank=rank,
                num_ranks=num_ranks,
                group=group,
                num_experts=0,  # Will be set later if needed
                is_intranode=is_intranode
            )

            Buffer._initialized_devices.add(device_index)
            Buffer._proxies_dict[device_index] = proxies
            Buffer._workers_dict[device_index] = workers
            Buffer._scratch_buffers[device_index] = scratch

            # If no rdma_buffer_ptr provided, use our scratch buffer
            if len(args) > 1:
                if args[1] is None and num_rdma_bytes > 0:
                    # Update args tuple with scratch data_ptr
                    args = list(args)
                    args[1] = scratch.data_ptr()
                    args = tuple(args)
            elif 'rdma_buffer_ptr' not in kwargs or kwargs['rdma_buffer_ptr'] is None:
                if num_rdma_bytes > 0:
                    kwargs['rdma_buffer_ptr'] = scratch.data_ptr()

        # Call original Buffer.__init__
        super().__init__(*args, **kwargs)

        # Connect atomic buffer using the first proxy
        proxies = Buffer._proxies_dict.get(device_index)
        if proxies and len(proxies) > 0:
            self.connect_atomic_buffer(proxies[0])

    @staticmethod
    def cleanup_uccl(device_index=None):
        """
        Clean up UCCL proxies for a specific device or all devices.

        Arguments:
            device_index: The device index to clean up. If None, clean up all devices.
        """
        if device_index is None:
            # Clean up all devices
            for dev_idx in list(Buffer._initialized_devices):
                proxies = Buffer._proxies_dict.get(dev_idx)
                workers = Buffer._workers_dict.get(dev_idx)
                if proxies:
                    destroy_uccl(proxies, workers)
            Buffer._initialized_devices.clear()
            Buffer._proxies_dict.clear()
            Buffer._workers_dict.clear()
            Buffer._scratch_buffers.clear()
        else:
            # Clean up specific device
            if device_index in Buffer._initialized_devices:
                proxies = Buffer._proxies_dict.get(device_index)
                workers = Buffer._workers_dict.get(device_index)
                if proxies:
                    destroy_uccl(proxies, workers)
                Buffer._initialized_devices.discard(device_index)
                Buffer._proxies_dict.pop(device_index, None)
                Buffer._workers_dict.pop(device_index, None)
                Buffer._scratch_buffers.pop(device_index, None)
