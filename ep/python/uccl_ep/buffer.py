"""
uccl.ep.buffer — High-level ``Buffer`` wrapper for Expert-Parallel communication.
"""

import os
from contextlib import nullcontext
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from uccl.ep import ep_cpp
from uccl.ep.ep_cpp import Config, EventHandle
from uccl.ep.utils import (
    EventOverlap,
    check_nvlink_connections,
    initialize_uccl,
    destroy_uccl,
    _fp8_e4m3_dtype,
)


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput intranode all-to-all (dispatch and combine, using NVLink)
        - high-throughput internode all-to-all (dispatch and combine, using RDMA and NVLink)
        - low-latency all-to-all (dispatch and combine, using RDMA)

    Attributes:
        num_sms: the SMs used in high-throughput kernels.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_nvl_bytes: the buffer size for intranode NVLink communication.
        num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
        runtime: the C++ runtime.
    """

    num_sms: int = 20

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
        is_intranode: bool = False,
    ) -> None:
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
            low_latency_mode: whether to enable low-latency mode.
            num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals
                to the number of local experts.
            allow_nvlink_for_low_latency_mode: whether allow NVLink traffic for low-latency mode, you should notice
                this is somehow incompatible with the hook-based overlapping.
                Warning: PCIe connections may lead to errors due to memory ordering issues,
                please make sure all connections are via NVLink.
            allow_mnnvl: whether to allow MNNVL
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
        """
        if "LOCAL_RANK" in os.environ:
            device_index = int(os.environ["LOCAL_RANK"])
        else:
            device_index = torch.cuda.current_device()

        if hasattr(ep_cpp, "get_rdma_buffer"):
            scratch_dlpack, rdma_buffer_is_host_allocated = ep_cpp.get_rdma_buffer(
                num_rdma_bytes, device_index
            )
            self.scratch = torch.utils.dlpack.from_dlpack(scratch_dlpack)
        else:
            rdma_buffer_is_host_allocated = False
            if num_rdma_bytes > 0:
                if hasattr(ep_cpp, "can_register_rdma_gpu_buffer"):
                    rdma_buffer_is_host_allocated = not bool(
                        ep_cpp.can_register_rdma_gpu_buffer(device_index, num_rdma_bytes)
                    )
                elif hasattr(ep_cpp, "rdma_buffer_should_use_host_alloc"):
                    rdma_buffer_is_host_allocated = bool(
                        ep_cpp.rdma_buffer_should_use_host_alloc(
                            device_index, num_rdma_bytes
                        )
                    )

            if num_rdma_bytes > 0 and rdma_buffer_is_host_allocated:
                self.scratch = torch.zeros(
                    (num_rdma_bytes,),
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )
            else:
                self.scratch = torch.zeros(
                    max(num_rdma_bytes, 1),
                    dtype=torch.uint8,
                    device=f"cuda:{device_index}",
                )

        rdma_buffer_ptr = self.scratch.data_ptr()
        _local_world = int(os.environ.get("LOCAL_WORLD_SIZE", -1))
        self.proxies, self.workers = initialize_uccl(
            rdma_buffer_ptr,
            num_rdma_bytes,
            group.rank(),
            dist.get_world_size(group),
            group,
            use_normal_mode=not low_latency_mode,
            is_intranode=is_intranode,
            rdma_buffer_is_host_allocated=rdma_buffer_is_host_allocated,
        )
        check_nvlink_connections(group)

        self.rank = group.rank()
        self.group_size = group.size()
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.explicitly_destroy = explicitly_destroy
        self._next_low_latency_combine_buffer = None
        self.runtime = ep_cpp.Buffer(
            self.rank,
            self.group_size,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode,
            explicitly_destroy,
            _local_world,
        )
        if num_rdma_bytes:
            self.runtime.set_rdma_buffer(rdma_buffer_ptr, rdma_buffer_is_host_allocated)

        device_ids = [None] * self.group_size
        local_device_id = self.runtime.get_local_device_id()
        dist.all_gather_object(device_ids, local_device_id, group)

        ipc_handles = [None] * self.group_size
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        dist.all_gather_object(ipc_handles, local_ipc_handle, group)

        rdma_ipc_handles = [None] * self.group_size
        local_rdma_ipc_handle = (
            self.runtime.get_local_rdma_ipc_handle()
            if self.num_rdma_bytes > 0 and not rdma_buffer_is_host_allocated
            else None
        )
        dist.all_gather_object(rdma_ipc_handles, local_rdma_ipc_handle, group)
        root_unique_id = None

        self.runtime.sync(
            device_ids,
            ipc_handles,
            root_unique_id,
            rdma_ipc_handles,
        )
        assert self.runtime.is_available()
        self.connect_atomic_buffer(self.proxies[0])

        for proxy in self.proxies:
            proxy.set_atomic_buffer_ptr(self.proxies[0].get_atomic_buffer_ptr())

    def _ll_compute_stream_ptr(self, device: torch.device):
        current = torch.cuda.current_stream(device=device)
        return int(current.cuda_stream)

    def reset_rdma_buffer(self):
        """Reset the RDMA buffer."""
        self.runtime.reset_rdma_buffer()

    def connect_atomic_buffer(self, proxy: "ep_cpp.Proxy"):
        ep_cpp.connect_atomic_buffer(proxy, self.runtime)

    def destroy(self):
        """Destroy the cpp runtime and release resources."""
        assert self.explicitly_destroy, "`explicitly_destroy` flag must be set"
        self.runtime.destroy()
        self.runtime = None
        destroy_uccl(self.proxies, self.workers)

    @staticmethod
    def is_sm90_compiled():
        return ep_cpp.is_sm90_compiled()

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the number of SMs to use in high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.
        """
        assert new_num_sms % 2 == 0, "The SM count must be even"
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a CUDA event on the current stream.

        Returns:
            event: the captured event.
        """
        stream_ptr = int(torch.cuda.current_stream().cuda_stream)
        return EventOverlap(EventHandle(stream_ptr))

    # noinspection PyTypeChecker
    def low_latency_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable
    ]:
        """
        A low-latency implementation for dispatching with IBGDA.

        Arguments:
            x: ``torch.Tensor`` with ``torch.bfloat16``, shaped as ``[num_tokens, hidden]``.
            topk_idx: ``torch.Tensor`` with ``torch.int64``, shaped as ``[num_tokens, num_topk]``.
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch.
            num_experts: the number of all experts.
            cumulative_local_expert_recv_stats: optional stats tensor.
            dispatch_wait_recv_cost_stats: optional stats tensor.
            use_fp8: whether to enable FP8 casting.
            round_scale: whether to round scaling factors into power of 2.
            use_ue8m0: whether to use UE8M0 as scaling factor format.
            async_finish: whether to skip waiting for kernel completion.
            return_recv_hook: whether to return a receiving hook.

        Returns:
            recv_x, recv_count, handle, event, hook
        """
        for proxy in self.proxies:
            proxy.calculate_and_set_dispatch_recv_data_offset(
                num_tokens=x.shape[0],
                hidden=x.shape[1],
                num_experts=num_experts,
            )
        num_ranks = self.group.size()
        num_local_experts = num_experts // num_ranks
        num_recv_tokens = num_ranks * num_max_dispatch_tokens_per_rank
        packed_recv_x = torch.empty(
            (num_local_experts, num_recv_tokens, x.size(1)),
            device=x.device,
            dtype=_fp8_e4m3_dtype() if use_fp8 else torch.bfloat16,
        )
        packed_recv_count = torch.empty(
            (num_local_experts,), device=x.device, dtype=torch.int32
        )
        packed_recv_src_info = torch.empty(
            (num_local_experts, num_recv_tokens), device=x.device, dtype=torch.int32
        )
        packed_recv_layout_range = torch.empty(
            (num_local_experts, num_ranks), device=x.device, dtype=torch.int64
        )
        packed_recv_x_scales_storage = None
        packed_recv_x_scales = None
        packed_recv_x_scales_ptr = 0
        if use_fp8:
            if use_ue8m0:
                packed_recv_x_scales_storage = torch.empty(
                    (num_local_experts, x.size(1) // 512, num_recv_tokens),
                    device=x.device,
                    dtype=torch.int32,
                )
            else:
                packed_recv_x_scales_storage = torch.empty(
                    (num_local_experts, x.size(1) // 128, num_recv_tokens),
                    device=x.device,
                    dtype=torch.float32,
                )
            packed_recv_x_scales = packed_recv_x_scales_storage.transpose(1, 2)
            packed_recv_x_scales_ptr = packed_recv_x_scales.data_ptr()

        compute_stream_ptr = self._ll_compute_stream_ptr(x.device)
        event, hook = self.runtime.low_latency_dispatch(
            x.data_ptr(),
            x.size(0),
            x.size(1),
            topk_idx.data_ptr(),
            topk_idx.size(0),
            topk_idx.size(1),
            packed_recv_x.data_ptr(),
            packed_recv_x_scales_ptr,
            packed_recv_count.data_ptr(),
            packed_recv_src_info.data_ptr(),
            packed_recv_layout_range.data_ptr(),
            (
                0
                if cumulative_local_expert_recv_stats is None
                else cumulative_local_expert_recv_stats.data_ptr()
            ),
            (
                0
                if dispatch_wait_recv_cost_stats is None
                else dispatch_wait_recv_cost_stats.data_ptr()
            ),
            compute_stream_ptr,
            int(num_max_dispatch_tokens_per_rank),
            int(num_experts),
            bool(use_fp8),
            bool(round_scale),
            bool(use_ue8m0),
            bool(async_finish),
            bool(return_recv_hook),
        )
        handle = (
            packed_recv_src_info,
            packed_recv_layout_range,
            num_max_dispatch_tokens_per_rank,
            x.size(1),
            num_experts,
        )
        tensors_to_record = (
            x,
            topk_idx,
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            packed_recv_x_scales_storage,
            cumulative_local_expert_recv_stats,
        )
        return (
            (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x,
            packed_recv_count,
            handle,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )

    # noinspection PyTypeChecker
    def low_latency_combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        use_logfmt: bool = False,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[torch.Tensor] = None,
        combine_wait_recv_cost_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, EventOverlap, Callable]:
        """
        A low-latency implementation for combining tokens with IBGDA.

        Arguments:
            x: ``[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]``.
            topk_idx: ``[num_combined_tokens, num_topk]``.
            topk_weights: ``[num_combined_tokens, num_topk]``.
            handle: the communication handle given by the ``dispatch`` function.
            use_logfmt: whether to use LogFMT format.
            zero_copy: whether the tensor is already copied into the RDMA buffer.
            async_finish: whether to skip waiting for kernel completion.
            return_recv_hook: whether to return a receiving hook.
            out: optional in-place output tensor.
            combine_wait_recv_cost_stats: optional stats tensor.

        Returns:
            combined_x, event, hook
        """
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
        ) = handle
        x_for_combine = x
        if zero_copy and self._next_low_latency_combine_buffer is not None:
            staged = self._next_low_latency_combine_buffer
            if (
                staged.shape == x.shape
                and staged.dtype == x.dtype
                and staged.device == x.device
            ):
                x_for_combine = staged
        combined_x = (
            out
            if out is not None
            else torch.empty((topk_idx.size(0), hidden), device=x.device, dtype=x.dtype)
        )
        compute_stream_ptr = self._ll_compute_stream_ptr(x.device)
        event, hook = self.runtime.low_latency_combine(
            x_for_combine.data_ptr(),
            x_for_combine.size(0),
            x_for_combine.size(1),
            x_for_combine.size(2),
            topk_idx.data_ptr(),
            topk_idx.size(0),
            topk_idx.size(1),
            topk_weights.data_ptr(),
            src_info.data_ptr(),
            src_info.size(0),
            src_info.size(1),
            layout_range.data_ptr(),
            layout_range.size(0),
            layout_range.size(1),
            (
                0
                if combine_wait_recv_cost_stats is None
                else combine_wait_recv_cost_stats.data_ptr()
            ),
            compute_stream_ptr,
            int(num_max_dispatch_tokens_per_rank),
            int(num_experts),
            bool(use_logfmt),
            False,
            bool(async_finish),
            bool(return_recv_hook),
            combined_x.data_ptr(),
        )
        tensors_to_record = (
            x_for_combine,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            combined_x,
        )
        return (
            combined_x,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )

    def get_next_low_latency_combine_buffer(self, handle: object):
        """
        Get the raw registered RDMA buffer tensor for next low-latency combine.

        Arguments:
            handle: the communication handle given by the ``dispatch`` function.

        Returns:
            buffer: the raw RDMA low-latency buffer as a BF16 PyTorch tensor.
        """
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
        ) = handle
        num_ranks = self.group.size()
        num_local_experts = num_experts // num_ranks
        num_recv_tokens = num_ranks * num_max_dispatch_tokens_per_rank
        self._next_low_latency_combine_buffer = torch.empty(
            (num_local_experts, num_recv_tokens, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        return self._next_low_latency_combine_buffer

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ranks: int,
        num_experts: int,
    ) -> int:
        """
        Get a minimum size requirement for the RDMA buffer.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch.
            hidden: the hidden dimension of each token.
            num_ranks: the number of EP group ranks.
            num_experts: the number of all experts.

        Returns:
            size: the RDMA buffer size recommended.
        """
        return ep_cpp.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts
        )

    def get_comm_stream(self) -> torch.Stream:
        """Get the communication stream."""
        ts = self.runtime.get_comm_stream()
        if isinstance(ts, torch.Stream):
            return torch.cuda.Stream(
                stream_id=ts.stream_id,
                device_index=ts.device_index,
                device_type=ts.device_type,
            )
        return torch.cuda.ExternalStream(int(ts))

    def get_local_buffer_tensor(
        self,
        dtype: torch.dtype,
        size: Optional[torch.Size] = None,
        offset: int = 0,
        use_rdma_buffer: bool = False,
    ) -> torch.Tensor:
        """
        Get the raw buffer (slice supported) as a PyTorch tensor.

        Argument:
            dtype: the data type (PyTorch ``dtype``) for the tensor.
            size: the slice size (by elements) to get from the buffer.
            offset: the offset of the beginning element.
            use_rdma_buffer: whether to return the RDMA buffer.
        """
        assert dtype in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.bool,
        }, f"Unsupported dtype for get_local_buffer_tensor: {dtype}"
        if use_rdma_buffer:
            tensor = self.scratch.view(dtype)
            if offset > 0:
                tensor = tensor[offset:]
        else:
            raise RuntimeError(
                "get_local_buffer_tensor(use_rdma_buffer=False) is not available "
                "without Torch C++ tensor bindings"
            )
        if size is None:
            return tensor
        assert tensor.numel() >= size.numel()
        return tensor[: size.numel()].view(size)

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    @staticmethod
    def _dtype_code(dtype: torch.dtype) -> int:
        table = {
            torch.uint8: 0,
            torch.int8: 1,
            torch.int16: 2,
            torch.int32: 3,
            torch.int64: 4,
            torch.float16: 5,
            torch.bfloat16: 6,
            torch.float32: 7,
            torch.float64: 8,
            torch.bool: 9,
            torch.float8_e4m3fn: 10,
            torch.float8_e4m3fnuz: 10,
        }
        if dtype not in table:
            raise ValueError(f"Unsupported dtype for uccl combine: {dtype}")
        return table[dtype]

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """Get a recommended dispatch config."""
        config_map = {
            2: Config(Buffer.num_sms, 24, 256, 6, 128),
            4: Config(Buffer.num_sms, 6, 256, 6, 128),
            8: Config(Buffer.num_sms, 6, 256, 6, 128),
            16: Config(Buffer.num_sms, 36, 288, 20, 128),
            24: Config(Buffer.num_sms, 8, 288, 32, 128),
            32: Config(Buffer.num_sms, 32, 288, 32, 128),
            64: Config(Buffer.num_sms, 20, 288, 28, 128),
            128: Config(Buffer.num_sms, 20, 560, 32, 128),
            144: Config(Buffer.num_sms, 32, 720, 12, 128),
            160: Config(Buffer.num_sms, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
        return config_map[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """Get a recommended combine config."""
        config_map = {
            2: Config(Buffer.num_sms, 10, 256, 6, 128),
            4: Config(Buffer.num_sms, 9, 256, 6, 128),
            8: Config(Buffer.num_sms, 4, 256, 6, 128),
            16: Config(Buffer.num_sms, 4, 288, 12, 128),
            24: Config(Buffer.num_sms, 1, 288, 8, 128),
            32: Config(Buffer.num_sms, 1, 288, 8, 128),
            64: Config(Buffer.num_sms, 1, 288, 20, 128),
            128: Config(Buffer.num_sms, 1, 560, 12, 128),
            144: Config(Buffer.num_sms, 2, 720, 8, 128),
            160: Config(Buffer.num_sms, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
        return config_map[num_ranks]

    # noinspection PyTypeChecker
    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap
    ]:
        """Calculate the layout required for later communication."""
        if allocate_on_comm_stream:
            assert previous_event is not None and async_finish

        alloc_ctx = (
            torch.cuda.stream(self.get_comm_stream())
            if allocate_on_comm_stream
            else nullcontext()
        )
        with alloc_ctx:
            num_tokens_per_rank = torch.empty(
                (self.group_size,), dtype=torch.int, device=topk_idx.device
            )
            num_tokens_per_rdma_rank = (
                torch.empty(
                    (self.runtime.get_num_rdma_ranks(),),
                    dtype=torch.int,
                    device=topk_idx.device,
                )
                if self.runtime.get_num_rdma_ranks() > 1
                else None
            )
            num_tokens_per_expert = torch.empty(
                (num_experts,), dtype=torch.int, device=topk_idx.device
            )
            is_token_in_rank = torch.empty(
                (topk_idx.size(0), self.group_size),
                dtype=torch.bool,
                device=topk_idx.device,
            )

        event = self.runtime.get_dispatch_layout(
            topk_idx.data_ptr(),
            topk_idx.size(0),
            topk_idx.size(1),
            num_experts,
            num_tokens_per_rank.data_ptr(),
            (
                0
                if num_tokens_per_rdma_rank is None
                else num_tokens_per_rdma_rank.data_ptr()
            ),
            num_tokens_per_expert.data_ptr(),
            is_token_in_rank.data_ptr(),
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
            self._ll_compute_stream_ptr(topk_idx.device),
        )
        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            EventOverlap(event),
        )

    # noinspection PyTypeChecker
    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """Dispatch tokens to different ranks."""
        config = self.get_dispatch_config(self.group_size) if config is None else config

        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_dispatch(
                x, handle, num_tokens_per_rank, num_tokens_per_rdma_rank,
                is_token_in_rank, num_tokens_per_expert, topk_idx, topk_weights,
                expert_alignment, num_worst_tokens, config, previous_event,
                async_finish, allocate_on_comm_stream,
            )

        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            (
                rank_prefix_matrix, channel_prefix_matrix,
                recv_channel_prefix_matrix, recv_src_idx,
                is_token_in_rank, send_head,
            ) = handle
            num_recv_tokens = recv_src_idx.size(0)
            num_topk = 0
            num_scales = (
                0 if x_scales is None
                else (1 if x_scales.dim() == 1 else x_scales.size(1))
            )
            scale_token_stride = 0 if x_scales is None else int(x_scales.stride(0))
            scale_hidden_stride = 0 if x_scales is None else int(x_scales.stride(1))
            alloc_ctx = (
                torch.cuda.stream(self.get_comm_stream())
                if allocate_on_comm_stream else nullcontext()
            )
            with alloc_ctx:
                recv_x = torch.empty(
                    (num_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
                )
                recv_x_scales = (
                    None if x_scales is None
                    else torch.empty(
                        ((num_recv_tokens,) if x_scales.dim() == 1
                         else (num_recv_tokens, num_scales)),
                        device=x.device, dtype=x_scales.dtype,
                    )
                )
            event = self.runtime.intranode_dispatch(
                x.data_ptr(), x.size(0), x.size(1), x.element_size(),
                0 if x_scales is None else x_scales.data_ptr(),
                int(num_scales), int(scale_token_stride), int(scale_hidden_stride),
                0, int(num_topk), 0,
                is_token_in_rank.data_ptr(),
                rank_prefix_matrix.data_ptr(),
                channel_prefix_matrix.data_ptr(),
                0, int(num_worst_tokens), True, config, int(num_recv_tokens),
                recv_x.data_ptr(),
                0 if recv_x_scales is None else recv_x_scales.data_ptr(),
                0, 0,
                recv_channel_prefix_matrix.data_ptr(),
                recv_src_idx.data_ptr(), send_head.data_ptr(),
                None, async_finish, allocate_on_comm_stream,
                self._ll_compute_stream_ptr(x.device),
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                None, None, None, None, EventOverlap(event),
            )
        else:
            assert (
                num_tokens_per_rank is not None
                and is_token_in_rank is not None
                and num_tokens_per_expert is not None
            )
            num_channels = int(getattr(config, "num_sms", Buffer.num_sms)) // 2
            rank_prefix_matrix = torch.empty(
                (self.group_size, self.group_size), dtype=torch.int32, device=x.device
            )
            channel_prefix_matrix = torch.empty(
                (self.group_size, num_channels), dtype=torch.int32, device=x.device
            )
            num_recv_tokens, num_recv_tokens_per_expert_list, _ = (
                self.runtime.intranode_prepare(
                    num_tokens_per_rank.data_ptr(),
                    is_token_in_rank.data_ptr(),
                    num_tokens_per_expert.data_ptr(),
                    x.size(0), num_tokens_per_expert.size(0),
                    rank_prefix_matrix.data_ptr(),
                    channel_prefix_matrix.data_ptr(),
                    expert_alignment, num_worst_tokens, config,
                    getattr(previous_event, "event", None),
                    False, False,
                    self._ll_compute_stream_ptr(x.device),
                )
            )
            num_scales = (
                0 if x_scales is None
                else (1 if x_scales.dim() == 1 else x_scales.size(1))
            )
            scale_token_stride = 0 if x_scales is None else int(x_scales.stride(0))
            scale_hidden_stride = 0 if x_scales is None else int(x_scales.stride(1))
            alloc_ctx = (
                torch.cuda.stream(self.get_comm_stream())
                if allocate_on_comm_stream else nullcontext()
            )
            with alloc_ctx:
                recv_x = torch.empty(
                    (num_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
                )
                recv_src_idx = torch.empty(
                    (num_recv_tokens,), dtype=torch.int32, device=x.device
                )
                recv_channel_prefix_matrix = torch.empty(
                    (self.group_size, num_channels), dtype=torch.int32, device=x.device
                )
                send_head = torch.empty(
                    (x.size(0), self.group_size), dtype=torch.int32, device=x.device
                )
                recv_topk_idx = None
                recv_topk_weights = None
                if topk_idx is not None:
                    recv_topk_idx = torch.empty(
                        (num_recv_tokens, topk_idx.size(1)),
                        dtype=topk_idx.dtype, device=x.device,
                    )
                    recv_topk_weights = torch.empty(
                        (num_recv_tokens, topk_weights.size(1)),
                        dtype=topk_weights.dtype, device=x.device,
                    )
                recv_x_scales = (
                    None if x_scales is None
                    else torch.empty(
                        ((num_recv_tokens,) if x_scales.dim() == 1
                         else (num_recv_tokens, num_scales)),
                        device=x.device, dtype=x_scales.dtype,
                    )
                )
            event = self.runtime.intranode_dispatch(
                x.data_ptr(), x.size(0), x.size(1), x.element_size(),
                0 if x_scales is None else x_scales.data_ptr(),
                int(num_scales), int(scale_token_stride), int(scale_hidden_stride),
                0 if topk_idx is None else topk_idx.data_ptr(),
                0 if topk_idx is None else int(topk_idx.size(1)),
                0 if topk_weights is None else topk_weights.data_ptr(),
                is_token_in_rank.data_ptr(),
                rank_prefix_matrix.data_ptr(),
                channel_prefix_matrix.data_ptr(),
                int(num_tokens_per_expert.size(0)),
                int(num_worst_tokens), False, config, int(num_recv_tokens),
                recv_x.data_ptr(),
                0 if recv_x_scales is None else recv_x_scales.data_ptr(),
                0 if recv_topk_idx is None else recv_topk_idx.data_ptr(),
                0 if recv_topk_weights is None else recv_topk_weights.data_ptr(),
                recv_channel_prefix_matrix.data_ptr(),
                recv_src_idx.data_ptr(), send_head.data_ptr(),
                None, async_finish, allocate_on_comm_stream,
                self._ll_compute_stream_ptr(x.device),
            )
            handle = (
                rank_prefix_matrix, channel_prefix_matrix,
                recv_channel_prefix_matrix, recv_src_idx,
                is_token_in_rank, send_head,
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                recv_topk_idx, recv_topk_weights,
                num_recv_tokens_per_expert_list, handle, EventOverlap(event),
            )

    # noinspection PyTypeChecker
    def combine(
        self,
        x: torch.Tensor,
        handle: Tuple,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """Combine (reduce) tokens from different ranks."""
        config = self.get_combine_config(self.group_size) if config is None else config

        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_combine(
                x, handle, topk_weights, bias, config,
                previous_event, async_finish, allocate_on_comm_stream,
            )

        (
            rank_prefix_matrix, _, channel_prefix_matrix,
            src_idx, is_recv_token_in_rank, send_head,
        ) = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        num_recv_tokens = send_head.size(0)
        num_topk = 0 if topk_weights is None else int(topk_weights.size(1))
        alloc_ctx = (
            torch.cuda.stream(self.get_comm_stream())
            if allocate_on_comm_stream else nullcontext()
        )
        with alloc_ctx:
            recv_x = torch.empty(
                (num_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
            )
            recv_topk_weights = (
                None if topk_weights is None
                else torch.empty(
                    (num_recv_tokens, num_topk),
                    device=x.device, dtype=topk_weights.dtype,
                )
            )
        event = self.runtime.intranode_combine(
            x.data_ptr(), x.size(0), x.size(1),
            Buffer._dtype_code(x.dtype), x.element_size(),
            0 if topk_weights is None else topk_weights.data_ptr(),
            num_topk,
            0 if bias_0 is None else bias_0.data_ptr(),
            0 if bias_1 is None else bias_1.data_ptr(),
            src_idx.data_ptr(), num_recv_tokens,
            rank_prefix_matrix.data_ptr(),
            channel_prefix_matrix.data_ptr(),
            send_head.data_ptr(), config,
            recv_x.data_ptr(),
            0 if recv_topk_weights is None else recv_topk_weights.data_ptr(),
            getattr(previous_event, "event", None),
            async_finish, allocate_on_comm_stream,
            self._ll_compute_stream_ptr(x.device),
        )
        return recv_x, recv_topk_weights, EventOverlap(event)

    # noinspection PyTypeChecker
    def internode_dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
        List[int], Tuple, EventOverlap,
    ]:
        """Internode dispatch implementation."""
        assert config is not None

        x, x_scales = x if isinstance(x, tuple) else (x, None)
        num_scales = (
            0 if x_scales is None else (1 if x_scales.dim() == 1 else x_scales.size(1))
        )
        scale_token_stride = 0 if x_scales is None else int(x_scales.stride(0))
        scale_hidden_stride = 0 if x_scales is None else int(x_scales.stride(1))
        num_topk = 0 if topk_idx is None else int(topk_idx.size(1))
        num_rdma_ranks = self.runtime.get_num_rdma_ranks()
        num_channels = int(getattr(config, "num_sms", Buffer.num_sms)) // 2
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            (
                is_token_in_rank,
                rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
                recv_src_meta, send_rdma_head, send_nvl_head,
            ) = handle
            num_recv_tokens = recv_src_meta.size(0)
            num_rdma_recv_tokens = send_nvl_head.size(0)
            alloc_recv_tokens = max(num_recv_tokens, 1)
            alloc_ctx = (
                torch.cuda.stream(self.get_comm_stream())
                if allocate_on_comm_stream else nullcontext()
            )
            with alloc_ctx:
                recv_x = torch.empty(
                    (alloc_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
                )
                recv_x_scales = (
                    None if x_scales is None
                    else torch.empty(
                        ((alloc_recv_tokens,) if x_scales.dim() == 1
                         else (alloc_recv_tokens, num_scales)),
                        device=x.device, dtype=x_scales.dtype,
                    )
                )
            event = self.runtime.internode_dispatch(
                x.data_ptr(), x.size(0), x.size(1), x.element_size(),
                0 if x_scales is None else x_scales.data_ptr(),
                int(num_scales), int(scale_token_stride), int(scale_hidden_stride),
                0, int(num_topk), 0,
                is_token_in_rank.data_ptr(),
                rdma_channel_prefix_matrix.data_ptr(),
                recv_rdma_rank_prefix_sum.data_ptr(),
                gbl_channel_prefix_matrix.data_ptr(),
                recv_gbl_rank_prefix_sum.data_ptr(),
                0, int(num_worst_tokens), True, int(num_rdma_recv_tokens),
                config, recv_x.data_ptr(),
                0 if recv_x_scales is None else recv_x_scales.data_ptr(),
                0, 0, 0, 0, 0, 0, 0,
                getattr(previous_event, "event", None),
                async_finish, allocate_on_comm_stream,
                self._ll_compute_stream_ptr(x.device),
            )
            recv_x = recv_x[:num_recv_tokens]
            if recv_x_scales is not None:
                recv_x_scales = recv_x_scales[:num_recv_tokens]
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                None, None, None, None, EventOverlap(event),
            )
        else:
            assert (
                num_tokens_per_rank is not None
                and num_tokens_per_rdma_rank is not None
                and is_token_in_rank is not None
                and num_tokens_per_expert is not None
            )
            rdma_channel_prefix_matrix = torch.empty(
                (num_rdma_ranks, num_channels), dtype=torch.int32, device=x.device
            )
            recv_rdma_rank_prefix_sum = torch.empty(
                (num_rdma_ranks,), dtype=torch.int32, device=x.device
            )
            gbl_channel_prefix_matrix = torch.empty(
                (self.group_size, num_channels), dtype=torch.int32, device=x.device
            )
            recv_gbl_rank_prefix_sum = torch.empty(
                (self.group_size,), dtype=torch.int32, device=x.device
            )
            (
                num_recv_tokens, num_rdma_recv_tokens,
                num_recv_tokens_per_expert_list, _,
            ) = self.runtime.internode_prepare(
                num_tokens_per_rank.data_ptr(),
                num_tokens_per_rdma_rank.data_ptr(),
                num_tokens_per_expert.data_ptr(),
                is_token_in_rank.data_ptr(),
                x.size(0), x.size(1), x.element_size(),
                int(num_scales), int(num_topk),
                int(num_tokens_per_expert.size(0)),
                int(expert_alignment), int(num_worst_tokens),
                config,
                rdma_channel_prefix_matrix.data_ptr(),
                recv_rdma_rank_prefix_sum.data_ptr(),
                gbl_channel_prefix_matrix.data_ptr(),
                recv_gbl_rank_prefix_sum.data_ptr(),
                getattr(previous_event, "event", None),
                False, False,
                self._ll_compute_stream_ptr(x.device),
            )
            alloc_recv_tokens = max(num_recv_tokens, 1)
            alloc_rdma_recv_tokens = max(num_rdma_recv_tokens, 1)
            alloc_ctx = (
                torch.cuda.stream(self.get_comm_stream())
                if allocate_on_comm_stream else nullcontext()
            )
            with alloc_ctx:
                recv_x = torch.empty(
                    (alloc_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
                )
                recv_x_scales = (
                    None if x_scales is None
                    else torch.empty(
                        ((alloc_recv_tokens,) if x_scales.dim() == 1
                         else (alloc_recv_tokens, num_scales)),
                        device=x.device, dtype=x_scales.dtype,
                    )
                )
                recv_topk_idx = (
                    None if topk_idx is None
                    else torch.empty(
                        (alloc_recv_tokens, topk_idx.size(1)),
                        dtype=topk_idx.dtype, device=x.device,
                    )
                )
                recv_topk_weights = (
                    None if topk_weights is None
                    else torch.empty(
                        (alloc_recv_tokens, topk_weights.size(1)),
                        dtype=topk_weights.dtype, device=x.device,
                    )
                )
                recv_src_meta = torch.empty(
                    (alloc_recv_tokens, self.runtime.get_source_meta_bytes()),
                    dtype=torch.uint8, device=x.device,
                )
                recv_rdma_channel_prefix_matrix = torch.empty(
                    (num_rdma_ranks, num_channels), dtype=torch.int32, device=x.device
                )
                recv_gbl_channel_prefix_matrix = torch.empty(
                    (self.group_size, num_channels), dtype=torch.int32, device=x.device
                )
                send_rdma_head = torch.empty(
                    (x.size(0), num_rdma_ranks), dtype=torch.int32, device=x.device
                )
                send_nvl_head = torch.empty(
                    (alloc_rdma_recv_tokens, self.runtime.get_num_max_nvl_peers()),
                    dtype=torch.int32, device=x.device,
                )
            event = self.runtime.internode_dispatch(
                x.data_ptr(), x.size(0), x.size(1), x.element_size(),
                0 if x_scales is None else x_scales.data_ptr(),
                int(num_scales), int(scale_token_stride), int(scale_hidden_stride),
                0 if topk_idx is None else topk_idx.data_ptr(),
                int(num_topk),
                0 if topk_weights is None else topk_weights.data_ptr(),
                is_token_in_rank.data_ptr(),
                rdma_channel_prefix_matrix.data_ptr(),
                recv_rdma_rank_prefix_sum.data_ptr(),
                gbl_channel_prefix_matrix.data_ptr(),
                recv_gbl_rank_prefix_sum.data_ptr(),
                int(num_tokens_per_expert.size(0)),
                int(num_worst_tokens), False, int(num_rdma_recv_tokens),
                config, recv_x.data_ptr(),
                0 if recv_x_scales is None else recv_x_scales.data_ptr(),
                0 if recv_topk_idx is None else recv_topk_idx.data_ptr(),
                0 if recv_topk_weights is None else recv_topk_weights.data_ptr(),
                recv_src_meta.data_ptr(),
                recv_rdma_channel_prefix_matrix.data_ptr(),
                recv_gbl_channel_prefix_matrix.data_ptr(),
                send_rdma_head.data_ptr(), send_nvl_head.data_ptr(),
                None, async_finish, allocate_on_comm_stream,
                self._ll_compute_stream_ptr(x.device),
            )
            recv_x = recv_x[:num_recv_tokens]
            if recv_x_scales is not None:
                recv_x_scales = recv_x_scales[:num_recv_tokens]
            if recv_topk_idx is not None:
                recv_topk_idx = recv_topk_idx[:num_recv_tokens]
            if recv_topk_weights is not None:
                recv_topk_weights = recv_topk_weights[:num_recv_tokens]
            recv_src_meta = recv_src_meta[:num_recv_tokens]
            send_nvl_head = send_nvl_head[: max(num_rdma_recv_tokens, 1)]
            handle = (
                is_token_in_rank,
                rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
                recv_src_meta, send_rdma_head, send_nvl_head,
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                recv_topk_idx, recv_topk_weights,
                num_recv_tokens_per_expert_list, handle, EventOverlap(event),
            )

    # noinspection PyTypeChecker
    def internode_combine(
        self,
        x: torch.Tensor,
        handle: Union[tuple, list],
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """Internode combine implementation."""
        assert config is not None

        (
            is_combined_token_in_rank, _, _,
            rdma_channel_prefix_matrix, rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix, gbl_rank_prefix_sum,
            src_meta, send_rdma_head, send_nvl_head,
        ) = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        num_combined_tokens = int(is_combined_token_in_rank.size(0))
        num_topk = 0 if topk_weights is None else int(topk_weights.size(1))

        alloc_combined_tokens = max(num_combined_tokens, 1)
        num_x_rows = x.size(0)
        if num_x_rows == 0:
            x = x.new_empty((1, x.size(1)))
        if src_meta.size(0) == 0:
            src_meta = src_meta.new_empty((1,) + src_meta.shape[1:])

        alloc_ctx = (
            torch.cuda.stream(self.get_comm_stream())
            if allocate_on_comm_stream else nullcontext()
        )
        with alloc_ctx:
            combined_x = torch.empty(
                (alloc_combined_tokens, x.size(1)), device=x.device, dtype=x.dtype
            )
            combined_topk_weights = (
                None if topk_weights is None
                else torch.empty(
                    (alloc_combined_tokens, num_topk),
                    device=x.device, dtype=topk_weights.dtype,
                )
            )
        event = self.runtime.internode_combine(
            x.data_ptr(), num_x_rows, x.size(1),
            Buffer._dtype_code(x.dtype), x.element_size(),
            0 if topk_weights is None else topk_weights.data_ptr(),
            num_topk,
            0 if bias_0 is None else bias_0.data_ptr(),
            0 if bias_1 is None else bias_1.data_ptr(),
            src_meta.data_ptr(), num_combined_tokens,
            is_combined_token_in_rank.data_ptr(),
            rdma_channel_prefix_matrix.data_ptr(),
            rdma_rank_prefix_sum.data_ptr(),
            gbl_channel_prefix_matrix.data_ptr(),
            send_rdma_head.data_ptr(), send_nvl_head.data_ptr(),
            config, combined_x.data_ptr(),
            0 if combined_topk_weights is None else combined_topk_weights.data_ptr(),
            getattr(previous_event, "event", None),
            async_finish, allocate_on_comm_stream,
            self._ll_compute_stream_ptr(x.device),
        )
        combined_x = combined_x[:num_combined_tokens]
        if combined_topk_weights is not None:
            combined_topk_weights = combined_topk_weights[:num_combined_tokens]
        return combined_x, combined_topk_weights, EventOverlap(event)

    def clean_low_latency_buffer(
        self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int
    ) -> None:
        """
        Clean the low-latency buffer (zero-initialize).

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch.
            hidden: the hidden dimension of each token.
            num_experts: the number of all experts.
        """
        compute_stream_ptr = self._ll_compute_stream_ptr(torch.device("cuda"))
        self.runtime.clean_low_latency_buffer(
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
            compute_stream_ptr,
        )
