import os
from contextlib import nullcontext
import torch
import torch.distributed as dist
from typing import Callable, Tuple, Optional, Union, List

try:
    from uccl import ep
except ImportError as exc:
    import sys

    sys.stderr.write("Failed to import uccl.ep\n")
    raise

from uccl.ep import EventHandle, Config

# Support both execution modes:
# 1) As part of the packaged deep_ep_wrapper (symlinked buffer inside a package): uses relative import `.utils`.
# 2) As a standalone benchmark script from the `ep/bench` directory (no package): falls back to plain `utils`.
try:
    from .utils import (
        EventOverlap,
        check_nvlink_connections,
        initialize_uccl,
        destroy_uccl,
        _fp8_e4m3_dtype,
    )
except ImportError:
    from utils import (
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

    # TODO(MaoZiming): Reduce SMs. UCCL Proxy should reduce the usage of SMs.
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

        if hasattr(ep, "get_rdma_buffer"):
            # Allocate outside PyTorch's CUDA allocator so RDMA/IPC sees a raw
            # cudaMalloc/cudaMallocHost-style allocation instead of a possibly
            # segmented caching-allocator mapping.
            scratch_dlpack, rdma_buffer_is_host_allocated = ep.get_rdma_buffer(
                num_rdma_bytes, device_index
            )
            self.scratch = torch.utils.dlpack.from_dlpack(scratch_dlpack)
        else:
            rdma_buffer_is_host_allocated = False
            if num_rdma_bytes > 0:
                if hasattr(ep, "can_register_rdma_gpu_buffer"):
                    rdma_buffer_is_host_allocated = not bool(
                        ep.can_register_rdma_gpu_buffer(device_index, num_rdma_bytes)
                    )
                elif hasattr(ep, "rdma_buffer_should_use_host_alloc"):
                    rdma_buffer_is_host_allocated = bool(
                        ep.rdma_buffer_should_use_host_alloc(
                            device_index, num_rdma_bytes
                        )
                    )

            if num_rdma_bytes > 0 and rdma_buffer_is_host_allocated:
                # Host-pinned fallback for platforms/NICs that cannot register GPU memory.
                self.scratch = torch.zeros(
                    (num_rdma_bytes,),
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )
            else:
                # Device buffer for normal RDMA path. Keep a valid pointer even when RDMA is disabled.
                self.scratch = torch.zeros(
                    max(num_rdma_bytes, 1),
                    dtype=torch.uint8,
                    device=f"cuda:{device_index}",
                )

        rdma_buffer_ptr = self.scratch.data_ptr()
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

        # Initialize the CPP runtime
        self.rank = group.rank()
        self.group_size = group.size()
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.explicitly_destroy = explicitly_destroy
        self._next_low_latency_combine_buffer = None
        self.runtime = ep.Buffer(
            self.rank,
            self.group_size,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode,
            explicitly_destroy,
            int(os.environ.get("LOCAL_WORLD_SIZE", -1)),
        )
        if num_rdma_bytes:
            self.runtime.set_rdma_buffer(rdma_buffer_ptr, rdma_buffer_is_host_allocated)

        # Synchronize device IDs
        device_ids = [
            None,
        ] * self.group_size
        local_device_id = self.runtime.get_local_device_id()
        # print("Before all_gather_object device_ids", local_device_id, flush=True)
        dist.all_gather_object(device_ids, local_device_id, group)
        # Synchronize IPC handles
        ipc_handles = [
            None,
        ] * self.group_size
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        # print("Before all_gather_object ipc_handles", local_ipc_handle, flush=True)
        dist.all_gather_object(ipc_handles, local_ipc_handle, group)

        rdma_ipc_handles = [None] * self.group_size
        # CUDA IPC only works with device memory; skip when using cudaHostAlloc.
        local_rdma_ipc_handle = (
            self.runtime.get_local_rdma_ipc_handle()
            if self.num_rdma_bytes > 0 and not rdma_buffer_is_host_allocated
            else None
        )
        dist.all_gather_object(rdma_ipc_handles, local_rdma_ipc_handle, group)
        root_unique_id = None
        # Make CPP runtime available
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
        """
        Return the current CUDA stream pointer for low-latency runtime calls.
        """
        current = torch.cuda.current_stream(device=device)
        return int(current.cuda_stream)

    def reset_rdma_buffer(self):
        """
        Reset the RDMA buffer, this is useful when you want to reuse the RDMA buffer for another run.

        """
        self.runtime.reset_rdma_buffer()

    def connect_atomic_buffer(self, proxy: "ep.UcclProxy"):
        ep.connect_atomic_buffer(proxy, self.runtime)

    def destroy(self):
        """
        Destroy the cpp runtime and release resources.

        """

        assert self.explicitly_destroy, "`explicitly_destroy` flag must be set"

        self.runtime.destroy()
        self.runtime = None
        destroy_uccl(self.proxies, self.workers)

    @staticmethod
    def is_sm90_compiled():
        return ep.is_sm90_compiled()

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
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

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
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            low-latency kernels' result tensors at a single moment.

        Arguments:
            x: `torch.Tensor` with `torch.bfloat16`, shaped as `[num_tokens, hidden]`, only several hidden shapes are
                supported. The number of tokens to be dispatched must be less than `num_max_dispatch_tokens_per_rank`.
            topk_idx: `torch.Tensor` with `torch.int64`, shaped as `[num_tokens, num_topk]`, only several top-k shapes
                are supported. `-1` indices (not selecting any expert) are supported.
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            num_experts: the number of all experts.
            cumulative_local_expert_recv_stats: a cumulative expert count tensor for statistics, which should have shape
                `[num_local_experts]` and be typed as `torch.int`. This is useful for online service EP load balance
                monitoring.
            dispatch_wait_recv_cost_stats: a cumulative time spent waiting to receive each token tensor for statistics,
                which should have shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
                This is useful for detecting and pre-cisely localizing slow anomalies.
            use_fp8: whether to enable FP8 casting, with this, the received data will be a tuple of FP8 tensor and scaling factors.
            round_scale: whether round the scaling factors into power of 2.
            use_ue8m0: whether use UE8M0 as scaling factor format (available only with `round_scale=True`).
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.

        Returns:
            recv_x: a tensor or tuple with received tokens for each expert.
                With `use_fp8=True`: the first element is a `torch.Tensor` shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.float8_e4m3fn`.
                The second tensor is the corresponding scales for the first element with shape
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 128]` with `torch.float`,
                if `use_ue8m0=False`. With `use_ue8m0=True`, the second one is packed and shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 512]` with type `torch.int`.
                Notice that, the last-two-dimension of the scaling tensors are in column-major for TMA compatibility.
                With `use_fp8=False`, the result would be a tensor shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`.
                Moreover, not all tokens are valid, only some of the `num_max_dispatch_tokens_per_rank * num_ranks` are,
                as we do not synchronize CPU received count with GPU (also not incompatible with CUDA graph if synced).
            recv_count: a tensor shaped `[num_local_experts]` with type `torch.int`, indicating how many tokens each
                expert receives. As mentioned before, not all tokens are valid in `recv_x`.
            handle: the communication handle to be used in the `low_latency_combine` function.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        for proxy in self.proxies:
            proxy.calculate_and_set_dispatch_recv_data_offset(
                num_tokens=x.shape[0],
                hidden=x.shape[1],
                num_experts=num_experts,
            )
        # On HIP platform, force return_recv_hook=True for correct async behavior
        is_hip = torch.version.hip is not None
        orig_return_recv_hook = return_recv_hook
        orig_async_finish = async_finish
        if is_hip and not return_recv_hook:
            return_recv_hook = True
            # HIP runtime doesn't support async and return_recv_hook together,
            # so disable async when forcing return_recv_hook
            async_finish = False
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
        # On HIP platform, if original return_recv_hook was False, call hook now
        # to ensure data arrival and return a no-op hook to the caller
        if is_hip and not orig_return_recv_hook:
            if hook is not None:
                hook()
            hook = None
        return (
            (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x,
            packed_recv_count,
            handle,
            EventOverlap(event, tensors_to_record if orig_async_finish else None),
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
        A low-latency implementation for combining tokens (reduce **with weights**) with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            low-latency kernels' result tensors at a single moment.

        Arguments:
            x: `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`,
                the local calculated tokens to be sent to this original rank and reduced.
            topk_idx: `[num_combined_tokens, num_topk]` with `torch.int64`, the expert indices selected by the dispatched
                tokens. `-1` indices (not selecting any expert) are supported. Note that, `num_combined_tokens` equals
                to the number of dispatched tokens.
            topk_weights: `[num_combined_tokens, num_topk]` with `torch.float`, the expert weights selected by the dispatched
                tokens. The received tokens will be reduced with the weights in this tensor.
            handle: the communication handle given by the `dispatch` function.
            use_logfmt: whether to use an internal "LogFMT with dynamic per-64-channel cast" format (10 bits).
            zero_copy: whether the tensor is already copied into the RDMA buffer, should be cooperative
                with `get_next_low_latency_combine_buffer`.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.
            out: the in-place output tensor, if set, the kernel will write the result to this tensor and return it directly.
            combine_wait_recv_cost_stats: a cumulative time spent waiting to receive each token tensor for statistics,
                which should have shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
                This is useful for detecting and pre-cisely localizing slow anomalies.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
        ) = handle
        # On HIP platform, force return_recv_hook=True for correct async behavior
        is_hip = torch.version.hip is not None
        orig_return_recv_hook = return_recv_hook
        orig_async_finish = async_finish
        if is_hip and not return_recv_hook:
            return_recv_hook = True
            # HIP runtime doesn't support async and return_recv_hook together,
            # so disable async when forcing return_recv_hook
            async_finish = False
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
        # On HIP platform, if original return_recv_hook was False, call hook now
        # to ensure data arrival and return a no-op hook to the caller
        if is_hip and not orig_return_recv_hook:
            if hook is not None:
                hook()
            hook = None
        return (
            combined_x,
            EventOverlap(event, tensors_to_record if orig_async_finish else None),
            hook,
        )

    def get_next_low_latency_combine_buffer(self, handle: object):
        """
        Get the raw registered RDMA buffer tensor for next low-latency combine, so that the next combine kernel can skip the copying.

        Arguments:
            handle: the communication handle given by the `dispatch` function.

        Returns:
            buffer: the raw RDMA low-latency buffer as a BF16 PyTorch tensor with shape
                `[num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden]`, you should fill this buffer
                by yourself.
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
        Get a minimum size requirement for the RDMA buffer. The size calculation will be done with BF16.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_ranks: the number of EP group ranks.
            num_experts: the number of all experts.

        Returns:
            size: the RDMA buffer size recommended.
        """
        return ep.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts
        )

    def get_comm_stream(self) -> torch.Stream:
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.
        """
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
            dtype: the data type (PyTorch `dtype`) for the tensor.
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
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
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
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
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
        """
        Calculate the layout required for later communication.

        Arguments:
            topk_idx: `[num_tokens, num_topk]`, dtype must be `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            num_experts: the number of experts.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
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
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync, and it
                will be CUDA-graph compatible. Please also notice that this flag is for intranode only.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
                received token count.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
                will be empty.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_dispatch_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_dispatch(
                x,
                handle,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                topk_idx,
                topk_weights,
                expert_alignment,
                num_worst_tokens,
                config,
                previous_event,
                async_finish,
                allocate_on_comm_stream,
            )

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            (
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                is_token_in_rank,
                send_head,
            ) = handle
            num_recv_tokens = recv_src_idx.size(0)
            num_topk = 0
            num_scales = (
                0
                if x_scales is None
                else (1 if x_scales.dim() == 1 else x_scales.size(1))
            )
            scale_token_stride = 0 if x_scales is None else int(x_scales.stride(0))
            scale_hidden_stride = 0 if x_scales is None else int(x_scales.stride(1))
            alloc_ctx = (
                torch.cuda.stream(self.get_comm_stream())
                if allocate_on_comm_stream
                else nullcontext()
            )
            with alloc_ctx:
                recv_x = torch.empty(
                    (num_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
                )
                recv_x_scales = (
                    None
                    if x_scales is None
                    else torch.empty(
                        (
                            (num_recv_tokens,)
                            if x_scales.dim() == 1
                            else (num_recv_tokens, num_scales)
                        ),
                        device=x.device,
                        dtype=x_scales.dtype,
                    )
                )
            event = self.runtime.intranode_dispatch(
                x.data_ptr(),
                x.size(0),
                x.size(1),
                x.element_size(),
                0 if x_scales is None else x_scales.data_ptr(),
                int(num_scales),
                int(scale_token_stride),
                int(scale_hidden_stride),
                0,
                int(num_topk),
                0,
                is_token_in_rank.data_ptr(),
                rank_prefix_matrix.data_ptr(),
                channel_prefix_matrix.data_ptr(),
                0,
                int(num_worst_tokens),
                True,
                config,
                int(num_recv_tokens),
                recv_x.data_ptr(),
                0 if recv_x_scales is None else recv_x_scales.data_ptr(),
                0,
                0,
                recv_channel_prefix_matrix.data_ptr(),
                recv_src_idx.data_ptr(),
                send_head.data_ptr(),
                None,
                async_finish,
                allocate_on_comm_stream,
                self._ll_compute_stream_ptr(x.device),
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                None,
                None,
                None,
                None,
                EventOverlap(event),
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
                    x.size(0),
                    num_tokens_per_expert.size(0),
                    rank_prefix_matrix.data_ptr(),
                    channel_prefix_matrix.data_ptr(),
                    expert_alignment,
                    num_worst_tokens,
                    config,
                    getattr(previous_event, "event", None),
                    False,
                    False,
                    self._ll_compute_stream_ptr(x.device),
                )
            )
            num_scales = (
                0
                if x_scales is None
                else (1 if x_scales.dim() == 1 else x_scales.size(1))
            )
            scale_token_stride = 0 if x_scales is None else int(x_scales.stride(0))
            scale_hidden_stride = 0 if x_scales is None else int(x_scales.stride(1))
            alloc_ctx = (
                torch.cuda.stream(self.get_comm_stream())
                if allocate_on_comm_stream
                else nullcontext()
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
                        dtype=topk_idx.dtype,
                        device=x.device,
                    )
                    recv_topk_weights = torch.empty(
                        (num_recv_tokens, topk_weights.size(1)),
                        dtype=topk_weights.dtype,
                        device=x.device,
                    )
                recv_x_scales = (
                    None
                    if x_scales is None
                    else torch.empty(
                        (
                            (num_recv_tokens,)
                            if x_scales.dim() == 1
                            else (num_recv_tokens, num_scales)
                        ),
                        device=x.device,
                        dtype=x_scales.dtype,
                    )
                )
            event = self.runtime.intranode_dispatch(
                x.data_ptr(),
                x.size(0),
                x.size(1),
                x.element_size(),
                0 if x_scales is None else x_scales.data_ptr(),
                int(num_scales),
                int(scale_token_stride),
                int(scale_hidden_stride),
                0 if topk_idx is None else topk_idx.data_ptr(),
                0 if topk_idx is None else int(topk_idx.size(1)),
                0 if topk_weights is None else topk_weights.data_ptr(),
                is_token_in_rank.data_ptr(),
                rank_prefix_matrix.data_ptr(),
                channel_prefix_matrix.data_ptr(),
                int(num_tokens_per_expert.size(0)),
                int(num_worst_tokens),
                False,
                config,
                int(num_recv_tokens),
                recv_x.data_ptr(),
                0 if recv_x_scales is None else recv_x_scales.data_ptr(),
                0 if recv_topk_idx is None else recv_topk_idx.data_ptr(),
                0 if recv_topk_weights is None else recv_topk_weights.data_ptr(),
                recv_channel_prefix_matrix.data_ptr(),
                recv_src_idx.data_ptr(),
                send_head.data_ptr(),
                None,
                async_finish,
                allocate_on_comm_stream,
                self._ll_compute_stream_ptr(x.device),
            )
            handle = (
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                is_token_in_rank,
                send_head,
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                EventOverlap(event),
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
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_combine_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_combine(
                x,
                handle,
                topk_weights,
                bias,
                config,
                previous_event,
                async_finish,
                allocate_on_comm_stream,
            )

        # NOTES: the second `_` is for the sending side, so we should use the third one
        (
            rank_prefix_matrix,
            _,
            channel_prefix_matrix,
            src_idx,
            is_recv_token_in_rank,
            send_head,
        ) = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        # Launch the kernel
        num_recv_tokens = send_head.size(0)
        num_topk = 0 if topk_weights is None else int(topk_weights.size(1))
        alloc_ctx = (
            torch.cuda.stream(self.get_comm_stream())
            if allocate_on_comm_stream
            else nullcontext()
        )
        with alloc_ctx:
            recv_x = torch.empty(
                (num_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
            )
            recv_topk_weights = (
                None
                if topk_weights is None
                else torch.empty(
                    (num_recv_tokens, num_topk),
                    device=x.device,
                    dtype=topk_weights.dtype,
                )
            )
        event = self.runtime.intranode_combine(
            x.data_ptr(),
            x.size(0),
            x.size(1),
            Buffer._dtype_code(x.dtype),
            x.element_size(),
            0 if topk_weights is None else topk_weights.data_ptr(),
            num_topk,
            0 if bias_0 is None else bias_0.data_ptr(),
            0 if bias_1 is None else bias_1.data_ptr(),
            src_idx.data_ptr(),
            num_recv_tokens,
            rank_prefix_matrix.data_ptr(),
            channel_prefix_matrix.data_ptr(),
            send_head.data_ptr(),
            config,
            recv_x.data_ptr(),
            0 if recv_topk_weights is None else recv_topk_weights.data_ptr(),
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
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
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """
        Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.
        """
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
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
            ) = handle
            num_recv_tokens = recv_src_meta.size(0)
            num_rdma_recv_tokens = send_nvl_head.size(0)
            alloc_ctx = (
                torch.cuda.stream(self.get_comm_stream())
                if allocate_on_comm_stream
                else nullcontext()
            )
            with alloc_ctx:
                recv_x = torch.empty(
                    (num_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
                )
                recv_x_scales = (
                    None
                    if x_scales is None
                    else torch.empty(
                        (
                            (num_recv_tokens,)
                            if x_scales.dim() == 1
                            else (num_recv_tokens, num_scales)
                        ),
                        device=x.device,
                        dtype=x_scales.dtype,
                    )
                )
            event = self.runtime.internode_dispatch(
                x.data_ptr(),
                x.size(0),
                x.size(1),
                x.element_size(),
                0 if x_scales is None else x_scales.data_ptr(),
                int(num_scales),
                int(scale_token_stride),
                int(scale_hidden_stride),
                0,
                int(num_topk),
                0,
                is_token_in_rank.data_ptr(),
                rdma_channel_prefix_matrix.data_ptr(),
                recv_rdma_rank_prefix_sum.data_ptr(),
                gbl_channel_prefix_matrix.data_ptr(),
                recv_gbl_rank_prefix_sum.data_ptr(),
                0,
                int(num_worst_tokens),
                True,
                int(num_rdma_recv_tokens),
                config,
                recv_x.data_ptr(),
                0 if recv_x_scales is None else recv_x_scales.data_ptr(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                getattr(previous_event, "event", None),
                async_finish,
                allocate_on_comm_stream,
                self._ll_compute_stream_ptr(x.device),
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                None,
                None,
                None,
                None,
                EventOverlap(event),
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
                num_recv_tokens,
                num_rdma_recv_tokens,
                num_recv_tokens_per_expert_list,
                _,
            ) = self.runtime.internode_prepare(
                num_tokens_per_rank.data_ptr(),
                num_tokens_per_rdma_rank.data_ptr(),
                num_tokens_per_expert.data_ptr(),
                is_token_in_rank.data_ptr(),
                x.size(0),
                x.size(1),
                x.element_size(),
                int(num_scales),
                int(num_topk),
                int(num_tokens_per_expert.size(0)),
                int(expert_alignment),
                int(num_worst_tokens),
                config,
                rdma_channel_prefix_matrix.data_ptr(),
                recv_rdma_rank_prefix_sum.data_ptr(),
                gbl_channel_prefix_matrix.data_ptr(),
                recv_gbl_rank_prefix_sum.data_ptr(),
                getattr(previous_event, "event", None),
                False,
                False,
                self._ll_compute_stream_ptr(x.device),
            )
            alloc_ctx = (
                torch.cuda.stream(self.get_comm_stream())
                if allocate_on_comm_stream
                else nullcontext()
            )
            with alloc_ctx:
                recv_x = torch.empty(
                    (num_recv_tokens, x.size(1)), device=x.device, dtype=x.dtype
                )
                recv_x_scales = (
                    None
                    if x_scales is None
                    else torch.empty(
                        (
                            (num_recv_tokens,)
                            if x_scales.dim() == 1
                            else (num_recv_tokens, num_scales)
                        ),
                        device=x.device,
                        dtype=x_scales.dtype,
                    )
                )
                recv_topk_idx = (
                    None
                    if topk_idx is None
                    else torch.empty(
                        (num_recv_tokens, topk_idx.size(1)),
                        dtype=topk_idx.dtype,
                        device=x.device,
                    )
                )
                recv_topk_weights = (
                    None
                    if topk_weights is None
                    else torch.empty(
                        (num_recv_tokens, topk_weights.size(1)),
                        dtype=topk_weights.dtype,
                        device=x.device,
                    )
                )
                recv_src_meta = torch.empty(
                    (num_recv_tokens, self.runtime.get_source_meta_bytes()),
                    dtype=torch.uint8,
                    device=x.device,
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
                    (num_rdma_recv_tokens, self.runtime.get_num_max_nvl_peers()),
                    dtype=torch.int32,
                    device=x.device,
                )
            event = self.runtime.internode_dispatch(
                x.data_ptr(),
                x.size(0),
                x.size(1),
                x.element_size(),
                0 if x_scales is None else x_scales.data_ptr(),
                int(num_scales),
                int(scale_token_stride),
                int(scale_hidden_stride),
                0 if topk_idx is None else topk_idx.data_ptr(),
                int(num_topk),
                0 if topk_weights is None else topk_weights.data_ptr(),
                is_token_in_rank.data_ptr(),
                rdma_channel_prefix_matrix.data_ptr(),
                recv_rdma_rank_prefix_sum.data_ptr(),
                gbl_channel_prefix_matrix.data_ptr(),
                recv_gbl_rank_prefix_sum.data_ptr(),
                int(num_tokens_per_expert.size(0)),
                int(num_worst_tokens),
                False,
                int(num_rdma_recv_tokens),
                config,
                recv_x.data_ptr(),
                0 if recv_x_scales is None else recv_x_scales.data_ptr(),
                0 if recv_topk_idx is None else recv_topk_idx.data_ptr(),
                0 if recv_topk_weights is None else recv_topk_weights.data_ptr(),
                recv_src_meta.data_ptr(),
                recv_rdma_channel_prefix_matrix.data_ptr(),
                recv_gbl_channel_prefix_matrix.data_ptr(),
                send_rdma_head.data_ptr(),
                send_nvl_head.data_ptr(),
                None,
                async_finish,
                allocate_on_comm_stream,
                self._ll_compute_stream_ptr(x.device),
            )
            handle = (
                is_token_in_rank,
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
            )
            return (
                (recv_x, recv_x_scales) if x_scales is not None else recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                EventOverlap(event),
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
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.
        """
        assert config is not None

        # Unpack handle and bias
        (
            is_combined_token_in_rank,
            _,
            _,
            rdma_channel_prefix_matrix,
            rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix,
            gbl_rank_prefix_sum,
            src_meta,
            send_rdma_head,
            send_nvl_head,
        ) = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        num_combined_tokens = int(is_combined_token_in_rank.size(0))
        num_topk = 0 if topk_weights is None else int(topk_weights.size(1))
        alloc_ctx = (
            torch.cuda.stream(self.get_comm_stream())
            if allocate_on_comm_stream
            else nullcontext()
        )
        with alloc_ctx:
            combined_x = torch.empty(
                (num_combined_tokens, x.size(1)), device=x.device, dtype=x.dtype
            )
            combined_topk_weights = (
                None
                if topk_weights is None
                else torch.empty(
                    (num_combined_tokens, num_topk),
                    device=x.device,
                    dtype=topk_weights.dtype,
                )
            )
        event = self.runtime.internode_combine(
            x.data_ptr(),
            x.size(0),
            x.size(1),
            Buffer._dtype_code(x.dtype),
            x.element_size(),
            0 if topk_weights is None else topk_weights.data_ptr(),
            num_topk,
            0 if bias_0 is None else bias_0.data_ptr(),
            0 if bias_1 is None else bias_1.data_ptr(),
            src_meta.data_ptr(),
            num_combined_tokens,
            is_combined_token_in_rank.data_ptr(),
            rdma_channel_prefix_matrix.data_ptr(),
            rdma_rank_prefix_sum.data_ptr(),
            gbl_channel_prefix_matrix.data_ptr(),
            send_rdma_head.data_ptr(),
            send_nvl_head.data_ptr(),
            config,
            combined_x.data_ptr(),
            0 if combined_topk_weights is None else combined_topk_weights.data_ptr(),
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
            self._ll_compute_stream_ptr(x.device),
        )
        return combined_x, combined_topk_weights, EventOverlap(event)

    def clean_low_latency_buffer(
        self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int
    ) -> None:
        """
        As low-latency kernels require part of the buffer to be zero-initialized, so it is vital to clean the buffer
            if the buffer is dirty at some time.
        For example, after running the normal dispatch/combine, you must run this function before executing any
            low-latency kernel.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
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
