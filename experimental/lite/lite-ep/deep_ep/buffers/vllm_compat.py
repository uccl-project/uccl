from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .elastic import EPHandle, ElasticBuffer
from ..utils.event import EventHandle, EventOverlap


@dataclass(frozen=True)
class Config:
    num_sms: int = 0


class Buffer:
    """vLLM-facing DeepEP v1 compatibility shim backed by ElasticBuffer.

    This intentionally implements only the high-throughput Buffer surface used by
    vLLM's `deepep_high_throughput` prepare/finalize path. DeepEP v1 low-latency
    APIs have different layout semantics and are not provided by Lite-EP v2.
    """

    _num_sms: int = 0

    def __init__(self,
                 group: dist.ProcessGroup,
                 num_nvl_bytes: int = 0,
                 num_rdma_bytes: Optional[int] = None,
                 low_latency_mode: bool = False,
                 num_qps_per_rank: int = 1,
                 explicitly_destroy: bool = False,
                 **_: Any) -> None:
        if low_latency_mode:
            raise NotImplementedError(
                'Lite-EP DeepEPv2 exposes ElasticBuffer dispatch/combine only; '
                'vLLM deepep_low_latency requires DeepEP v1 low_latency_* APIs.'
            )
        self.group = group
        self.device_group = self._resolve_device_group(group)
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        if int(os.environ.get('EP_FORCE_NO_NVLINK', '0')):
            self.num_qps_per_rank = 1
        else:
            self.num_qps_per_rank = max(1, int(num_qps_per_rank or 1))
        self.explicitly_destroy = explicitly_destroy
        self._buffer: Optional[ElasticBuffer] = None
        self._buffer_key: Optional[Tuple[int, int, int, bool]] = None
        self._pending_num_experts: Optional[int] = None

    @staticmethod
    def _resolve_device_group(group: dist.ProcessGroup) -> dist.ProcessGroup:
        try:
            from vllm.distributed.parallel_state import get_ep_group

            ep_group = get_ep_group()
            if getattr(ep_group, 'cpu_group', None) is group:
                device_group = getattr(ep_group, 'device_group', None)
                if device_group is not None:
                    return device_group
        except Exception:
            pass
        return group

    @classmethod
    def set_num_sms(cls, num_sms: int) -> None:
        cls._num_sms = max(0, int(num_sms))

    @classmethod
    def get_dispatch_config(cls, *_: Any, **__: Any) -> Config:
        return Config(num_sms=cls._num_sms)

    @classmethod
    def get_combine_config(cls, *_: Any, **__: Any) -> Config:
        return Config(num_sms=cls._num_sms)

    @staticmethod
    def get_low_latency_rdma_size_hint(*_: Any, **__: Any) -> int:
        raise NotImplementedError(
            'Lite-EP DeepEPv2 does not implement DeepEP v1 low-latency RDMA sizing; '
            'use vLLM deepep_high_throughput for the ElasticBuffer shim.'
        )

    @staticmethod
    def capture() -> EventHandle:
        return ElasticBuffer.capture()

    def _world_size(self) -> int:
        if hasattr(self.device_group, 'size'):
            return self.device_group.size()
        return dist.get_world_size(self.device_group)

    def _max_tokens_per_rank(self, num_tokens: int, device: torch.device) -> int:
        if not dist.is_available() or not dist.is_initialized() or \
                self._world_size() == 1:
            return num_tokens
        value = torch.tensor([num_tokens], dtype=torch.int32, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=self.device_group)
        return int(value.item())

    def _get_or_create_buffer(self,
                              x: Union[torch.Tensor,
                                       Tuple[torch.Tensor, torch.Tensor]],
                              topk_idx: torch.Tensor,
                              num_experts: int) -> ElasticBuffer:
        token_tensor = x[0] if isinstance(x, tuple) else x
        num_tokens, hidden = token_tensor.shape
        num_topk = topk_idx.shape[1]
        use_fp8_dispatch = isinstance(x, tuple)
        num_max_tokens_per_rank = self._max_tokens_per_rank(num_tokens, token_tensor.device)
        key = (num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch)
        if self._buffer is not None and self._buffer_key == key:
            return self._buffer
        if self._buffer is not None:
            self._buffer.destroy()
        self._buffer = ElasticBuffer(
            self.device_group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden,
            num_topk=num_topk,
            use_fp8_dispatch=use_fp8_dispatch,
            num_allocated_qps=self.num_qps_per_rank,
            explicitly_destroy=True,
        )
        self._buffer_key = key
        return self._buffer

    def get_dispatch_layout(self,
                            topk_idx: torch.Tensor,
                            num_experts: int,
                            previous_event: Optional[EventHandle] = None,
                            async_finish: bool = False,
                            allocate_on_comm_stream: bool = False,
                            **_: Any) -> Tuple[torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor,
                                               EventOverlap]:
        del previous_event, async_finish, allocate_on_comm_stream
        self._pending_num_experts = num_experts
        num_ranks = self._world_size()
        device = topk_idx.device
        num_tokens_per_rank = torch.zeros((num_ranks,), dtype=torch.int32,
                          device=device)
        num_tokens_per_rdma_rank = torch.zeros((num_ranks,), dtype=torch.int32,
                               device=device)
        num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int32,
                            device=device)
        is_token_in_rank = torch.zeros((topk_idx.shape[0], num_ranks),
                           dtype=torch.bool, device=device)
        return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, EventOverlap()

    def dispatch(self,
                 x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 handle: Optional[EPHandle] = None,
                 num_tokens_per_rank: Optional[torch.Tensor] = None,
                 num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                 is_token_in_rank: Optional[torch.Tensor] = None,
                 num_tokens_per_expert: Optional[torch.Tensor] = None,
                 topk_idx: Optional[torch.Tensor] = None,
                 topk_weights: Optional[torch.Tensor] = None,
                 expert_alignment: int = 1,
                 config: Optional[Config] = None,
                 previous_event: Optional[EventHandle] = None,
                 async_finish: bool = False,
                 allocate_on_comm_stream: bool = False,
                 **_: Any) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                                    Optional[torch.Tensor], Optional[torch.Tensor],
                                    list, EPHandle, EventOverlap]:
        del num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank
        del num_tokens_per_expert
        if handle is None and topk_idx is None:
            raise ValueError('topk_idx is required when no cached handle is provided')
        if handle is not None:
            buffer = self._buffer
            if buffer is None:
                raise ValueError(
                    'cached dispatch handle cannot be used before Buffer has '
                    'created ElasticBuffer'
                )
            num_experts = handle.num_experts
        else:
            assert topk_idx is not None
            if self._pending_num_experts is None:
                raise ValueError(
                    'get_dispatch_layout must be called before dispatch so '
                    'num_experts is known'
                )
            num_experts = self._pending_num_experts
            buffer = self._get_or_create_buffer(x, topk_idx, num_experts)

        num_sms = self._num_sms if config is None else config.num_sms
        recv_x, recv_topk_idx, recv_topk_weights, ep_handle, event = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=num_experts,
            expert_alignment=expert_alignment,
            num_sms=num_sms,
            num_qps=self.num_qps_per_rank,
            previous_event=previous_event,
            async_with_compute_stream=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            handle=handle,
            do_handle_copy=True,
            do_cpu_sync=True if handle is None else False,
        )
        return (recv_x, recv_topk_idx, recv_topk_weights,
            ep_handle.num_recv_tokens_per_expert_list, ep_handle, event)

    def combine(self,
                x: torch.Tensor,
                handle: EPHandle,
                topk_weights: Optional[torch.Tensor] = None,
                config: Optional[Config] = None,
                previous_event: Optional[EventHandle] = None,
                async_finish: bool = False,
                allocate_on_comm_stream: bool = False,
                **_: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        if self._buffer is None:
            raise ValueError('combine called before dispatch created ElasticBuffer')
        num_sms = self._num_sms if config is None else config.num_sms
        return self._buffer.combine(
            x,
            handle=handle,
            topk_weights=topk_weights,
            num_sms=num_sms,
            num_qps=self.num_qps_per_rank,
            previous_event=previous_event,
            async_with_compute_stream=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

    def low_latency_dispatch(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            'Lite-EP DeepEPv2 Buffer shim does not implement low_latency_dispatch'
        )

    def low_latency_combine(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            'Lite-EP DeepEPv2 Buffer shim does not implement low_latency_combine'
        )

    def destroy(self) -> None:
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None
            self._buffer_key = None
            self._pending_num_experts = None