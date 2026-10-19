"""Actual Qwen3 MoE weights with forward HT and selectable expert arithmetic."""
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, cast

import torch
from expert_mma import ExpertMma
from expert_routes import expert_routes
from ht_bridge import HtBridge
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeMLP,
    Qwen3MoeSparseMoeBlock,
)


@dataclass
class Worker:
    stream: torch.cuda.Stream
    transport: HtBridge
    mma: ExpertMma
    backend: Literal['torch', 'torch-ht', 'cutlass-ht'] = 'torch'
    routing: Literal['where', 'sorted'] = 'where'


ACTIVE_WORKER: ContextVar[Worker | None] = ContextVar('qwen_ht_worker', default=None)


class QwenHtMoe(nn.Module):
    def __init__(self, original: Qwen3MoeSparseMoeBlock):
        super().__init__()
        assert original.num_experts == 128 and original.top_k == 8 and original.norm_topk_prob
        self.original = original

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        worker = ACTIVE_WORKER.get()
        if worker is None or worker.backend == 'torch':
            return cast(tuple[torch.Tensor, torch.Tensor], self.original(hidden_states))
        batch, length, hidden = hidden_states.shape
        value = hidden_states.reshape(-1, hidden).contiguous()
        logits = self.original.gate(value)
        weights, indices = torch.topk(F.softmax(logits, dim=1, dtype=torch.float32), 8, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
        dispatched = worker.transport.dispatch(value, indices.contiguous(), weights.float().contiguous())
        combined = torch.zeros_like(dispatched.hidden)
        per_rank = 128 // worker.transport.world
        for route in expert_routes(dispatched, per_rank, worker.routing):
            expert = cast(Qwen3MoeMLP, self.original.experts[worker.transport.rank * per_rank + route.expert])
            selected = dispatched.hidden.index_select(0, route.rows).contiguous()
            if worker.backend == 'torch-ht':
                output = expert(selected)
            else:
                gate = worker.mma.linear(selected, expert.gate_proj.weight)
                up = worker.mma.linear(selected, expert.up_proj.weight)
                activated = worker.mma.swiglu(gate, up)
                output = worker.mma.linear(activated, expert.down_proj.weight)
            combined.index_add_(0, route.rows, output * route.probability[:, None])
        return worker.transport.combine(combined).reshape(batch, length, hidden), logits
