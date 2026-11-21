"""Stable expert segments without a separate CUDA nonzero for every expert."""
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, cast

import torch
from ht_bridge import Dispatched


@dataclass
class ExpertRoute:
    expert: int
    rows: torch.Tensor
    probability: torch.Tensor


def expert_routes(value: Dispatched, experts: int,
                  strategy: Literal['where', 'sorted']) -> Iterator[ExpertRoute]:
    if strategy == 'where':
        for expert in range(experts):
            rows, positions = torch.where(value.experts == expert)
            if rows.numel():
                yield ExpertRoute(expert, rows, value.weights[rows, positions].to(torch.bfloat16))
        return
    # HT uses -1 for unused slots and local expert IDs in [0, experts).
    keys = value.experts.flatten() + 1
    order = torch.argsort(keys, stable=True)
    counts = torch.zeros(experts + 1, device=keys.device, dtype=torch.int64)
    counts.scatter_add_(0, keys, torch.ones_like(keys))
    sizes = cast(list[int], counts.cpu().tolist())
    rows = torch.div(order, value.experts.shape[1], rounding_mode='floor')
    probability = value.weights.flatten().index_select(0, order).to(torch.bfloat16)
    start = sizes[0]
    for expert, size in enumerate(sizes[1:]):
        if size:
            yield ExpertRoute(expert, rows[start:start + size], probability[start:start + size])
        start += size
