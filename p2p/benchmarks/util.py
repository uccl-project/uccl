import torch
import numpy as np
import random
import torch.distributed as dist
from dataclasses import dataclass


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_fcp_comm_plan(num_nodes: int):
    send_to = np.random.permutation(num_nodes)
    recv_from = np.empty(num_nodes, dtype=int)
    recv_from[send_to] = np.arange(num_nodes)
    return send_to, recv_from


def get_fcp_comm_plans(num_nodes: int, num_iters: int):
    plans = []
    for it in range(num_iters):
        plans.append(get_fcp_comm_plan(num_nodes))
    return plans


def sync_all():
    dist.barrier()
    torch.cuda.synchronize()


@dataclass
class Metrics:
    avg_time: float
    total_flops: float
    mem_buckets: np.ndarray
    flops_buckets: np.ndarray
    seq_lens: np.ndarray
