import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
import os
import sys
import torch
import torch.distributed as dist
from typing import Optional, Union


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
    rank = dist.get_rank(group)
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
