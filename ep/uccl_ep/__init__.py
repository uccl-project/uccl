"""
UCCL Expert Parallel (EP) Communication Library

This package provides high-performance communication primitives for
Mixture of Experts (MoE) models, supporting:
- High-throughput intranode all-to-all (using NVLink)
- High-throughput internode all-to-all (using RDMA and NVLink)
- Low-latency all-to-all (using RDMA)
"""

import torch

from .utils import EventOverlap, check_nvlink_connections, initialize_uccl, destroy_uccl
from .buffer import Buffer

# Import C++ extension types
try:
    from ep import Config, EventHandle
except ImportError as exc:
    import sys

    sys.stderr.write(
        "Failed to import C++ extension 'ep'. Make sure the package is installed correctly.\n"
    )
    raise

__version__ = "0.0.1"

__all__ = [
    "Buffer",
    "EventOverlap",
    "Config",
    "EventHandle",
    "check_nvlink_connections",
    "initialize_uccl",
    "destroy_uccl",
]
