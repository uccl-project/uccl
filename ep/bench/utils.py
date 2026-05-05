"""Thin compatibility shim — all implementation lives in uccl.ep.utils.

Bench scripts traditionally do ``from utils import init_dist, ...``. Keep
that working without duplicating the source file by re-exporting from the
canonical module.
"""

from uccl.ep.utils import *  # noqa: F401,F403
from uccl.ep.utils import (  # noqa: F401  explicit re-exports for static analysis
    EventOverlap,
    EventHandle,
    bench,
    bench_kineto,
    calc_diff,
    check_nvlink_connections,
    create_grouped_scores,
    destroy_uccl,
    detect_group_topology,
    detect_ib_hca,
    empty_suppress,
    get_cpu_proxies_meta,
    get_peer_ip,
    hash_tensor,
    init_dist,
    init_dist_under_torchrun,
    initialize_uccl,
    inplace_unique,
    per_token_cast_back,
    per_token_cast_to_fp8,
    suppress_stdout_stderr,
    _fp8_e4m3_dtype,
)
