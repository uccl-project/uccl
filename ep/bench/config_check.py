"""Cross-rank configuration check for ``Buffer``.

Every rank of an EP group must agree on the build constants, buffer sizes,
channel configs and protocol knobs that decide where a peer writes into its
buffers. Today a mismatch (for example ``UCCL_EP_DISPATCH_CONFIG`` set on one
rank only) is found by a device timeout or a crash. This module gathers a
small signature from every rank once, before any buffer is allocated or any
proxy is started, and makes every rank take the same decision from the same
gathered list: raise the same error, print the same warning, or pass.

``UCCL_EP_CONFIG_CHECK`` = ``1`` (default) | ``warn`` | ``0``. The most
permissive value set on any rank wins, so all ranks still agree.

Scope: only init-time state is compared. ``Buffer.set_num_sms`` and configs
passed per call after construction are not covered.
"""

import os
import re
from typing import Any, Dict, List, Tuple

CHECK_ENV = "UCCL_EP_CONFIG_CHECK"
SIGNATURE_VERSION = 1

# Keys whose mismatch changes the layout or protocol a peer relies on.
HARD_KEYS = (
    "build.num_max_nvl_peers",  # rank split, barrier arrays (ep_configs.cuh)
    "build.num_proxy_threads",  # number of D2H channels -> LL channel layout
    "build.per_expert_batching",  # RDMA double-buffer stride (rdma.cpp)
    "args.low_latency_mode",
    "args.num_nvl_bytes",
    "args.num_rdma_bytes",
    "config.dispatch",  # channel count and ring size (intranode.cu layout)
    "config.combine",
    "env.UCCL_EP_TRANSPORT",  # proxy transport (proxy.cpp)
)

# Keys that change only how one rank sends; no cross-rank dependence found.
WARN_KEYS = (
    "build.sm90_features",
    "build.disable_aggressive_ptx_instrs",
    "env.UCCL_EP_DEDUP_CE",
    "env.UCCL_EP_DEDUP_SEND",
    "env.UCCL_EP_LL_NUM_SMS",
    "env.UCCL_EP_GPU_MR_RO",
    "env.UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC",
)


def _atoi(value) -> int:
    """C ``atoi``: leading whitespace, optional sign, digits; else 0."""
    if value is None:
        return 0
    m = re.match(r"\s*([+-]?\d+)", value)
    return int(m.group(1)) if m else 0


def effective_env() -> Dict[str, Any]:
    """Env knobs as the C++ side interprets them, so unset == its default."""
    g = os.environ.get
    dedup_send = g("UCCL_EP_DEDUP_SEND")
    mr_ro = g("UCCL_EP_GPU_MR_RO")
    aggressive = g("UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC")
    return {
        # proxy.cpp: transport && string(transport) == "cxi"
        "UCCL_EP_TRANSPORT": "cxi" if g("UCCL_EP_TRANSPORT") == "cxi" else "default",
        # uccl_ep.cc: e ? atoi(e) : 0, used as != 0
        "UCCL_EP_DEDUP_CE": _atoi(g("UCCL_EP_DEDUP_CE")) != 0,
        # internode_ll_dedup.cu: unset -> auto, "warp" -> warp, else block
        "UCCL_EP_DEDUP_SEND": (
            "auto"
            if dedup_send is None
            else ("warp" if dedup_send == "warp" else "block")
        ),
        # uccl_ep.cc: cap > 0 ? min(cap, num_sms) : num_sms
        "UCCL_EP_LL_NUM_SMS": max(_atoi(g("UCCL_EP_LL_NUM_SMS")), 0),
        # uccl_ep.cc / rdma.cpp: e != nullptr and e[0] == '1'
        "UCCL_EP_GPU_MR_RO": bool(mr_ro) and mr_ro[0] == "1",
        # common.hpp: env ? atoi(env) : build default
        "UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC": (
            "build-default" if aggressive is None else _atoi(aggressive) != 0
        ),
    }


def check_mode() -> str:
    value = os.environ.get(CHECK_ENV, "1").strip().lower()
    if value in ("0", "off", "false", "no"):
        return "off"
    if value == "warn":
        return "warn"
    return "on"


def _flatten(sig: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in sig.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def compare(gathered: List[Dict[str, Any]]) -> Tuple[str, str, List[str]]:
    """Return (decision, mode, lines). ``decision`` is one of
    ``"pass"``, ``"warn"``, ``"error"``. Pure function of ``gathered``, so
    every rank computes the same result."""
    modes = [s.get("mode", "on") for s in gathered]
    mode = "off" if "off" in modes else ("warn" if "warn" in modes else "on")
    flat = [_flatten(s.get("signature", {})) for s in gathered]
    errors = [(r, s["error"]) for r, s in enumerate(gathered) if s.get("error")]

    hard, warn = [], []
    keys = sorted(set().union(*[f.keys() for f in flat]))
    for key in keys:
        values = [f.get(key, "<missing>") for f in flat]
        if all(v == values[0] for v in values):
            continue
        groups: Dict[str, List[int]] = {}
        for r, v in enumerate(values):
            groups.setdefault(repr(v), []).append(r)
        desc = "; ".join(f"{v} on ranks {rs}" for v, rs in sorted(groups.items()))
        line = f"  {key}: {desc}"
        if key in HARD_KEYS:
            hard.append(line)
        elif key in WARN_KEYS:
            warn.append(line)
        else:
            # Unknown keys (newer signature on some rank) are treated as hard.
            hard.append(line + " (unclassified)")

    lines = []
    if errors:
        lines.append("signature could not be built on some ranks:")
        lines += [f"  rank {r}: {e}" for r, e in errors]
    if hard:
        lines.append("settings that must match on every rank differ:")
        lines += hard
    if warn:
        lines.append("settings that differ across ranks (allowed, check intent):")
        lines += warn

    if mode == "off":
        return "pass", mode, []
    if (errors or hard) and mode == "on":
        return "error", mode, lines
    if errors or hard or warn:
        return "warn", mode, lines
    return "pass", mode, []


def check_consistency(group, signature: Dict[str, Any], error: str = "") -> str:
    """Gather ``signature`` from every rank of ``group`` and act on the result.

    Always gathers (even when this rank opted out) so that ranks with
    different opt-out settings cannot hang each other.
    """
    import torch.distributed as dist

    mine = {
        "version": SIGNATURE_VERSION,
        "mode": check_mode(),
        "signature": signature,
        "error": error,
    }
    gathered: List[Dict[str, Any]] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, mine, group)
    decision, mode, lines = compare(gathered)
    if decision == "error":
        raise RuntimeError(
            "UCCL EP Buffer: cross-rank configuration mismatch "
            f"(set {CHECK_ENV}=warn or 0 to override):\n" + "\n".join(lines)
        )
    if decision == "warn":
        print(
            "[UCCL EP] cross-rank configuration check:\n" + "\n".join(lines),
            flush=True,
        )
    return decision
