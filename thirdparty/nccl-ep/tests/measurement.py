"""Read qualified HT fixture reports without selecting historical policies."""

import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Measurement:
    slots: int
    stages: int
    blocks: int
    pool: int
    gpu_ms: float
    host_ms: float
    workload: tuple[int, int, int, int, int, int, int, int, int, int, int, str]
    source_sha256: str
    binary_sha256: str
    topology: tuple[int, int, int] = (0, 0, 0)


def read_pair(
    directory: Path,
    slots: int,
    repeats: int,
    nodes: int,
    physical_gpus: int | None = None,
    *,
    traffic: bool = False,
    phase: int | None = None,
) -> Measurement:
    ranks = nodes if physical_gpus is None else physical_gpus
    assert nodes in (1, 2) and ranks in (1, 2, 4, 8)
    assert ranks % nodes == 0 and ranks // nodes in (1, 4, 8)
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["exit_code"] == 0
    assert receipt["nodes"] == nodes
    assert receipt["generation_mode"] == 1 and not receipt["diagnostic"]
    assert not receipt["instrument"]
    reports = []
    for rank in range(ranks):
        suffix = "" if phase is None else f"-phase{phase}"
        path = directory / f"rank{rank}-slots{slots}{suffix}.json"
        report = json.loads(path.read_text())
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == receipt["reports"][path.name]
        )
        assert (
            report["schema"] == 2
            and report["combine_backward"]
            and report["full_probability_validation"]
            and report["status"] == "PASS"
            and report["rank"] == rank
        )
        assert report["nodes"] == nodes and report["physical_gpus"] == ranks
        # Current policy models describe the uniform-weight fixture only.
        assert not report.get("nonuniform_probability_weights", False)
        # Traffic and balance experiments require their own workload identity.
        assert not report.get("phase_events", False)
        if not traffic:
            assert phase is None
            assert report.get("expert_path", "copy") == "copy"
            assert report.get("expert_passes", 1) == 1
            assert not report.get("balance_from_profile_table", False)
            assert report.get(
                "requested_resident_groups", report["requested_active"]
            ) == report["requested_active"]
        assert (
            report["logical_slots_per_rank"] == slots
            and report["total_logical_slots"] == ranks * slots
        )
        assert report["errors"] == 0 and report["completed"] == slots * (repeats + 3)
        assert report["per_handle_peak"] == 1 and report["normal_drain"]
        assert report["callback_final_live"] == 0
        assert report["invalid_combine_rejections"] == 0
        assert not report["memory_limited"]
        assert report["allocated_EP_groups"] == report["resource_pool_limit"]
        if "launcher_cpu_budget_count" in report:
            budget = report["launcher_cpu_budget_count"]
            mode = report["proxy_taskset"]
            assert type(budget) is int and budget > (1 if mode == 2 else 0)
            assert mode in (0, 1, 2)
            assert report["configured_proxy_cpu_budget_count"] == (
                budget - (mode == 2) if mode else 0
            )
            assert report["submit_cpu_budget_count"] == (1 if mode == 2 else budget)
        else:
            assert report["proxy_taskset"] in (0, 1)
        for key in (
            "tokens",
            "hidden",
            "sm_major",
            "sm_minor",
            "cuda_runtime",
            "nccl_version",
            "requested_active",
            "proxy_taskset",
        ):
            assert type(report[key]) is int
        assert report["observed_peak_admitted"] == report["agreed_active"]
        operations = ["dispatch", "combine"]
        if traffic and report["expert_path"] != "copy":
            operations.append("expert")
        for operation in operations:
            kernel = report[operation]
            smem = kernel["static_smem"] + kernel["dynamic_smem"]
            assert smem <= kernel["smem_per_block"]
            resident = kernel["resident_blocks_api"]
            if smem:
                resident = min(resident, kernel["smem_per_sm"] // smem)
            assert (
                report["agreed_active"] * kernel["blocks"] <= kernel["sms"] * resident
            )
        for metric in ("gpu_window_sum_ms", "host_wave_with_validation_ms"):
            assert len(report[metric]) == repeats and min(report[metric]) > 0
        reports.append(report)
    first = reports[0]
    for report in reports[1:]:
        assert report.get("launcher_cpu_budget_count", 0) == first.get(
            "launcher_cpu_budget_count", 0
        )
        for key in (
            "stages",
            "blocks",
            "tokens",
            "hidden",
            "agreed_active",
            "allocated_EP_groups",
            "sm_major",
            "sm_minor",
            "cuda_runtime",
            "nccl_version",
            "backend",
            "proxy_taskset",
            "requested_active",
        ):
            assert report[key] == first[key]
    return Measurement(
        slots,
        first["stages"],
        first["blocks"],
        first["agreed_active"],
        max(statistics.median(r["gpu_window_sum_ms"]) for r in reports),
        max(statistics.median(r["host_wave_with_validation_ms"]) for r in reports),
        tuple(
            first[key]
            for key in (
                "tokens",
                "hidden",
                "sm_major",
                "sm_minor",
                "cuda_runtime",
                "nccl_version",
                "requested_active",
                "proxy_taskset",
            )
        )
        + tuple(
            first["dispatch"][key] for key in ("sms", "smem_per_sm", "smem_per_block")
        )
        + (first["backend"],),
        receipt["source_manifest_sha256"],
        hashlib.sha256(
            json.dumps(receipt["binaries"], sort_keys=True).encode()
        ).hexdigest(),
        (ranks, nodes, first.get("launcher_cpu_budget_count", 0)),
    )
