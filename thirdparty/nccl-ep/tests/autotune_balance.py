#!/usr/bin/env python3
"""Fit drained-wave traffic policies and compare an independent traffic sweep."""

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from measurement import Measurement, read_pair

CLASSES = (1, 4, 8)
POLICIES = ((8, 18), (4, 37), (2, 74), (1, 148))
SWEEP = (1, 4, 8, 4, 1)


@dataclass(frozen=True)
class TrafficMeasurement:
    timing: Measurement
    path: str
    passes: int
    active: int
    ctas: int
    resident: int
    dynamic: bool

    def identity(self) -> tuple:
        # Active groups and CTA count are the tuned variables; resident memory is fixed.
        m = self.timing
        return (
            m.workload[:6] + m.workload[7:], m.topology, m.source_sha256,
            m.binary_sha256, self.path, self.resident,
        )


def read(directory: Path, repeats: int, ranks: int, phase: int | None = None) -> TrafficMeasurement:
    timing = read_pair(directory, 64, repeats, 1, ranks, traffic=True, phase=phase)
    suffix = "" if phase is None else f"-phase{phase}"
    reports = [json.loads((directory / f"rank{r}-slots64{suffix}.json").read_text())
               for r in range(ranks)]
    first = reports[0]
    keys = ("expert_path", "expert_passes", "expert_ctas", "allocated_EP_groups",
            "requested_resident_groups", "balance_from_profile_table")
    assert all(all(r[k] == first[k] for k in keys) for r in reports)
    assert first["expert_path"] in ("direct", "tma", "tmem-input", "tmem-output")
    assert (timing.stages, timing.blocks) == (0, 16)
    assert first["proxy_taskset"] == 1
    assert first["requested_resident_groups"] == first["allocated_EP_groups"] == 8
    assert first["expert_passes"] in CLASSES
    assert timing.pool == first["requested_active"]
    return TrafficMeasurement(timing, first["expert_path"], first["expert_passes"],
                              timing.pool, first["expert_ctas"], first["allocated_EP_groups"],
                              first["balance_from_profile_table"])


def train(root: Path, sessions: int, repeats: int, ranks: int) -> dict:
    choices, candidates = [], []
    identity = None
    for passes in CLASSES:
        rows = []
        for active, ctas in POLICIES:
            measurements = [read(root / f"session-{s}/p{passes}-a{active}-c{ctas}", repeats, ranks)
                            for s in range(sessions)]
            for m in measurements:
                if identity is None:
                    identity = m.identity()
                assert m.identity() == identity and not m.dynamic
                assert (m.passes, m.active, m.ctas) == (passes, active, ctas)
            row = {"passes": passes, "active": active, "ctas": ctas,
                   "gpu_ms": statistics.median(m.timing.gpu_ms for m in measurements),
                   "host_ms": statistics.median(m.timing.host_ms for m in measurements)}
            rows.append(row)
        candidates.extend(rows)
        choices.append(min(rows, key=lambda r: r["gpu_ms"]))
    return {"schema": 1, "phase": "TRAINING_ONLY", "sessions": sessions,
            "samples_per_launch": repeats, "physical_gpus": ranks, "nodes": 1,
            "identity": identity, "ranking": "median of session maximum-rank GPU window medians",
            "classes": CLASSES, "candidate_policies": POLICIES, "selected": choices,
            "candidates": candidates}


def holdout(root: Path, model: Path, sessions: int, repeats: int, ranks: int) -> dict:
    policy = json.loads(model.read_text())
    assert policy["phase"] == "TRAINING_ONLY" and policy["physical_gpus"] == ranks
    rows = []
    for phase, passes in enumerate(SWEEP):
        pairs = []
        choice = next(r for r in policy["selected"] if r["passes"] == passes)
        for session in range(sessions):
            directory = root / f"session-{session}"
            baseline = read(directory / "baseline", repeats, ranks, phase)
            candidate = read(directory / "candidate", repeats, ranks, phase)
            assert json.loads(json.dumps(baseline.identity())) == policy["identity"]
            assert baseline.identity() == candidate.identity()
            assert baseline.passes == candidate.passes == passes
            assert not baseline.dynamic and candidate.dynamic
            assert (baseline.active, baseline.ctas) == POLICIES[0]
            assert (candidate.active, candidate.ctas) == (choice["active"], choice["ctas"])
            pairs.append({"gpu_ratio": baseline.timing.gpu_ms / candidate.timing.gpu_ms,
                          "host_ratio": baseline.timing.host_ms / candidate.timing.host_ms,
                          "baseline_gpu_ms": baseline.timing.gpu_ms, "candidate_gpu_ms": candidate.timing.gpu_ms,
                          "baseline_host_ms": baseline.timing.host_ms, "candidate_host_ms": candidate.timing.host_ms})
        ratios = [p["gpu_ratio"] for p in pairs]
        rows.append({"phase": phase, "passes": passes, "choice": choice,
                     "gpu_ratio_median": statistics.median(ratios), "gpu_ratio_range": [min(ratios), max(ratios)],
                     "host_ratio_median": statistics.median(p["host_ratio"] for p in pairs), "pairs": pairs})
    return {"schema": 1, "phase": "INDEPENDENT_HOLDOUT", "sessions": sessions,
            "model": policy, "comparisons": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "holdout"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ranks", type=int, choices=(4, 8), default=8)
    args = parser.parse_args()
    if args.phase == "holdout":
        assert args.model is not None
        result = holdout(args.root, args.model, args.sessions, args.repeats, args.ranks)
    else:
        result = train(args.root, args.sessions, args.repeats, args.ranks)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
