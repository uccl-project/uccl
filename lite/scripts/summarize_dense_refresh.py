#!/usr/bin/env python3
"""Validate the three-run dense shape logs and retain every variant sample."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

SHAPES = (
    "llama8-m128", "llama8-m256", "llama8-m512", "llama8-m1024",
    "llama8-m2048", "qwen32-m2048", "llama70-m2048", "reference-m4096",
)
HEADINGS = {
    "AG": "# AG+GEMM ",
    "RS": "# GEMM+RS ",
    "AR": "# GEMM+AllReduce ",
}
BASELINES = {
    "AG": "NCCL-AG+cuBLAS",
    "RS": "cuBLAS+NCCL-RS",
    "AR": "cuBLAS+NCCL-AllReduce",
}
CANDIDATES = {
    "RS": "ready-ring-GEMM+RS",
    "AR": "ready-ring-GEMM+AR",
}
ROW = re.compile(r"^\s{2}(\S+)\s+(\d+\.\d+)\s+\d+\.\d+x$", re.M)


def parse(path: Path) -> dict[str, tuple[dict[str, float], str]]:
    text = path.read_text()
    if any(f"# {name} correctness" not in text for name in ("AG", "RS", "AR")):
        raise ValueError(f"missing dense correctness checks: {path}")
    if re.search(r"\bbad\s+[1-9]\d*\b", text):
        raise ValueError(f"incorrect output: {path}")
    sections = {}
    for operation, heading in HEADINGS.items():
        start = text.find(heading)
        if start < 0:
            raise ValueError(f"missing {operation} section: {path}")
        next_heading = [text.find(marker, start + 1) for marker in HEADINGS.values()]
        end = min((value for value in next_heading if value >= 0), default=len(text))
        section = text[start:end]
        variants = dict((name, float(value)) for name, value in ROW.findall(section))
        if BASELINES[operation] not in variants:
            raise ValueError(f"missing {operation} baseline: {path}")
        if operation == "AG":
            match = re.search(r"^# AG dispatch: (\S+),", section, re.M)
            if not match:
                raise ValueError(f"missing AG dispatch: {path}")
            candidate = match.group(1)
        else:
            candidate = CANDIDATES[operation]
        if candidate not in variants:
            raise ValueError(f"missing {operation} candidate {candidate}: {path}")
        if any(not math.isfinite(value) or value <= 0 for value in variants.values()):
            raise ValueError(f"invalid timing: {path}")
        sections[operation] = variants, candidate
    return sections


def summarize(directory: Path, samples_path: Path, summary_path: Path) -> None:
    samples = []
    selected = defaultdict(list)
    paths = defaultdict(set)
    for shape in SHAPES:
        for repeat in (1, 2, 3):
            path = directory / f"{shape}_run{repeat}.log"
            for operation, (variants, candidate) in parse(path).items():
                paths[(shape, operation)].add(candidate)
                for variant, value in variants.items():
                    samples.append({"shape": shape, "operation": operation,
                                    "repeat": repeat, "variant": variant,
                                    "latency_ms": value})
                selected[(shape, operation, "baseline")].append(
                    variants[BASELINES[operation]])
                selected[(shape, operation, "uccl")].append(variants[candidate])
    if any(len(values) != 1 for values in paths.values()):
        raise ValueError("a shape changed selected path across repetitions")
    summary = []
    for shape in SHAPES:
        for operation in HEADINGS:
            own = selected[(shape, operation, "uccl")]
            baseline = selected[(shape, operation, "baseline")]
            if len(own) != 3 or len(baseline) != 3:
                raise ValueError("missing dense samples")
            summary.append({
                "shape": shape, "operation": operation,
                "uccl_path": next(iter(paths[(shape, operation)])),
                "baseline": BASELINES[operation],
                "uccl_min_ms": min(own), "uccl_median_ms": median(own),
                "uccl_max_ms": max(own),
                "baseline_min_ms": min(baseline),
                "baseline_median_ms": median(baseline),
                "baseline_max_ms": max(baseline),
                "speedup": median(baseline) / median(own),
            })
    for destination, rows in ((samples_path, samples), (summary_path, summary)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    summarize(args.directory, args.samples, args.summary)
