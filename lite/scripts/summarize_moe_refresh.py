#!/usr/bin/env python3
"""Validate the settled MoE round samples and summarize complete-layer time."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

VARIANTS = ("nccl+cublas", "ours+cublas", "fused")
SAMPLE = re.compile(r"^# sample,(\d+),([^,]+),(\d+),([\d.]+)$", re.M)
RESULT = re.compile(
    r"^\s*(\d+)\s+\d+\s+[\d.]+\s+(nccl\+cublas|ours\+cublas|fused)\s+"
    r"([\d.]+)\s+[\d.]+\s+[\d.]+\s+(\d+)\s+([\d.eE+-]+)$",
    re.M,
)


def summarize(source: Path, samples_path: Path, summary_path: Path) -> None:
    text = source.read_text()
    samples = defaultdict(dict)
    sample_rows = []
    for token_count, variant, round_number, latency in SAMPLE.findall(text):
        tokens, round_id, us = int(token_count), int(round_number), float(latency)
        if variant not in VARIANTS or not 1 <= round_id <= 12 or not math.isfinite(us) or us <= 0:
            raise ValueError("invalid MoE round sample")
        key = (tokens, variant)
        if round_id in samples[key]:
            raise ValueError("duplicate MoE round sample")
        samples[key][round_id] = us
        sample_rows.append({"tokens_per_rank": tokens, "variant": variant,
                            "round": round_id, "latency_us": us})
    expected_tokens = [1 << power for power in range(3, 16)]
    if sorted({tokens for tokens, _ in samples}) != expected_tokens:
        raise ValueError("incomplete MoE token sweep")
    for tokens in expected_tokens:
        for variant in VARIANTS:
            if sorted(samples[(tokens, variant)]) != list(range(1, 13)):
                raise ValueError(f"missing MoE rounds: {tokens}, {variant}")
    reported = {}
    for tokens, variant, latency, bad, relerr in RESULT.findall(text):
        key = (int(tokens), variant)
        if key in reported or int(bad) != 0 or not math.isfinite(float(relerr)):
            raise ValueError(f"invalid MoE result: {key}")
        reported[key] = float(latency)
    if len(reported) != len(expected_tokens) * len(VARIANTS):
        raise ValueError("missing reported MoE results")
    summary = []
    for tokens in expected_tokens:
        for variant in VARIANTS:
            measured = min(samples[(tokens, variant)].values())
            if abs(measured - reported[(tokens, variant)]) > 0.02:
                raise ValueError(f"MoE summary disagrees with rounds: {tokens}, {variant}")
        baseline = reported[(tokens, "nccl+cublas")]
        fused = reported[(tokens, "fused")]
        summary.append({
            "tokens_per_rank": tokens,
            "baseline": "NCCL + grouped cuBLAS",
            "uccl_path": "fused dispatch + grouped GEMM + combine",
            "baseline_us": baseline,
            "uccl_us": fused,
            "speedup": baseline / fused,
            "baseline_tokens_per_s_rank": tokens / (baseline * 1e-6),
            "uccl_tokens_per_s_rank": tokens / (fused * 1e-6),
        })
    for destination, rows in ((samples_path, sample_rows),
                              (summary_path, summary)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    summarize(args.source, args.samples, args.summary)
