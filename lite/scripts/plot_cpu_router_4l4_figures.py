#!/usr/bin/env python3
"""Draw four-L4 BF16 CPU-router comparisons in the existing figure style."""

import argparse

from figure_style import ROOT, configure
import plot_refresh_collectives as reference


DATA = ROOT / "benchmarks/data/cpu_router_4l4_figures_20261111"
OPERATIONS = (
    ("allreduce", "collectives-4l4-bf16-router-allreduce", 6 / 4),
    ("allgather", "collectives-4l4-bf16-router-allgather", 3 / 4),
    ("reducescatter", "collectives-4l4-bf16-router-reducescatter", 3 / 4),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    configure()
    reference.DATA = DATA
    for operation, filename, factor in OPERATIONS:
        data = reference.load("figure_summary.csv", "operation", operation)
        if any("router" not in row["uccl_path"] and
               "gpu-progress" not in row["uccl_path"]
               for row in data.values()):
            raise ValueError(f"non-UCCL point in {operation}")
        reference.plot_operation(
            data, filename, factor, args.preview,
            nccl_label="NCCL 2.28.7",
            uccl_label="UCCL-Lite CPU router",
            x_label="Complete tensor bytes per GPU")


if __name__ == "__main__":
    main()
