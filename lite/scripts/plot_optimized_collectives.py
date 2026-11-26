#!/usr/bin/env python3
"""Draw optimized AR/AG/RS paths with the current eight-L4 figure style."""

import argparse
import shutil

from figure_style import ROOT, configure
import plot_refresh_collectives as reference
import plot_multinode_host_router as host_router


OPTIMIZED = ROOT / "benchmarks/data/optimization_20261110"
MIDRANGE_AG = ROOT / "benchmarks/data/ag_midrange_4l4_20261111"
PACKET_AG = ROOT / "benchmarks/data/fp32_packet_ag_4l4_20261111"
PER_SOURCE_AG = ROOT / "benchmarks/data/fp32_packet_ag_per_source_4l4_20261111"
SERIAL_RS = ROOT / "benchmarks/data/rs_ar_serial_4l4_20261111"
TUNED_MULTI = ROOT / "benchmarks/data/nccl_tuned_2x4l4_20261111"
HOST_MULTI_AG = ROOT / "benchmarks/data/host_multicast_selected_2x4l4_20261111"
OPERATIONS = (
    ("l40_selected.csv", "operation", "allgather",
     "collectives-4l4-allgather-optimized", 3 / 4),
    ("selected.csv", "operation", "allgather",
     "collectives-4l4-allgather-packet-router", 3 / 4),
    ("selected.csv", "operation", "allgather",
     "collectives-4l4-allgather-per-source", 3 / 4),
    ("l40_selected.csv", "operation", "reducescatter",
     "collectives-4l4-reducescatter-optimized", 3 / 4),
    ("tuned_summary.csv", "collective", "allreduce",
     "collectives-2x4l4-allreduce-optimized", 14 / 8),
    ("selected_summary.csv", "collective", "allgather",
     "collectives-2x4l4-allgather-optimized", 7 / 8),
    ("tuned_summary.csv", "collective", "reducescatter",
     "collectives-2x4l4-reducescatter-optimized", 7 / 8),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    configure()
    for source, field, operation, filename, factor in OPERATIONS:
        if filename == "collectives-4l4-allgather-optimized":
            reference.DATA = MIDRANGE_AG
        elif filename == "collectives-4l4-allgather-packet-router":
            reference.DATA = PACKET_AG
        elif filename == "collectives-4l4-allgather-per-source":
            reference.DATA = PER_SOURCE_AG
        elif filename == "collectives-4l4-reducescatter-optimized":
            reference.DATA = SERIAL_RS
        elif filename == "collectives-2x4l4-allgather-optimized":
            reference.DATA = HOST_MULTI_AG
        elif filename.startswith("collectives-2x4l4-"):
            reference.DATA = TUNED_MULTI
        else:
            reference.DATA = OPTIMIZED
        data = reference.load(source, field, operation)
        reference.plot_operation(
            data, filename, factor, args.preview,
            nccl_label=("NCCL 2.28.7 tuned" if filename.startswith(
                "collectives-2x4l4-") else "NCCL 2.28.7"),
            x_label=("Complete output bytes per GPU" if filename in (
                     "collectives-2x4l4-allgather-optimized",
                     "collectives-4l4-allgather-packet-router",
                     "collectives-4l4-allgather-per-source") else
                     "Complete tensor bytes per rank"))
        if filename == "collectives-2x4l4-allgather-optimized":
            shutil.copyfile(
                ROOT / "docs/assets" / f"{filename}.svg",
                ROOT / "docs/assets/collectives-2x4l4-allgather-host-multicast.svg")

    host_router.draw(args.preview)


if __name__ == "__main__":
    main()
