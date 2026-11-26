#!/usr/bin/env python3
"""Calibrate existing router options; experimental transports are not selected."""
from run_compact_transfer import DATA, run


for operation, options in (
    ("allgather", ["--small-ag-gpu-progress-max-bytes", "1048576"]),
    ("reducescatter", ["--small-rs-gpu-progress-max-bytes", "65536",
                       "--small-rs-workers-per-node", "4"]),
    ("allreduce", ["--small-ar-gpu-progress-max-bytes", "8192",
                   "--workers-per-node", "4", "--transport-blocks", "16",
                   "--in-place"]),
):
    common = ["--gpus", "8", "--op", operation, "--fallback", "nccl",
              "--main-cpu", "47", "--batch-cpu-jobs",
              "--gpu-launch-workers-per-node", "4",
              "--large-ag-packet-bytes", "2097152", *options]
    profile = DATA / ("existing_" + operation + ".profile")
    run("calibrate_" + operation,
        common + ["--begin", "1024", "--end", "1048576", "--warmup", "5",
                  "--iterations", "10", "--repeats", "3",
                  "--samples", str(DATA / ("calibrate_" + operation + "_samples.csv")),
                  "--calibrate", str(profile)], binary="uccl_lite_collectives")
    replay = common + ["--backend", "auto", "--profile", str(profile),
                       "--warmup", "1", "--iterations", "2", "--repeats", "3"]
    run("auto_" + operation, replay + ["--begin", "1024", "--end", "1048576"],
        binary="uccl_lite_collectives")
    run("unknown_size_" + operation, replay + ["--begin", "1536", "--end", "1536"],
        binary="uccl_lite_collectives")

# The attribution sweep separately tests existing GPU progress for larger AG.
common = ["--gpus", "8", "--op", "allgather", "--fallback", "nccl",
          "--main-cpu", "47", "--batch-cpu-jobs",
          "--gpu-launch-workers-per-node", "4", "--gpu-progress", "--ag-per-source",
          "--large-ag-packet-bytes", "2097152"]
profile = DATA / "existing_allgather_full_progress.profile"
run("calibrate_allgather_full_progress",
    common + ["--begin", "2097152", "--end", "16777216", "--warmup", "5",
              "--iterations", "10", "--repeats", "3",
              "--samples", str(DATA / "calibrate_allgather_full_progress_samples.csv"),
              "--calibrate", str(profile)], binary="uccl_lite_collectives")
replay = common + ["--backend", "auto", "--profile", str(profile),
                   "--warmup", "1", "--iterations", "2", "--repeats", "3"]
run("auto_allgather_full_progress", replay + ["--begin", "2097152", "--end", "16777216"],
    binary="uccl_lite_collectives")
run("unknown_size_allgather_full_progress", replay + ["--begin", "3145728", "--end", "3145728"],
    binary="uccl_lite_collectives")
