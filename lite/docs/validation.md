# Validation scope

The current runtime and its limits are described in
[topology-runtime-status.md](topology-runtime-status.md). Build and launch
commands are in [build.md](build.md).

The final topology-runtime implementation passed 14 runtime cases and 13 CPU
planner checkers on the recorded Xeon Silver 4410Y/L4 machines. Runtime coverage
includes four GPUs on one host, two-host 1+1/3+1/4+4 layouts, isolated CUDA
visibility, FP32/BF16, complete buffers and guards, aliases, scalar tails, direct
cache cold/warm reuse, C API reporting, ordinary NCCL ABI ownership and strict
execution without native NCCL. The four-rank BF16 Compute Sanitizer run reports
zero device errors; only handled MSCCL++ capability-probe API reports were
suppressed. These checks are correctness evidence, not a speedup claim.

The core's 2–32-rank capacity does not validate deployments larger than the two
physical hosts/eight GPUs tested. Distributed graphs retain the caller's graph
implementation; specialized executors keep explicit topology restrictions.
Hardware/software/configuration changes invalidate qualification evidence.

The source cleanup changes benchmark/documentation packaging and build entry
points, not collective algorithms or admission thresholds. Retained figures
use historical, scoped observations and do not qualify a subsequent source
change. [Archived experiments](archived-experiments.md) retain earlier numerical
checks, failed candidates, performance regressions and original run protocols.

For new router or collective changes, run the applicable policy/dispatcher,
eager/graph and four-GPU checks; network changes additionally require two
four-GPU hosts. Compilation alone must be reported as such.
