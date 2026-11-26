# Eight-L4 SHM refresh

Eight NVIDIA L4s, GPUs 0–3 on NUMA node 0 and 4–7 on node 1; driver 580.178.04,
CUDA 12.9, NCCL 2.28.9. BF16; AR in place, AG/RS out of place. Sizes are complete
input bytes/rank for AR/RS and complete output bytes/rank for AG.

`--shm-only` disables every peer link in the benchmark topology. NCCL uses
`NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0`, `NCCL_IB_DISABLE=1` and Socket.
The [initialization probe](raw/nccl_probe.csv) verifies all **16 channel edges
use SHM**; the router transfers through mapped host packets.

Two sweeps cover every power of two from 1 KiB through 4 MiB, in AR/AG/RS then
RS/AG/AR order. Per backend and size: ten warmups, five alternating repetitions
of twenty calls. Every output element is checked on every GPU, including all
**15,600 timed calls** and **1,560 warmup calls**. Each repeat reports its median
slowest-rank CUDA-event interval, including host submission gaps. Points are
the median of two sweep medians. Calls have no deliberate idle. Compute process
lists are checked before and after every sweep.

All operations use CPU callback batching and four submission lanes per NUMA
node, with submission CPU 47. AR uses a fixed four-worker/16-block configuration
and GPU progress through 8 KiB; AG uses per-source GPU progress through 1 MiB;
RS uses GPU progress through 64 KiB and a four-worker CPU cap at 256 KiB–1 MiB.
These choices are fixed across sizes, opt-in and require exact local calibration.
Parallel lanes reserve eight CPU threads and spin up to 2 ms. No application
speedup or sparse-call gain is established.

| Collective | Bytes/rank | UCCL Lite (µs) | NCCL SHM (µs) | NCCL / Lite |
| --- | ---: | ---: | ---: | ---: |
| all_reduce | 1 KiB | 93.2 | 147.2 | 1.58x |
| all_reduce | 1 MiB | 468.2 | 621.3 | 1.33x |
| all_gather | 1 KiB | 82.4 | 132.6 | 1.61x |
| all_gather | 1 MiB | 272.4 | 383.0 | 1.41x |
| reduce_scatter | 1 KiB | 93.4 | 134.7 | 1.44x |
| reduce_scatter | 1 MiB | 313.1 | 358.7 | 1.15x |

[summary.csv](summary.csv) contains all 39 points and both sweep medians.
[provenance.json](provenance.json) records exact options, binary/source identity,
timing and validation. `raw/` retains stdout, logs, local profiles, command
manifests and losslessly compressed per-call samples. `summarize.py` reproduces
the summary. Copy `reproduction/` scripts to `experimental/lite/build/shm_refresh_20261114`
before running; use the recorded binary/source state and fresh output names.
The matching source previously passed [13 core correctness checks](../ar_ag_transfer_8l4_20261113/correctness.json).

The [six overview figures](../overview_8l4_20261114/README.md) use these paired
results through 4 MiB. Their 8-MiB–1-GiB points retain the recorded
[historical SHM results](../shm_8l4_20261111/README.md); these larger sizes were
not remeasured. [checksums.json](checksums.json) covers every archived file.
