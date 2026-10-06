# Combined GIN/NCCL-EP fix validation

Tested device source: `3e012cf8c76f500f5c90dc0c2d34a6bfe46034fc` on official
main `540cc775ba8ab231122f9a98a6d832699f239f70` plus the GIN integration.
Actual hardware: **2×RTX PRO 6000 Blackwell Server Edition**, 96 GB/GPU,
CUDA 13.0.88, driver 580.95.05. GPUs were tested separately.

All **75 controller-launched children exited 0**. This includes native SM120
builds, two compile-rejection gates, 48 signal cases (IDs 0/1023, thread/full
warp, 3/7/8-argument forms, local-rank counts 1/2/8), 28 cooperative correctness
processes (uneven Q, delayed acknowledgements, private/thread groups), 14 tensor
cases, 140 HT rounds and 18 fresh performance processes. Complete source reuse,
routing fields, token/probability values and signal/QUIET counts were checked.

## Large activation / producer-concurrency comparison

B=2048, hidden=2048 BF16 **per warp**, G=64 full warps (2048 GPU producers),
Q=32, capacity=4096, four proxy-major lanes, shared queues, warp channel hints.
Each round copies 1.5 GiB of checked payload across three iterations. All arms
use the same source-poison oracle and entry/exit warp rendezvous. Scalar drains
all Q on every member, elected drains on member zero, production partitions Q
among members. They generate 196,608 / 6,144 / 6,144 QUIET commands per round.

Three fresh trials alternate arm order: scalar/elected/warp, warp/elected/scalar,
elected/scalar/warp. Each process completes five rounds; the first two are
warmup. Medians below use all nine remaining rounds per arm/device.

| Physical GPU | Scalar ms | Elected ms | Warp ms | Speedup vs scalar | Speedup vs elected | Paired trial speedup vs elected (min–max) |
|---|---:|---:|---:|---:|---:|---|
| 0 | 3223.060 | 2541.162 | 1790.843 | 1.800× | 1.419× | 1.398–1.539× |
| 1 | 3838.772 | 2501.774 | 2384.567 | 1.610× | 1.049× | 1.020–1.273× |

The improvement varies substantially by device; GPU1 gains about 5% over
elected drain. These are communication fixtures with one host consumer per
device, including full payload verification. They do not measure pretrained
model inference, HTTP concurrency or a distributed RDMA receiver. The host
reported 100% utilization with zero memory/processes before admission; this
telemetry anomaly remains a measurement limitation.

## Single-rank HT regression comparison

B groups requests of 128 tokens; C is a finite queue drained through serial
batches. Both native NCCL and UCCL-compiled backends run real preprocessing,
dispatch, expert transform and combine, comparing every output value and
routing result. **Both generate zero network commands** in this single-rank
case. GPU event times below exclude the host oracle; speed ratios are a
regression check, not a rail optimization gain.

| GPU | B/C | Token tail | NCCL median GPU ms/wave | UCCL median GPU ms/wave | NCCL/UCCL | Complete hashes |
|---|---|---:|---:|---:|---:|---|
| 0 | 8/32 | 0 | 0.276448 | 0.280288 | 0.986× | identical |
| 0 | 16/64 | 0 | 0.381664 | 0.380544 | 1.003× | identical |
| 0 | 32/128 | 0 | 0.618496 | 0.615776 | 1.004× | identical |
| 0 | 64/128 | 0 | 0.546176 | 0.549568 | 0.994× | identical |
| 0 | 64/128 | 124 | 0.545472 | 0.554016 | 0.985× | identical |
| 1 | 8/32 | 0 | 0.311904 | 0.323264 | 0.965× | identical |
| 1 | 16/64 | 0 | 0.410080 | 0.411456 | 0.997× | identical |
| 1 | 32/128 | 0 | 0.657408 | 0.656384 | 1.002× | identical |
| 1 | 64/128 | 0 | 0.564256 | 0.567296 | 0.995× | identical |
| 1 | 64/128 | 124 | 0.559104 | 0.560160 | 0.998× | identical |

The GPU programs link no NCCL runtime and initialize no communicator. Complete
host/proxy and NCCL-EP library linkage is a separate CUDA CPU CI gate. Distributed
EFA operation and original Thor/RTX 5080 model E2E remain unqualified.

All logs, command receipts, binary hashes and controller/source packet digests
were collected after all 75 children and their controller exited. Another
human-authorized task subsequently acquired GPU0/heavy IO; collection was
read-only and is not a resource handoff. The original canonical lock inodes
remain unchanged. Subsequent source changes restore the mainline shared typedef
and add build-only stub search paths; they do not alter the tested device path.

## Max-Q supplemental native campaign

Source `0fc87bf756f5d01d6d4d2d5da603a9a3db2b3190` was tested on physical GPU1 of
**2×RTX PRO 6000 Blackwell Max-Q Workstation Edition**, 96 GB/GPU, driver
595.104.02 and CUDA 13.0.88. All 34 children exited 0: SM120/compile-rejection
builds, 1024-producer queue, 24 signal cases, 14 cooperation processes, seven
tensor cases, 42 HT rounds and nine fresh drain-comparison processes. Their
controller and every child were absent before the logs were collected.

The B2048/hidden2048/G64/Q32 experiment uses the same three-arm oracle and
three alternating trials described above, with all nine post-warmup rounds
per arm retained. GPU1 was leased independently while another task downloaded
a checkpoint. The container rejected NUMA mempolicy calls; this is a shared-host
measurement rather than an isolated-node performance qualification.

| Physical GPU | Scalar ms | Elected ms | Warp ms | Speedup vs scalar | Speedup vs elected | Paired elected speedup range |
|---|---:|---:|---:|---:|---:|---|
| 1 (Max-Q) | 3452.305 | 2259.253 | 1964.263 | 1.758× | 1.150× | 1.025–1.491× |

| B/C | Token tail | NCCL median GPU ms | UCCL median GPU ms | NCCL/UCCL | Complete hashes |
|---|---:|---:|---:|---:|---|
| 32/128 | 0 | 0.655936 | 0.670176 | 0.979× | identical |
| 64/128 | 0 | 0.586784 | 0.581664 | 1.009× | identical |
| 64/128 | 124 | 0.575424 | 0.584704 | 0.984× | identical |

The HT cases still issue zero network commands. Archive SHA256:
`e8e427b3c49128fc753ca37341701d8c39e3217090a6242d776032de71058e82`.

## Complete library/build gate

[CUDA CPU CI](https://github.com/0z5a/uccl/actions/runs/37477449458) passed at
`0fc87bf7`: the full standalone executable/extension, SM120/SM90 compilation,
contract rejection gates and NCCL-EP shared libraries with both native NCCL and
UCCL backends (LSA1, nodes1/2) all build against NCCL 2.30.4 and pinned RDMA SDK
`bd3282a1`. Upstream format/addressing checks passed; L4/GH200 execution jobs
were skipped.
