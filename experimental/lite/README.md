<div align="center">

<p align="center">
  <img src="./docs/assets/uccl-lite-logo.png" alt="UCCL Lite logo with a navy lightning outline and gold accents" width="500"><br>
  <strong>Efficient Communication Library for PCIe GPU nodes.</strong>
</p>

</div>

## Design principles

- **Exploit PCIe hierarchy.** Map collective stages to PCIe and NUMA groups.
  Reduce within groups before exchanging across them, then distribute results
  locally to limit cross-socket traffic.
- **Specialize by size.** Cut launches and handshakes for small messages, where
  fixed overhead dominates. Pipeline large transfers to sustain bandwidth and
  amortize coordination costs.
- **Put CPUs to work.** When P2P is limited, our CPU router uses spare cores and
  NUMA-local memory to combine contributions before forwarding and serve
  multiple destinations from one staged copy, reducing redundant transfers.
- **Overlap communication and compute.** Start GEMM on arriving inputs and
  communication on ready outputs. Choose chunk sizes that expose overlap
  without fragmenting GEMM into inefficient small operations.

## Testbeds

<div align="center">

<table align="center">
  <tr><th>#</th><th>GPUs</th><th>GPU P2P</th><th>PCIe switch P2P</th><th>PCIe root complex P2P</th></tr>
  <tr><td>1</td><td>8 × NVIDIA L4</td><td>Partial support</td><td align="center">✓</td><td align="center">✗</td></tr>
  <tr><td>2</td><td>8 × NVIDIA L4</td><td>Not supported</td><td align="center">✗</td><td align="center">✗</td></tr>
  <tr><td>3</td><td>4 × NVIDIA L4</td><td>Full support</td><td align="center">✗</td><td align="center">✓</td></tr>
  <tr><td>4</td><td>2 × NVIDIA RTX PRO 6000</td><td>Full support</td><td align="center">✓</td><td align="center">✗</td></tr>
  <tr><td>5</td><td>8 × NVIDIA RTX PRO 6000</td><td>Full support</td><td align="center">✓</td><td align="center">✓</td></tr>
</table>

</div>

## What is included

<div align="center">

<table align="center">
  <tr><th>Component</th><th>Purpose</th></tr>
  <tr><td><a href="docs/hardware-profiler.md">Profiler</a></td><td>Inspect GPU, PCIe, and NUMA topology and measure transfer bandwidth.</td></tr>
  <tr><td><a href="docs/collectives.md">Single-node collectives</a></td><td>AllReduce, AllGather, and ReduceScatter on one node.</td></tr>
  <tr><td><a href="docs/multinode.md">Multi-node collectives</a></td><td>Collective implementations across nodes using local GPU access and network rails.</td></tr>
  <tr><td><a href="docs/cpu-router.md">CPU router</a></td><td>Optional NUMA-aware path for local collectives, selected from measured benefit.</td></tr>
  <tr><td><a href="docs/fusion.md">Fusion kernels</a></td><td>Combine collective communication with GEMM and MoE stages.</td></tr>
</table>

</div>

## Measured results

### End-to-end serving and training

#### Testbed 1

<table>
  <tr><th>Serving</th><th>Training</th></tr>
  <tr>
    <td width="50%" valign="top"><img src="docs/assets/sglang-large-prefill-e2e-8l4.svg" alt="GLM-4.5-Air-FP8 serving latency and latency reduction versus NCCL on eight L4 GPUs" width="100%"></td>
    <td width="50%" valign="top"><img src="docs/assets/training-large-message-e2e-8l4.svg" alt="DDP and experimental adapted FSDP training-step latency and latency reduction versus NCCL on eight L4 GPUs" width="100%"></td>
  </tr>
</table>

#### Testbed 3: four L4 GPUs

GPT-2-XL BF16 · experimental parameter-sharded serving and checkpointed training.

<table>
  <tr><th>Serving</th><th>Training</th></tr>
  <tr>
    <td width="50%" valign="top"><img src="docs/assets/serving-e2e-1x4l4.svg" alt="Four L4 GPT-2-XL BF16 parameter-sharded whole HTTP batch latency, packet AG plus peer RS versus default NCCL" width="100%"></td>
    <td width="50%" valign="top"><img src="docs/assets/training-e2e-1x4l4.svg" alt="Four L4 GPT-2-XL BF16 checkpointed full SGD step latency, packet AG plus peer RS versus default NCCL" width="100%"></td>
  </tr>
</table>

#### Testbed 3 × 2 nodes

DP2 × shard4; global batch doubles with GPU count (weak scaling).

<table>
  <tr><th>Serving</th><th>Training</th></tr>
  <tr>
    <td width="50%" valign="top"><img src="docs/assets/serving-e2e-2x4l4.svg" alt="Two four-L4 replicas GPT-2-XL BF16 whole distributed HTTP batch latency versus NCCL, DP2 and shard4" width="100%"></td>
    <td width="50%" valign="top"><img src="docs/assets/training-e2e-2x4l4.svg" alt="Two four-L4 replicas GPT-2-XL BF16 checkpointed full synchronized SGD step latency versus NCCL, DP2 and shard4" width="100%"></td>
  </tr>
</table>

[Experiment setup, numerical checks and raw data](integration/l4_e2e/README.md).

#### Testbed 4

Both workloads below use an explicit parameter-sharded application adapter,
not a drop-in NCCL replacement.

<table>
  <tr><th>Serving</th><th>Training</th></tr>
  <tr>
    <td width="50%" valign="top"><img src="docs/assets/serving-large-message-e2e-pro6000.svg" alt="Pretrained GPT-2-large experimental weight-sharded HTTP serving on two RTX PRO 6000 GPUs, UCCL Lite versus NCCL" width="100%"></td>
    <td width="50%" valign="top"><img src="docs/assets/training-large-message-e2e-pro6000.svg" alt="GPT-2-large experimental block fine-tuning full training steps on two RTX PRO 6000 GPUs, UCCL Lite versus NCCL" width="100%"></td>
  </tr>
</table>

The [PRO 6000 experiment](integration/peer_e2e/README.md) records communication
fractions, numerical checks, raw repetitions and the parameter-sharded adapter's
scope. Its per-layer BF16 AG/RS messages are 37.53 MiB; this is a separate
workload from Testbed 1's SGLang serving and DDP/FSDP training.

### Single-node collectives

#### Testbed 1

<table>
  <tr><th>AllReduce</th><th>AllGather</th><th>ReduceScatter</th></tr>
  <tr>
    <td width="33%"><img src="docs/assets/collectives-8l4-allreduce-overview.svg" alt="8xL4 GPU AllReduce: 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput, UCCL Lite, NCCL CE (tuned), NCCL SM (tuned), NCCL (default)" width="100%"></td>
    <td width="33%"><img src="docs/assets/collectives-8l4-allgather-overview.svg" alt="8xL4 GPU AllGather: 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput, UCCL Lite, NCCL CE (tuned), NCCL SM (tuned), NCCL (default)" width="100%"></td>
    <td width="33%"><img src="docs/assets/collectives-8l4-reducescatter-overview.svg" alt="8xL4 GPU ReduceScatter: 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput, UCCL Lite, NCCL CE (tuned), NCCL SM (tuned), NCCL (default)" width="100%"></td>
  </tr>
</table>

#### Testbed 2

<table>
  <tr><th>AllReduce</th><th>AllGather</th><th>ReduceScatter</th></tr>
  <tr>
    <td width="33%"><img src="docs/assets/collectives-8l4-shm-allreduce-overview.svg" alt="8xL4 SHM AllReduce: 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput, UCCL Lite, NCCL CE (tuned), NCCL SM (tuned), NCCL (default)" width="100%"></td>
    <td width="33%"><img src="docs/assets/collectives-8l4-shm-allgather-overview.svg" alt="8xL4 SHM AllGather: 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput, UCCL Lite, NCCL CE (tuned), NCCL SM (tuned), NCCL (default)" width="100%"></td>
    <td width="33%"><img src="docs/assets/collectives-8l4-shm-reducescatter-overview.svg" alt="8xL4 SHM ReduceScatter: 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput, UCCL Lite, NCCL CE (tuned), NCCL SM (tuned), NCCL (default)" width="100%"></td>
  </tr>
</table>

#### Testbed 3

<table>
  <tr><th>AllReduce</th><th>AllGather</th><th>ReduceScatter</th></tr>
  <tr>
    <td width="33%"><img src="docs/assets/collectives-4l4-allreduce-all-sizes.svg" alt="AllReduce: 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput, UCCL Lite, NCCL CE (tuned), NCCL SM (tuned), NCCL (default)" width="100%"></td>
    <td width="33%"><img src="docs/assets/collectives-4l4-allgather-all-sizes.svg" alt="AllGather: 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput, UCCL Lite, NCCL CE (tuned), NCCL SM (tuned), NCCL (default)" width="100%"></td>
    <td width="33%"><img src="docs/assets/collectives-4l4-reducescatter-all-sizes.svg" alt="ReduceScatter: 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput, UCCL Lite, NCCL CE (tuned), NCCL SM (tuned), NCCL (default)" width="100%"></td>
  </tr>
</table>

#### Testbed 4

<table>
  <tr><th>AllReduce</th><th>AllGather</th><th>ReduceScatter</th></tr>
  <tr>
    <td width="33%"><img src="docs/assets/collectives-pro6000x2-allreduce-overview.svg" alt="PRO6000 x2 AllReduce: UCCL Lite, NCCL CE (tuned), NCCL SM (tuned), NCCL (default P2P)" width="100%"></td>
    <td width="33%"><img src="docs/assets/collectives-pro6000x2-allgather-overview.svg" alt="PRO6000 x2 AllGather: UCCL Lite, NCCL CE (tuned), NCCL SM (tuned), NCCL (default P2P)" width="100%"></td>
    <td width="33%"><img src="docs/assets/collectives-pro6000x2-reducescatter-overview.svg" alt="PRO6000 x2 ReduceScatter: UCCL Lite, NCCL CE (tuned), NCCL SM (tuned), NCCL (default P2P)" width="100%"></td>
  </tr>
</table>

#### Testbed 5

<table>
  <tr><th>AllReduce</th><th>AllGather</th><th>ReduceScatter</th></tr>
  <tr>
    <td width="33%"><a href="benchmarks/data/pro6000_nccl_baselines_cpp_20261116/README.md"><img src="docs/assets/collectives-pro6000x8-allreduce-fp32-nccl-baselines.svg" alt="Eight RTX PRO 6000 FP32 AllReduce: UCCL Lite, NCCL (tuned SM/CE), and NCCL (default); median latency and bus throughput over five independent runs" width="100%"></a></td>
    <td width="33%"><a href="benchmarks/data/pro6000_nccl_baselines_cpp_20261116/README.md"><img src="docs/assets/collectives-pro6000x8-allgather-fp32-nccl-baselines.svg" alt="Eight RTX PRO 6000 FP32 AllGather: UCCL Lite, NCCL (tuned SM/CE), and NCCL (default); median latency and bus throughput over five independent runs" width="100%"></a></td>
    <td width="33%"><a href="benchmarks/data/pro6000_nccl_baselines_cpp_20261116/README.md"><img src="docs/assets/collectives-pro6000x8-reducescatter-fp32-nccl-baselines.svg" alt="Eight RTX PRO 6000 FP32 ReduceScatter: UCCL Lite, NCCL (tuned SM/CE), and NCCL (default); median latency and bus throughput over five independent runs" width="100%"></a></td>
  </tr>
</table>

### Multi-node collectives

#### Testbed 3 × 2 nodes

<table>
  <tr><th>AllReduce</th><th>AllGather</th><th>ReduceScatter</th></tr>
  <tr>
    <td width="33%"><img src="docs/assets/collectives-2x4l4-allreduce-nccl-tuned.svg" alt="Two-node FP32 AllReduce: 1 KiB–1 MiB latency and 1 MiB–1 GiB throughput, UCCL Lite, NCCL SM (tuned), NCCL CE (tuned), NCCL (default)" width="100%"></td>
    <td width="33%"><a href="benchmarks/data/ag_combined_latest_2x4l4_20261117/README.md"><img src="docs/assets/collectives-2x4l4-allgather-combined.svg" alt="Two-node FP32 AllGather: latest measured combined CE/SM implementation, 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput; recorded NCCL controls retained" width="100%"></a></td>
    <td width="33%"><img src="docs/assets/collectives-2x4l4-reducescatter-nccl-tuned.svg" alt="Two-node ReduceScatter latency and throughput, UCCL Lite, NCCL SM (tuned), NCCL CE (tuned), NCCL (default)" width="100%"></td>
  </tr>
</table>

For **two four-L4 hosts, FP32 eager AllReduce**, select the experimental
**host one-shot path below 32 KiB** with `--host-gather 1 --host-oneshot 1`.
The tested sub-32-KiB sizes are 1, 2, 4, 8, and 16 KiB: latency is
**27.32 vs 36.22 µs at 1 KiB** and **32.43 vs 35.99 µs at 8 KiB** against
the current GPU hierarchy. At 32 KiB it loses (42.11 vs 39.63 µs), so retain
the existing hybrid selection from 32 KiB upward by omitting `--host-oneshot`.
This is an explicit selection for the recorded topology, not automatic dispatch;
sizes below 1 KiB and between the measured points are not performance-validated.

The path uses copy-engine D2H of raw inputs, MPI/UCX host-packet exchange, and
an SM kernel that reads mapped host memory and computes the full sum, with no
GPU P2P payload traffic. The [fresh three-variant comparison and reproduction
commands](benchmarks/data/host_oneshot_small_2x4l4_20261117/README.md) include
three rounds of 20 warmups and 30 checked calls per size, measured from host
enqueue to completion on the slowest rank. Values are medians of round means,
not a best-variant-per-size curve. The AllReduce figure above retains its
historical measurements; this new comparison does not rerun NCCL.

### Fusion

#### Testbed 3

##### Dense AG+GEMM, GEMM+RS, GEMM+AR

![Complete-operation latency at every dense shape, UCCL Lite versus NCCL and cuBLAS](docs/assets/dense-fusion-4l4-refresh.svg)

##### Expert-parallel MoE

![Complete-layer latency at every token count, UCCL Lite versus NCCL and grouped cuBLAS](docs/assets/moe-fusion-4l4-refresh.svg)

#### Testbed 1

##### GEMM + AllReduce

![Eight-L4 GEMM plus AllReduce decode latency bars for four model shapes at M=1,32,64](docs/assets/fusion-8l4-model-gemm-ar-decode-m64-labeled-bars.svg)

![Eight-L4 GEMM plus AllReduce prefill latency bars for four model shapes at M=2048,4096,8192](docs/assets/fusion-8l4-model-gemm-ar-prefill-labeled-bars.svg)

##### AllGather + GEMM

![Eight-L4 AllGather plus GEMM decode latency bars for four model shapes at M=8,32,64](docs/assets/fusion-8l4-model-ag-gemm-decode-labeled-bars.svg)

![Eight-L4 AllGather plus GEMM prefill latency bars for four model shapes at M=2048,4096,8192](docs/assets/fusion-8l4-model-ag-gemm-prefill-labeled-bars.svg)

##### GEMM + ReduceScatter

![Eight-L4 GEMM plus ReduceScatter decode latency bars for four model shapes at M=1,32,64](docs/assets/fusion-8l4-model-gemm-rs-decode-optimized-labeled-bars.svg)

![Eight-L4 GEMM plus ReduceScatter prefill latency bars for four model shapes at M=2048,4096,8192](docs/assets/fusion-8l4-model-gemm-rs-prefill-optimized-labeled-bars.svg)

## Optional CPU router

### Motivation

When GPU Direct P2P is unavailable between GPU groups, their data already
needs a host-mediated path. The CPU router uses that host-memory stop to do
useful work with otherwise idle CPU cores: reduce packets before forwarding
them and share a packet with multiple destinations. Reusable packet operators
also support packing and deduplication. Routing is most promising when PCIe
transfers constrain the collective while NUMA-local host
memory and spare CPU cores can keep up with the packet flow. Fast direct P2P
or contention for host memory can erase that advantage.

### High-level design

The optional CPU router exchanges packets through NUMA-local pinned host
memory. **Forward** moves a packet, **Multicast** sends one packet to multiple
destinations, and **Process (CPU)** transforms packet contents with Reduce,
Pack, and Deduplicate operators. Collectives compose these operations, with
packet slots and reader credits preventing reuse until consumers finish.

```mermaid
flowchart TB
    R["CPU Router"]
    F["Forward"]
    PC["Process (CPU)"]
    M["Multicast"]
    RED["Reduce"]
    PACK["Pack"]
    DEDUP["Deduplicate"]

    R --> F
    R --> PC
    R --> M
    PC --> RED
    PC --> PACK
    PC --> DEDUP

    classDef router fill:#164F86,stroke:#164F86,color:#fff
    classDef stage fill:#E5F0F8,stroke:#7CA8C8,color:#17364F
    classDef operator fill:#F4F7FA,stroke:#A8BAC9,color:#17364F
    class R router
    class F,PC,M stage
    class RED,PACK,DEDUP operator
```

The `Process (CPU)` operators can be used in different collectives:

- **Reduce** combines contributions for AllReduce and ReduceScatter.
- **Pack** assembles fragments for AllGather and token buffers for
  dispatch/combine.
- **Deduplicate** removes repeated token records in dispatch/combine. The
  included fixed-width operator is intended for small routing metadata.

## Documentation

- [Build targets and correctness commands](docs/build.md)
- [Hardware profiler methods and measurements](docs/hardware-profiler.md)
- [Collective algorithms](docs/collectives.md) and [two-node design](docs/multinode.md)
- [Fusion design and workloads](docs/fusion.md)
- [NCCL drop-in and SGLang integration](integration/nccl_lite/README.md)
- [Full measurement files](benchmarks/README.md) and [validation limits](docs/validation.md)

UCCL Lite provides specialized research kernels and benchmarks, with an
experimental NCCL compatibility layer and SGLang hooks under `integration/`.
The project follows the repository [license](../../LICENSE); the adapted CUTLASS wrapper retains
its [upstream notice](NOTICE.md).
