<div align="center">

<p align="center"> <img src="./docs/assets/uccl-lite-logo.png" alt="UCCL Lite logo with a navy lightning outline and gold accents" width="300"> </p>

# UCCL Lite

**Collectives and communication–GEMM fusion for PCIe GPU nodes.**

[Results](#measured-results) · [CPU router](#optional-cpu-router) · [Documentation](#documentation)

</div>

## Testbeds

| # | GPUs | GPU P2P |
| --- | --- | --- |
| 1 | 8 × NVIDIA L4 | Partial P2P |
| 2 | 8 × NVIDIA L4 | Without P2P |
| 3 | 4 × NVIDIA L4 | With P2P |
| 4 | 2 × NVIDIA RTX PRO 6000 | With P2P |

UCCL Lite is an experimental collection of CUDA collective kernels, an
optional CPU router, fusion kernels, and hardware probes. The direct peer paths
target **four NVIDIA L4 GPUs on a PCIe Gen4 dual-socket node**; the network
paths target **two such nodes**, with two NIC rails per node. The CPU router is
a separate module for 1–32 local GPUs. Most collective and fusion paths
use MSCCL++; standalone peer AG/RS uses CUDA peer access.

The eight-L4 results were measured on a separate host with four GPUs per
NUMA node. A third, **8xL4 SHM**, testbed disables GPU peer transport on
that host by policy. Fresh four-L4, two-node, and eight-L4 dense fusion
measurements appear below with their own timing scopes. Small messages
and small GEMMs can lose to NCCL; negative results are retained.

## What is included

| Component | Purpose |
| --- | --- |
| [Profiler](docs/hardware-profiler.md) | Inspect GPU, PCIe, and NUMA topology and measure transfer bandwidth. |
| [Single-node collectives](docs/collectives.md) | AllReduce, AllGather, and ReduceScatter on one node. |
| [Multi-node collectives](docs/multinode.md) | Collective implementations across nodes using local GPU access and network rails. |
| [CPU router](docs/cpu-router.md) | Optional NUMA-aware path for local collectives, selected from measured benefit. |
| [Fusion kernels](docs/fusion.md) | Combine collective communication with GEMM and MoE stages. |

The RTX PRO 6000 BF16 peer path shares inline copy/reduction primitives with
four-L4 FP32 peer collectives and the eight-L4 BF16 local fallback. One prepared
peer executor owns the two/four-rank CUDA schedule; the joint layer owns
validation and calibrated backend selection. See the
[shared implementation and refactor checks](docs/retained-optimizations.md#shared-collective-components).

## Measured results

### End-to-end serving and training

#### Testbed 1

These historical figures are **not all stock NCCL drop-in results**: serving
and DDP use the experimental NCCL proxy; FSDP uses a separate RS communicator
and an application communication hook. See the
[original FSDP boundary](integration/nccl_lite/e2e_large_messages_8l4_20261115.md#fsdp-compatibility-boundary).
The [calibrated eight-L4 drop-in](integration/nccl_lite/DROPIN.md) has its own
stock-FSDP validation and fresh performance records; it does not inherit these
figures.

<table>
  <tr><th>Serving</th><th>Training</th></tr>
  <tr>
    <td width="50%" valign="top"><img src="docs/assets/sglang-large-prefill-e2e-8l4.svg" alt="GLM-4.5-Air-FP8 serving latency and latency reduction versus NCCL on eight L4 GPUs" width="100%"></td>
    <td width="50%" valign="top"><img src="docs/assets/training-large-message-e2e-8l4.svg" alt="DDP and experimental adapted FSDP training-step latency and latency reduction versus NCCL on eight L4 GPUs" width="100%"></td>
  </tr>
</table>

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

### Multi-node collectives

#### Testbed 3 × 2 nodes

<table>
  <tr><th>AllReduce</th><th>AllGather</th><th>ReduceScatter</th></tr>
  <tr>
    <td width="33%"><img src="docs/assets/collectives-2x4l4-allreduce-small-fixed.svg" alt="Two-node FP32 AllReduce: 1 KiB–1 MiB latency and 1 MiB–1 GiB bus throughput, UCCL Lite versus NCCL" width="100%"></td>
    <td width="33%"><img src="docs/assets/collectives-2x4l4-allgather-host-router.svg" alt="Two-node FP32 AllGather: 1–8 MiB latency and 1 MiB–1 GiB bus throughput, UCCL Lite CPU router versus paired NCCL Ring+Simple" width="100%"></td>
    <td width="33%"><img src="docs/assets/collectives-2x4l4-reducescatter-direct-medium.svg" alt="Two-node ReduceScatter latency and bus throughput, UCCL Lite versus NCCL" width="100%"></td>
  </tr>
</table>

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
