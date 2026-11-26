# Four-L4 analysis checks, November 13, 2026

These short local confirmation runs preceded the
[pipelined ReduceScatter study](../optimization_4l4_20261113/README.md) and
[small/medium follow-up](../small_medium_4l4_20261113/README.md). They describe
the archived source in `identity.json`, not the current implementation.

Hardware: `mibura-sky-test-02`, four NVIDIA L4s, dual Xeon Silver 4410Y,
NUMA pairs (0,1)/(2,3), PCIe Gen4 x16 with peer access across all GPU pairs,
driver 595.58.03 and CUDA 12.9. The hardware probe used CUDA-allocated pinned
buffers without forced NUMA placement, two warmups, ten iterations and three
repetitions. Host-to-device and device-to-host directions were measured
separately.

## Recorded measurements and limitations

| Resource/operation | Measured result |
| --- | ---: |
| Isolated directed GPU peer copies | 22.63–22.64 GB/s |
| Concurrent ring 0→1→3→2→0, four sources | 81.904 GB/s aggregate |
| Concurrent H2D / D2H, four GPUs | 99.450 / 55.641 GB/s aggregate |
| FP32 AllReduce, 1 GiB full input/GPU | ring CE 80.200 ms (host timer) |
| FP32 AllGather, 1 GiB full output/GPU | peer 45.530 ms / NCCL 47.749 ms |
| FP32 ReduceScatter, 1 GiB full input/GPU | peer 48.603 ms / NCCL 46.661 ms |
| FP32 packet AllGather, separate paired window, 1 GiB | packet 36.429 ms / peer 45.539 ms / NCCL 47.564 ms |

The AR log also retains hierarchy CE 88.546 ms, pipelined hierarchy 93.648 ms,
hierarchy SM 123.247 ms, tree CE 221.215 ms and ring plus flat gather 83.681 ms.
Neither fewer hierarchy phases nor flat gathering beat the ring in this run.
The packet harness still printed `selected,...,peer`; that label did not
automatically dispatch the faster optional packet candidate.

These traffic patterns do not share one universal bandwidth ceiling. Dividing
the measured ring rate by four gives a same-traffic reference of 20.476 GB/s
per GPU, corresponding to 78.66 ms for AR and 39.33 ms for AG/RS at 1 GiB under
the conventional 1.5N/t and 0.75N/t bus-bandwidth normalization. Packet AG's
36.429 ms uses a different host-memory route. Isolated peer-copy rates and
separate DMA directions do not establish simultaneous duplex, NUMA or mapped
host-memory limits. No graph/device-triggered latency floor or FP32 CPU
reduction performance was measured by this audit.

## Reproduction

Commands executed sequentially on idle `mibura-sky-test-02`:

```bash
make hardware_profiler peer_ag_rs allreduce
./build/uccl_lite_hardware_profiler --devices 0,1,2,3 \
  --peer-order 0,1,3,2 --gpu-bytes 16777216 --cpu-bytes 8388608 \
  --iterations 10 --warmups 2 --repeats 3 --threads-per-node 4 > hardware.txt
./build/uccl_lite_peer_ag_rs -b 67108864 -e 1073741824 \
  -w 2 -n 5 -r 3 > peer_large.log
./build/uccl_lite_allreduce -b 1073741824 -e 1073741824 \
  -w 2 -n 3 > ar_1g.log
make peer-ag-rs-packet
./build/uccl_lite_peer_ag_rs_packet --packet-router-ag \
  --packet-router-per-source -b 1073741824 -e 1073741824 \
  -w 2 -n 5 -r 3 > packet_1g.log
```

Paths above are from the lite directory; outputs were redirected into this
record directory. `identity.json` records revision/source/library and binary
hashes; the packet binary is updated after its Makefile rebuild.

AR is FP32 in place, one forked process per GPU, maximum-rank host call latency.
AG/RS are FP32 out of place, one process, maximum-GPU event latency including
host submission gaps. Both harnesses check full output outside timing on every
warmup/timed collective. AR's null control is intentionally not checked.
Peer baseline is linked NCCL 2.28.7 default. There is no paired NCCL AR run here.
AR reports means of 3 calls; peer reports median of 3 run means of 5 calls.
Raw variants, including slower paths, remain in logs. Packet and peer selection
are harness candidate comparisons; never pool their NCCL samples across runs.
