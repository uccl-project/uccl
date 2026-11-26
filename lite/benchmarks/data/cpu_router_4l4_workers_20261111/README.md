# Four-L4 CPU-router worker tuning (2026-11-11)

A later [transport-geometry follow-up](../cpu_router_ag_4l4_geometry_20261111/README.md)
raises the four-L4 AllGather block upper bound to 248 (180 actual at 1 GiB)
and uses a 4-MiB packet cap;
its 1-GiB paired latency falls to 37.577 ms out of place. The measurements
below remain the original worker-count sweep and eight-worker profile.
The [archived BF16 figures](../cpu_router_4l4_figures_20261111/README.md)
show the UCCL Lite router at every size. Automatic replay decisions below
are historical diagnostics and do not replace any plotted UCCL point.

This follow-up tests the [previous 23-worker-per-NUMA calibration](../cpu_router_4l4_apply_20261111/README.md)
on `mibura-sky-test-02`. The host has two Xeon Silver 4410Y sockets,
**12 physical cores / 24 logical CPUs per NUMA node**. The previous
23-worker setting occupies 23 logical CPUs per node, including SMT siblings.
The recommended AllGather-only setting below uses **8 workers per NUMA node,
16 total**, pinned to eight distinct physical cores per socket. The
[hardware/software record](hardware_software.txt) and raw logs show the
CPU mapping, router geometry, BF16 dtype, exact binary/NCCL hashes, and GPU
topology. All four L4 GPUs have mutual CUDA peer capability, but NCCL's
transport trace for this harness uses SHM on every Ring edge; see below.

Seven out-of-place scans compare **4, 8, 11, 12, 16, 20, and 23 workers per
NUMA node** over 64 KiB–1 GiB for AllReduce, AllGather, and ReduceScatter.
Two in-place scans use 8 and 12 workers over 1 KiB–1 GiB. Every size uses the
same one-process/four-GPU harness, NCCL 2.28.7 default local control,
complete BF16 tensor byte count per GPU, slowest-GPU CUDA-event timing,
five warmups, three backend-alternating rounds of ten timed calls, and a
complete output check on every call. Packet and transport settings otherwise
match the prior run: 512 KiB default packet cap, 128 KiB small AR/RS packets,
and 2 MiB AG packets from 4 MiB upward. `commands.json` records every exact
launch command and order. The [full per-size summary](summary.csv) is
regenerated and validated by `python3 summarize.py` from 29,100 paired
timed calls. Each size's NCCL control comes from its own paired run.

## Worker-count result

Representative **out-of-place 1-GiB** router latencies are in milliseconds:

| Workers / NUMA | AllReduce | AllGather | ReduceScatter |
| ---: | ---: | ---: | ---: |
| 4 | 237.03 | 42.90 | 106.88 |
| 8 | 143.35 | 42.87 | 55.83 |
| 12 | 96.52 | 42.78 | 54.50 |
| 20 | 86.93 | 43.12 | 54.71 |
| 23 | 86.13 | 42.86 | 54.71 |

The paired NCCL 1-GiB controls range about 86–88 ms for AR, 48–49 ms for
AG, and 48–49 ms for RS across these scans. Restricting AR to physical cores
does **not** make it faster: 12 workers take 96.52 ms versus 86.13 ms with
23 in this screening window, while NCCL remains near 86 ms. AllGather latency
changes little with worker count, so its 23-worker setting spends CPU threads
without an observable 1-GiB latency advantage. ReduceScatter is best among
the tested values near 12/23 workers, yet every 64 KiB–1 GiB point remains
slower than its paired NCCL control. No global worker value makes the three
collectives collectively faster.

At 1 KiB in-place AR, 8 workers showed a small paired lead in the full scan
and a dedicated 20-warmup recheck (47.23 vs 51.09 µs and 43.52 vs 45.06 µs).
The NCCL control shifted between runs, and the 23-worker recheck had a lower
absolute router latency (41.47 µs). This exploratory point is retained in
`ar_1k_in_recheck_workers_*`; it is not promoted into the AllGather-only
profile or a main result.

## Adapted AllGather profile

The final profile calibrates **only AllGather** with eight workers per NUMA,
20 warmups, three alternating paired rounds of ten calls, and full output
checks at all 21 sizes from 1 KiB–1 GiB in **both placements**. The
automatic policy is replayed in fresh processes over all three collectives.
Because its profile contains AG measurements only, AR and RS use NCCL.

| Placement and size | Router (µs) | Paired NCCL (µs) | NCCL/router |
| --- | ---: | ---: | ---: |
| Out of place, 512 KiB | 81.92 | 89.60 | 1.09× |
| In place, 512 KiB | 80.90 | 87.57 | 1.08× |
| Out of place, 1 GiB | 42,818.56 | 48,112.13 | 1.12× |
| In place, 1 GiB | 42,921.98 | 49,431.04 | 1.15× |

Both fresh-process replays select the router only at **512 KiB and
32/64/128/256/512 MiB and 1 GiB**: seven AG sizes in each placement. The
16-MiB point showed a gain in one worker scan but less than 3% in this final
paired AG calibration, so it remains NCCL. All AR/RS keys remain NCCL.
The [automatic selections](auto_selection.csv) and `ag_only_w8_auto_*.log`
show all 126 replay decisions; their one-call-per-round timings are selection
and correctness checks, not performance estimates. The 8-worker AG profile
is **configuration evidence for this exact host and binary**, not a universal
four-GPU default or a replacement for the eight-L4 profile.

The final profiles are `ag_only_w8_out.profile` and
`ag_only_w8_in.profile`. To create a new out-of-place profile from
`experimental/lite` after building, use fresh output names:

```bash
./build/uccl_lite_collectives --gpus 4 --fallback nccl --op allgather \
  --begin 1024 --end 1073741824 --warmup 20 --iterations 10 --repeats 3 \
  --workers-per-node 8 --large-ag-packet-bytes 2097152 \
  --calibrate build/new-ag-w8.profile --samples build/new-ag-w8-samples.csv
```

`--in-place` gives the other placement. To replay a profile, use
`--op all --profile PATH --workers-per-node 8` with the same packet settings.
The earlier 23-worker raw data and negative AR/RS variants remain archived;
this follow-up does not change the main FP32 figures or the global router
default.

## NCCL SHM Ring traffic on this host

The diagnostic [`nccl_shm_default_probe.log`](nccl_shm_default_probe.log)
shows two Ring channels with order `0→1→2→3→0`, each edge labeled
`via SHM/direct/direct`, despite CUDA peer capability. The separate
[`nccl_shm_ring_simple_probe.log`](nccl_shm_ring_simple_probe.log) sets
`NCCL_ALGO=Ring,NCCL_PROTO=Simple` and confirms the same SHM transport.
Both probes run the checked NCCL AllGather control at 1 MiB; their timings
are **not** included in any performance summary. The default probe confirms
SHM links, not which collective algorithm NCCL selected for every size.

For four GPUs, let `B` be the complete output bytes **per GPU** and each
source contribution be `S=B/4`. Logical useful payload, excluding protocol
headers, caching effects and local GPU copies, is:

| Transport work, four GPUs combined | Host-packet CPU router | Ring over NCCL SHM |
| --- | ---: | ---: |
| GPU→host payload | `B` (one upload per source) | `3B` (three Ring hops per source) |
| Host→GPU payload | `3B` (three remote source shards per GPU) | `3B` (three Ring receives per GPU) |
| Total GPU PCIe endpoint payload | `4B` | `6B` |
| Host-memory useful writes + reads | `B + 3B = 4B` | `3B + 3B = 6B` |

For `B=1 GiB`, the router logically uploads 1 GiB and reads 3 GiB from host
memory; Ring+SHM writes 3 GiB to shared host buffers and reads 3 GiB back.
The router saves **2 GiB of GPU→host payload** and 2 GiB of total useful
host-memory traffic, a one-third reduction in total endpoint payload.
The router's CPU workers publish packet readiness for this copy operation;
they do not duplicate each shard into a separate host buffer per destination.
Actual PCIe/DRAM counter bytes can differ from these useful-payload counts.
