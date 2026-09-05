# L40S benchmark report — SM-budget revision (2026-09-04)

Shim (ukernel) vs native NCCL on the two-node L40S cluster, built from
`uk-300` (HEAD `14391f89`; fused-RS thread sync, per-fifo TaskArgs pools,
single-coordinator idle exit). Medians of 3, nccl-tests `-n 10 -w 2`,
validation on, all cells 0 wrong.

## Environment

- node5 `10.31.154.11`, node6 `10.31.154.12`: 8× NVIDIA L40S each (PCIe
  Gen5 x16, no NVLink), ConnectX-6 HDR dual-port 200G, driver 610.57.04.
- Shim: `experimental/ukernel` (`build/nccl/lib`). Native: system NCCL
  2.31.2+cuda13.3.
- nccl-tests MPI build; AllToAll via `bench/alltoall_perf`
  (`ncclAllToAll`) with per-node local device wrapper (shim keys its OOB
  leader election on the CUDA device id, so cross-node ranks must use
  device 0..3 of their own node).
- Same-node S2/S4/S8 on node5; cross-node X4/X8/X16 split 2+2/4+4/8+8.
  Cross-node shim adds `UK_CCL_RDMA_FUSED_MODE=proxy`.
- Raw logs: `/tmp/l40s_final`, `/tmp/l40s_small_final`,
  `/tmp/l40s_rotation`, `/tmp/l40s_sweep`, `/tmp/l40s_confirm`.

## SM-budget design rule

Shim blocks `b` must be `<=` native coll channels `B` at the same
placement; every cell reports the best measured `b` in that budget
(selected at 256 MiB, ties go to fewer SMs). AllToAll is a pure CE/IPC
path and is reported at **0 worker SMs**. Small messages (1M/4M) on this
platform need the budget maximum for AllReduce/ReduceScatter
(`b=4` on S2/S4, `b=2` on S8/X*), so the tables below list per-size `b`.

Measured budgets (native coll channels, `NCCL_DEBUG=INFO`):

| placement | S2 | S4 | S8 | X4 | X8 | X16 |
|---:|---:|---:|---:|---:|---:|---:|
| B | 4 | 4 | 2 | 2 | 2 | 2 |

Best `b*` (256 MiB basis): AllReduce 1 everywhere; ReduceScatter S2=4,
S4=2, S8/X4/X8=1, X16=2; AllGather and AllToAll are pure CE paths at
**0 worker SMs** (the shim publishes AllGather's own shard with a
user-stream copy-engine memcpy, so no worker op exists in the plan).

## AllReduce — busbw GB/s (median of 3), shim (unfused CE/IPC) vs native

| size | S2 sh/nat | S4 sh/nat | S8 sh/nat | X4 sh/nat | X8 sh/nat | X16 sh/nat |
|---|---:|---|---:|---|---:|---|---:|---|---:|
| 1M | 8.31/14.48 | 5.68/15.13 | 2.92/9.37 | 4.99/11.19 | 3.09/5.11 | 1.41/5.06 |
| 4M | 19.96/20.18 | 13.03/21.27 | 6.77/14.53 | 8.73/13.54 | 7.49/14.41 | 4.62/12.07 |
| 16M | 23.45/21.14 | 23.91/21.85 | 12.90/14.27 | 11.49/13.73 | 11.49/14.49 | 9.70/15.01 |
| 64M | 24.46/21.39 | 24.61/21.68 | 15.22/14.16 | 11.53/13.60 | 11.52/14.31 | 11.77/14.46 |
| 256M | 25.61/21.48 | 25.64/21.72 | 16.02/14.43 | 10.97/13.67 | 11.06/14.43 | 10.96/14.64 |

Per-size shim blocks: ≤4M uses budget max (4 on S2/S4, 2 elsewhere);
16M uses `b=1` (X16 uses `b=2`); 64M/256M use `b=1` (X16 uses `b=1`
at 256M and `b=2` at 16-64M). Native rows are unchanged across sizes.

## ReduceScatter — busbw GB/s (median of 3), shim vs native

| size | S2 sh/nat | S4 sh/nat | S8 sh/nat | X4 sh/nat | X8 sh/nat | X16 sh/nat |
|---|---:|---|---:|---|---:|---|---:|---|---:|
| 1M | 6.05/11.18 | 4.38/12.71 | 2.33/8.95 | 3.97/9.65 | 2.41/8.27 | 1.23/5.92 |
| 4M | 12.90/17.33 | 10.55/19.53 | 5.66/13.78 | 7.51/12.45 | 6.37/13.61 | 3.94/12.44 |
| 16M | 20.21/19.13 | 20.68/21.10 | 10.66/13.52 | 11.10/12.82 | 11.30/13.88 | 8.20/14.22 |
| 64M | 23.30/19.60 | 23.62/21.08 | 14.56/13.57 | 11.30/12.70 | 11.50/13.77 | 11.60/14.07 |
| 256M | 24.98/19.87 | 24.97/21.21 | 15.77/13.62 | 10.99/12.79 | 11.08/13.76 | 11.09/13.98 |

Shim blocks: S2=4, S4=2, S8/X4/X8=1 (≤4M uses 2 on S8/X*), X16=2.

## AllGather — busbw GB/s (median of 3), shim CE path (0 SM) vs native

| size | S2 sh/nat | S4 sh/nat | S8 sh/nat | X4 sh/nat | X8 sh/nat | X16 sh/nat |
|---|---:|---|---:|---|---:|---|---:|---|---:|
| 1M | 9.61/11.09 | 6.59/12.53 | 3.53/8.55 | 5.89/9.50 | 3.62/7.82 | 1.83/5.74 |
| 4M | 17.65/18.12 | 14.75/19.71 | 8.57/13.82 | 10.19/12.78 | 8.51/13.90 | 5.90/11.87 |
| 16M | 21.81/20.07 | 23.05/21.39 | 14.39/14.30 | 11.38/13.23 | 11.52/14.68 | 11.45/14.97 |
| 64M | 23.14/20.56 | 24.15/21.38 | 14.77/14.27 | 11.60/13.16 | 11.34/14.55 | 11.61/14.88 |
| 256M | 23.84/20.79 | 25.10/21.50 | 15.65/14.36 | 10.92/13.16 | 11.26/14.52 | 11.20/14.83 |

The CE-external AllGather matches the previous worker-local-copy numbers
within noise at 0 worker SMs.

## AllToAll — busbw GB/s (median of 3), shim pure CE path (0 SM)

| size | S2 sh/nat | S4 sh/nat | S8 sh/nat | X4 sh/nat | X8 sh/nat | X16 sh/nat |
|---|---:|---|---:|---|---:|---|---:|---|---:|
| 1M | 8.1/10.2 | 8.4/12.7 | 4.6/4.6 | 6.3/6.8 | 4.9/4.9 | 1.9/3.4 |
| 4M | 16.4/18.1 | 16.7/17.9 | 6.0/5.9 | 8.4/9.4 | 5.8/6.5 | 2.0/4.3 |
| 16M | 21.8/22.3 | 14.2/20.3 | 6.5/6.8 | 9.2/10.8 | 6.2/6.6 | 2.1/4.4 |
| 64M | 23.7/23.6 | 15.4/20.3 | 6.7/6.6 | 8.6/11.3 | 6.2/6.4 | 2.0/4.6 |
| 256M | 23.8/24.0 | 15.6/20.0 | 6.6/7.0 | 9.0/11.3 | 6.1/6.4 | 2.0/4.7 |

### Rotation A/B at 256M (CE path, 0 SM), medians

| np | rot0 (ascending) | rot1 (default rotated) |
|---:|---:|---:|
| 2 | 23.8 | 23.8 |
| 4 | 15.2 | 15.6 |
| 8 | 6.5 | 6.7 |

Rotation is neutral at np2 and +2-3% at np4/np8; keep default on.

## Fusion ablation (2026-09-04)

Fusion is **negative on L40S**. Fused config =
`UK_CCL_DEV_BLOCKS=32` + `FUSE_REDUCE_COPY=1 FUSE_AG_COPY=1
LARGE_TILES=16 TILE_MIN_BYTES=8M IPC_BATCH=16`; unfused = `DEV_BLOCKS=32`
only. AllReduce OOP busbw (GB/s), unfused / fused, medians of 3:

| size | np2 | np4 | np8 |
|---|---:|---:|---:|
| 16M | 23.5 / 20.5 | 24.0 / 12.9 | 13.8 / 7.4 |
| 64M | 24.4 / 22.2 | 24.6 / 17.9 | 14.7 / 10.1 |
| 256M | 25.6 / 22.9 | 25.5 / 19.7 | 15.6 / 10.2 |

Fused@32 is 11-13% slower at 2 ranks and 20-46% slower at 4-8 ranks.
With PCIe as the bottleneck the CE/IPC copy path beats the fused
device-copy path (SM LD/ST to peer), so the shipped L40S AllReduce column
stays unfused CE/IPC; fusion is a B300/NVLink story.

## Blocks sensitivity (formal table, medians of 3, 2026-09-05)

AllReduce/ReduceScatter at 256M (b\* selection) and 4M (small-message
justification for the budget maximum). Raw logs: `/tmp/l40s_blocks`.

### AllReduce — busbw GB/s

| placement | size | b=1 | b=2 | b=4 |
|---|---:|---:|---:|---:|
| S2 | 256M | **25.60** | 25.61 | 25.61 |
| S2 | 4M | 16.58 | 18.53 | **19.56** |
| S4 | 256M | **25.64** | 25.49 | 25.64 |
| S4 | 4M | 10.60 | 11.96 | **12.84** |
| S8 | 256M | **15.79** | 15.56 | — |
| S8 | 4M | 6.37 | **6.79** | — |

### ReduceScatter — busbw GB/s

| placement | size | b=1 | b=2 | b=4 |
|---|---:|---:|---:|---:|
| S2 | 256M | 22.69 | **24.64** | 24.98 |
| S2 | 4M | 9.59 | 11.67 | **13.17** |
| S4 | 256M | 23.64 | **24.64** | 25.19 |
| S4 | 4M | 7.51 | 9.14 | **10.53** |
| S8 | 256M | **15.83** | 15.28 | — |
| S8 | 4M | 5.05 | **5.59** | — |

`b*` is bold. Reading: at 256M the PCIe ceiling is reached at `b=1`
(AllReduce) or `b=2` (ReduceScatter, within 1-2% of `b=4`), so adding
blocks inside the budget buys nothing at large messages. At 4M the
small-message regime measurably needs the budget maximum (AllReduce
S2/S4 +18-21% at `b=4` vs `b=1`; ReduceScatter +29-40%), which is the
measured justification for the per-size `b` used in the headline tables.

## Factor analysis

- Same-node AllReduce at 16M+ is 9-19% above native on S2/S4, near
  parity/above on S8 (PCIe-bound); 1M-4M is host-dispatch-floor-bound
  (native 1.5-3× ahead, 4M S4 13.0 vs 21.3).
- Cross-node 16M+ is 65-84% of native; X16 at 16M uses `b=2` (9.7 vs
  native 15.0 GB/s). The shortfall tracks the software RDMA fused proxy.
- AllGather is data-only and 0-SM: ring hops ride CE/IPC and the own
  shard is published by the user-stream copy engine, matching the
  AllReduce same-node story without worker SMs.
- AllToAll CE is 0-SM by construction; the 16-rank cross-node fan-out
  case (2.0-2.1 vs 4.4-4.7 GB/s) remains the open gap.
- Blocks are otherwise neutral at 256M (b1 vs b8-64 within ~1-3%),
  because PCIe, not worker parallelism, is the bottleneck.

## G1: concurrent collectives across CUDA streams (2026-09-05)

Validation for the FSDP backward-prefetch shape: AllGather of the next
layer's parameters and ReduceScatter of the current layer's gradients
issued on two CUDA streams. Harness `bench/stream_concurrent.cu`;
runners `bench/run_g1.sh` / `bench/retest_g1.sh`. Medians of 3, all
cells 0 wrong. Raw logs: `/tmp/g1_final`, `/tmp/g1_retest2`,
`/tmp/g1_s2k`, `/tmp/g1_x8`, `/tmp/g1_s4`, `/tmp/g1_confirm_free`
(node5).

Method: `W` = per-rank full layer tensor (AG input/RS output = `W/n`);
per iteration the harness launches the scenario's ops back-to-back
(AG before RS, each inside its own `ncclGroupStart/End`, i.e. the
launch pattern of ProcessGroupNCCL). `--sync-every K` syncs the device
every K batches: K=1 is end-to-end wall per batch (host cannot run
ahead), K=30 is fully pipelined (host dispatch overlaps device work).
Busbw uses nccl-tests factors (AG/RS `(n-1)/n*W`, AR `2(n-1)/n*W`).
Shim blocks per the SM-budget rule (RS `b`: S2=4, S8=2 at 1M / 1 at
256M, X16=2); AG stays 0-SM CE. Cross-node adds
`UK_CCL_RDMA_FUSED_MODE=proxy`.

Wall per batch (µs), shim / native NCCL 2.31.2:

| placement | W | scenario / comm | K1 shim/nat | K30 shim/nat |
|---|---|---|---|---|
| S2 | 1M | fsdp2 shared | 380 / 113 | 97 / 99 |
| S2 | 1M | fsdp2 per-op | 389 / 84 | 117 / 72 |
| S2 | 1M | seqfsdp shared | 409 / 116 | 165 / 99 |
| S2 | 256M | fsdp2 shared | 10,400 / 13,328 | 10,542 / 13,355 |
| S2 | 256M | fsdp2 per-op | 10,748 / 12,412 | 10,696 / 12,364 |
| S2 | 256M | seqfsdp shared | 11,008 / 13,431 | 11,014 / 13,300 |
| S8 | 1M | fsdp2 shared | 704 / 226 | 356 / 217 |
| S8 | 1M | fsdp2 per-op | 25,259 / 222 | 4,420 / 150 |
| S8 | 1M | seqfsdp shared | 927 / 226 | 743 / 219 |
| S8 | 256M | fsdp2 shared | 29,535 / 34,102 | 29,221 / 33,686 |
| S8 | 256M | fsdp2 per-op | 53,597 / 32,250 | 41,411 / 32,416 |
| S8 | 256M | seqfsdp shared | 30,565 / 33,675 | 29,799 / 33,765 |
| X16 | 1M | fsdp2 shared | 1,126 / 480 | 684 / 354 |
| X16 | 1M | fsdp2 per-op | 70,959 / 251 | 72,532 / 234 |
| X16 | 1M | seqfsdp shared | 1,374 / 366 | 1,500 / 358 |
| X16 | 256M | fsdp2 shared | 46,021 / 36,601 | 49,563 / 36,949 |
| X16 | 256M | fsdp2 per-op | 106,115 / 35,971 | 94,086 / 35,441 |
| X16 | 256M | seqfsdp shared | 49,982 / 36,686 | 49,065 / 36,772 |

Clean-GPU follow-up (all GPUs idle; X8 = 4+4 across nodes, S4 = 4 GPUs
on node5, no occupied card in the set), medians of 3:

| placement | W | scenario / comm | K1 shim/nat | K30 shim/nat |
|---|---|---|---|---|
| X8 | 1M | fsdp2 shared | 685 / 254 | 316 / 237 |
| X8 | 1M | fsdp2 per-op | 720 / 202 | 470 / 165 |
| X8 | 1M | seqfsdp shared | 898 / 251 | 686 / 243 |
| X8 | 256M | fsdp2 shared | 42,611 / 33,646 | 42,833 / 33,616 |
| X8 | 256M | fsdp2 per-op | 43,082 / 32,277 | 42,422 / 31,972 |
| X8 | 256M | seqfsdp shared | 43,121 / 33,786 | 43,088 / 33,558 |
| S4 | 1M | fsdp2 per-op | 503 / 119 | 229 / 98 |
| S4 | 1M | fsdp2 shared | — | 170 / 135 |
| S4 | 256M | fsdp2 per-op | 18,196 / 18,701 | 16,020 / 18,710 |
| S4 | 256M | fsdp2 shared | — | 15,760 / 18,907 |

Readings:

- **Native, same comm (`shared`), does not overlap.** fsdp2-shared wall
  equals seqfsdp wall at every placement/size (ratio 0.98-1.01), i.e.
  NCCL 2.31.2 serializes two group-launched collectives on one comm.
- **Native, separate comms per op (`per-op`), overlaps at 1M.**
  K30 wall drops 27-35% vs seqfsdp (S2 72 vs 99 µs, S8 150 vs 219 µs,
  X16 234 vs 358 µs); at 256M the gain is 4-7%. This is the FSDP2 /
  comm-split reference the shim would need to chase for small layers.
- **Shim (`shared`) already overlaps its own concurrent ops.** K30
  fsdp2-shared is 1.7-2.2× faster than its own seqfsdp at 1M (S2
  97/165, S8 356/743, X16 684/1500 µs). The single shared executor
  feeds both streams' runs into one worker FIFO and CE path; host
  dispatch of the second op hides behind the first op's device work.
- **Shim host floor is a run-ahead story.** SyncK sweep at S2/1M/fsdp2
  shared (medians of 3): shim 378 → 340 (K2) → 221 (K4) → 92 µs
  (K30); native stays 101-117 µs. A real prefetch depth of 1-2 layers
  (K=2-4) still leaves shim 1.9-2.9× behind native at 1M; only deep
  pipelining reaches parity.
- **Shim vs native under concurrency.** Same-node large messages shim
  wins (S2 256M 0.78-0.87×, S8 256M 0.87×); X16 loses 1.26-1.37×
  (proxy/network critical path). Small messages at K1 lose 2.3-4.1×;
  at K30, S2 reaches parity (0.98×) while S8/X16 still lose 1.6-1.9×.
- **Shim multi-comm (`per-op`) hits a real 8-local-GPU bug.** At S8 and
  X16 (8 GPUs per node participating), 1M per-op shim shows a 4-70 ms
  per-batch floor (AG worst; fixed ~17 ms at S8 / ~28 ms at X16) and
  ~2-3× penalty at 256M, plus intermittent all-bad ReduceScatter output
  and a teardown abort (`cudart` unloading race, rc=134). Reproduced
  with every GPU idle, so it is not co-tenant noise. The same runs are
  healthy at S2/S4 (1-4 local GPUs) and at X8 (4+4: 720/470 µs at 1M,
  ~43 ms at 256M ≈ shared). The trigger is 2 comms × 8 local IPC peers
  in one process — a Phase B debug item, not a fundamental FSDP2-shape
  blocker (see `concurrent_collectives_plan.md`).
- **Data quality note.** vLLM co-tenant driver resets during the first
  matrix aborted/contaminated three cells (failures all logged
  `driver shutting down`); every affected cell was re-measured on a
  clean window and shared/seq numbers reproduced. One X16 seqfsdp 256M
  shim run livelocked (16 procs spinning at ~100% CPU, GPU idle)
  waiting on a device flag after a co-tenant reset; the cell completed
  cleanly on rerun (50.0 / 49.1 ms K1/K30).

See `concurrent_collectives_plan.md` §G1 for the gate verdict and
implementation routing.
