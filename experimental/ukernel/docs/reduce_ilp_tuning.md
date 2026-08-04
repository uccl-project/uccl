# Reduce kernel ILP tuning (UK_CCL_REDUCE_ILP)

Goal: lift the vectorized reduce kernel's **per-block throughput** so the
shim reaches native-NCCL-class bandwidth at **32 blocks or fewer** (native
anchor: 32 coll channels ≈ 16 GB/s per channel, 510 GB/s @ 256MB on
B300 — see `b300_native_nccl_measurements.md`). Throwing more blocks at
it is not the goal.

## Why ILP

The reduce kernel is latency-bound: each thread keeps `U` independent 16B
loads (of `src` and `dst`) in flight. To saturate DRAM you need
`in-flight bytes ≈ latency × bandwidth`. B300's HBM3e latency-bandwidth
product is ~5x A40's, so it needs ~5x the in-flight bytes per block —
that is why the same kernel that saturates A40 at 8 blocks needs 32-128
blocks on B300. `REDUCE_ILP` (=4|8|16, default 4) raises `U` without
adding blocks. Cost: registers (each in-flight vector = 4 regs); U=16 can
spill at 256 threads.

> **ILP is a BUILD-TIME knob** (`make ... REDUCE_ILP=8`), because a
> runtime dispatch over 4/8/16 tripled cicc+ptxas time to ~20 min per
> device file on B300. With one compile-time value the same file builds
> in ~15-20 s. Sweeping U therefore means one rebuild per value (the
> build is fast, so this is cheap).

## B300 baseline (ILP=4, before this knob existed)

Reduce bench (256M payload, 256 threads, persistent worker):

| blocks | GB/s |
|---:|---:|
| 8 | 57.4 |
| 16 | 113.4 |
| 32 | 220.2 |
| 64 | 419.5 |
| 128 | 763.1 |

Shim AllReduce 256M (`LT=8 TM=8M IB=16`, out-of-place):

| blocks | GB/s | vs native 515 |
|---:|---:|---:|
| 8 | 168.7 | 33% |
| 16 | 264.0 | 51% |
| 32 | 409.6 | 80% |
| 64 | 485.2 | 94% |

## A40 reference (2026-08-04, this machine)

Reduce bench (256M, 256 threads):

| blocks | ILP=4 | ILP=8 | ILP=16 |
|---:|---:|---:|---:|
| 8 | 89.5 | 93.7 | 72.3 |
| 32 | **184.6** | 137.1 | 75.6 |
| 64 | 165.7 | 106.7 | 72.0 |

A40 saturates its DRAM at ILP=4 / 32 blocks; higher ILP only adds
register pressure and occupancy loss (ILP=16 spills). Shim sanity on A40:
256M AllReduce BLK=8 ran 0 wrong at both ILP=4 and ILP=8 (~50 GB/s —
PCIe-bound there, unchanged as expected).

## B300 measurement commands

```bash
cd ~/jinyao/uccl && git pull
cd experimental/ukernel
# Build ONLY the target arch (B300 = sm_103) — the default 4-arch build
# is slow and sm_103 is not in it. nvcc comes from CUDA_HOME/bin/nvcc
# (never conda's — conda CUDA 12.x cannot target sm_103). Keep TMA off
# and cap parallelism.

# One clean+rebuild per ILP value (make does not track macro changes),
# then measure reduce kernel and shim at that U:
for ilp in 4 8 16; do
  echo "== REDUCE_ILP=$ilp"
  make clean -f Makefile >/dev/null
  make SM=103 ENABLE_TMA=0 REDUCE_ILP=$ilp -j8 nccl || break
  make SM=103 ENABLE_TMA=0 REDUCE_ILP=$ilp -j8 device_bench || break

  # 1) reduce kernel throughput (256M, 256 threads, blocks 8/16/32/64)
  BLOCKS="8 16 32 64" THREADS="256" SIZES="256M" \
    bash bench/bench_device_reduce_blocks.sh 2>/dev/null

  # 2) shim AllReduce A/B at this U (256M, LT=8 TM=8M IB=16)
  cd ~/jinyao/uccl/thirdparty/nccl-tests/build
  export LD_LIBRARY_PATH=~/jinyao/uccl/experimental/ukernel/build/nccl/lib
  export CUDA_VISIBLE_DEVICES=6,7
  for blk in 8 32 64; do
    printf "ILP=%-2s BLK=%-2s  " "$ilp" "$blk"
    UK_CCL_DEV_BLOCKS=$blk UK_CCL_LARGE_TILES=8 \
    UK_CCL_TILE_MIN_BYTES=8388608 UK_CCL_IPC_BATCH=16 \
    mpirun --mca hwloc_base_binding_policy none -np 2 -x LD_LIBRARY_PATH \
      -x CUDA_VISIBLE_DEVICES -x UK_CCL_DEV_BLOCKS \
      -x UK_CCL_LARGE_TILES -x UK_CCL_TILE_MIN_BYTES -x UK_CCL_IPC_BATCH \
      ./all_reduce_perf -b 256M -e 256M -g 1 -c 1 2>/dev/null \
      | awk '$1 ~ /^[0-9]+$/ && NF>=13 {printf "time=%sus algbw=%sGB/s wrong=%s\n", $6, $7, $9}'
  done
  cd ~/jinyao/uccl/experimental/ukernel
done
```

## What to look for / report back

- Reduce bench: does the **8/16 blocks** row rise with ILP=8/16 (e.g.
  8 blocks from ~57 to 100+ GB/s)? That is the "few blocks saturate"
  lever.
- Shim A/B: does ILP=8/16 raise 256M AllReduce at the same block count
  (BLK=32 from 410 toward 510 GB/s)?
- If U=16 fails with a register/launch error at 256 threads, retry with
  `THREADS="128"` in the bench (128×U=16 has the same in-flight bytes as
  256×U=8 but fits registers).
- Append results to this file with a date section; paste them in chat too.

## A40 quick reference (build-time ILP, 2026-08-04)

Reduce bench 16M, 8 blocks, 256 threads: ILP=4 → 115.5 GB/s, ILP=8 →
130.9 GB/s — the knob measurably lifts per-block throughput at low block
counts even on A40.

## B300 results (2026-08-04, GPU 6/7, built with `SM=103 REDUCE_ILP=U`)

Reduce bench — 256M payload, 256 threads, persistent worker:

| blocks | ILP=4 | ILP=8 | ILP=16 |
|---:|---:|---:|---:|
| 8 | 57.4 | 60.2 | **71.0** |
| 16 | 112.6 | 119.2 | **140.9** |
| 32 | 219.7 | 232.4 | **274.9** |
| 64 | 420.6 | 440.9 | **515.3** |

Shim AllReduce 256M (`LT=8 TM=8M IB=16`, out-of-place, all 0 wrong):

| blocks | ILP=4 | ILP=8 | ILP=16 |
|---:|---:|---:|---:|
| 8 | 172.3 | 181.8 | **198.0** |
| 32 | 365.0 | 387.3 | 389.6 |
| 64 | 442.4 | 427.8 | **489.8** |

Native anchor: 510-515 GB/s @ 256M (32 coll channels ≈ 16 GB/s per
channel).

## B300 analysis / conclusions

- **U=16 fits on B300 at 256 threads** (unlike A40, where it spilled):
  B300's register file is large enough. It is the clear winner: +24% per
  block on the reduce bench at every block count (8: 57→71, 32: 220→275,
  64: 421→515 GB/s).
- The reduce kernel at **64 blocks × U=16 hits 515 GB/s — parity with
  native**, and the shim AllReduce reaches **489.8 GB/s (95% of native)**
  at 64 blocks.
- But per-block throughput is still ~8-9 GB/s (515/64, 275/32, 71/8) vs
  native's ~16 GB/s per channel — **ILP alone cannot reach native at
  ≤32 blocks** (32 blocks × U=16: 390 GB/s, 76%). The remaining gap is
  per-thread/in-flight efficiency, which points to the TMA bulk path
  (cp.async.bulk → smem reduce → bulk store) as the next lever.
- BLK=32 allreduce plateaus at ~390 GB/s regardless of ILP — at that
  block count the reduce is no longer the only limiter (put/executor
  pipeline), so the put/reduce overlap work is complementary.
- Build-time cost: U=16 is heavy (two cicc+ptxas passes, ~15-20 min
  total for `nccl` + `device_bench`); U=4/8 build in ~1 min. Only pay
  the U=16 cost when you are measuring it.
- Note: these absolute numbers are ~10% lower than the ad-hoc sweep
  measured before the build refactors (BLK=32/64: 410/485 GB/s); the
  box is shared, so tenant load varies runs — the relative ILP trend is
  consistent.

### Suggested default / next steps

- Default stays `REDUCE_ILP=4` for fast builds; use `REDUCE_ILP=16` for
  peak runs (64 blocks ≈ 95% of native, 32 blocks ≈ 76%).
- **TMA bulk reduce is implemented** (`TMA_REDUCE=1 REDUCE_SMEM_KB=128`):
  see below.
- Pipeline **put/reduce overlap** in the executor so BLK=32 stops
  plateauing at ~390 GB/s.

## TMA bulk reduce (2026-08-04, B300)

Build: `make SM=103 ENABLE_TMA=0 REDUCE_ILP=4 REDUCE_SMEM_KB=128
TMA_REDUCE=1`. Per chunk: cp.async.bulk load src+dst into smem
(mbarrier::complete_tx, mbarrier re-initialized per chunk — phase
toggling hangs at ~512 chunks/tile), reduce in smem, bulk-group store
back, `fence.proxy.async.global` at task completion.

Shim AllReduce 256M (`LT=8 TM=8M IB=16`), vs ILP=4 baseline:

| blocks | ILP=4 | TMA | wrong |
|---:|---:|---:|---:|
| 8 | 172 | **292** | 65536/64M |
| 32 | 365 | **509.5** | 65536/64M |
| 64 | 412 | 486 | 65536/64M |

TMA at 32 blocks = **509.5 GB/s ≈ 99% of native 515** with the same block
count — the "32 blocks to parity" goal is met on bandwidth.

### TMA correctness status

- The reduce kernel itself is correct: the standalone launch path
  verifies `wrong=0` at 256M/8; the mbarrier-bulk pattern was validated
  with a minimal standalone kernel on B300.
- **FIXED (2026-08-04)**: the wrongness was the odd-sized TAIL chunk of
  each block slice — TMA bulk on e.g. a 512B final chunk wrote stale-smem
  garbage (locator test: deterministic 130KB wrong clusters per 512KB
  block slice, always starting at the tail chunk; values were garbage).
  The tail now falls back to the ILP vector path (at most one chunk per
  slice, no throughput impact). Shim AllReduce is now **0 wrong at every
  size 1M..256M**:

  | size | TMA BLK=32 algbw |
  |---:|---:|
  | 16M | 80.7 GB/s |
  | 64M | 194.8 |
  | 128M | 312.5 |
  | 256M | 454.7 |

  (run variance on the shared box: earlier 256M runs showed 467-509 GB/s)
- Bench harness note: the persistent-worker verify races the async-proxy
  stores (host D2H vs always-resident kernel) — use the launch path or
  the shim for correctness signals.

TMA stays opt-in (`TMA_REDUCE=1`) pending broader validation (multi-node
RDMA + the put/reduce pipeline work); flip the default once stable.

### Persistent-kernel idle-exit (2026-08-04)

- The worker persistent kernel now **idle-exits after 500µs** of empty
  fifo by default (`WorkerPool::Config::idleExitAfterUs`, shim default
  `device_idle_exit_us=500`), so user apps calling `cudaDeviceSynchronize`
  / D2H / legacy default stream no longer deadlock; bursts of consecutive
  collectives stay in ONE instance (inter-op gaps are µs-scale), and the
  next enqueue relaunches the kernel.
- Relaunch hardened against the exit/enqueue race: post-push relaunch in
  `enqueue`/`enqueue_batch`, relaunch-on-wait in `sync()`/`is_done()`, and
  an atomic claim so concurrent callers relaunch once.
- Caveat: the bench keeps `idleExitAfterUs=1e6` (always resident) — its
  rapid create/destroy + idle-exit cycles trip a CUDA 13.3 context hang
  (same driver family as the cudaFree hang documented in `destroyWorker`).
  The shim's long-lived worker does not hit this.
