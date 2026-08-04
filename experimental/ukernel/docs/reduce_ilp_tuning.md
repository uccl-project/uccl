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
blocks on B300. `UK_CCL_REDUCE_ILP` (=4|8|16, default 4) raises `U`
without adding blocks. Cost: registers (each in-flight vector = 4 regs);
U=16 can spill at 256 threads.

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
# Build ONLY the target arch (B300 = sm_103); the default 4-arch build is
# very slow with the ILP dispatch and sm_103 is not in it anyway. Keep
# TMA off and cap parallelism to avoid ptxas thrash.
make SM=103 ENABLE_TMA=0 -j8 nccl
make SM=103 ENABLE_TMA=0 -j8 device_bench

# 1) reduce kernel throughput vs ILP (256M, 256 threads, blocks 8/16/32/64)
for ilp in 4 8 16; do
  echo "== UK_CCL_REDUCE_ILP=$ilp"
  UK_CCL_REDUCE_ILP=$ilp BLOCKS="8 16 32 64" THREADS="256" SIZES="256M" \
    bash bench/bench_device_reduce_blocks.sh 2>/dev/null
done

# 2) shim AllReduce A/B: ILP x DEV_BLOCKS (256M, LT=8 TM=8M IB=16)
cd ~/jinyao/uccl/thirdparty/nccl-tests/build
export LD_LIBRARY_PATH=~/jinyao/uccl/experimental/ukernel/build/nccl/lib
export CUDA_VISIBLE_DEVICES=6,7
for ilp in 4 8 16; do
  for blk in 8 32 64; do
    printf "ILP=%-2s BLK=%-2s  " "$ilp" "$blk"
    UK_CCL_REDUCE_ILP=$ilp UK_CCL_DEV_BLOCKS=$blk UK_CCL_LARGE_TILES=8 \
    UK_CCL_TILE_MIN_BYTES=8388608 UK_CCL_IPC_BATCH=16 \
    mpirun --mca hwloc_base_binding_policy none -np 2 -x LD_LIBRARY_PATH \
      -x CUDA_VISIBLE_DEVICES -x UK_CCL_REDUCE_ILP -x UK_CCL_DEV_BLOCKS \
      -x UK_CCL_LARGE_TILES -x UK_CCL_TILE_MIN_BYTES -x UK_CCL_IPC_BATCH \
      ./all_reduce_perf -b 256M -e 256M -g 1 -c 1 2>/dev/null \
      | awk '$1 ~ /^[0-9]+$/ && NF>=13 {printf "time=%sus algbw=%sGB/s wrong=%s\n", $6, $7, $9}'
  done
done
```

## What to look for / report back

- Reduce bench: does the **8/16 blocks** row rise with ILP=8/16 (e.g.
  8 blocks from ~57 to 100+ GB/s)? That is the "few blocks saturate"
  lever.
- Shim A/B: does ILP=8/16 raise 256M AllReduce at the same block count
  (BLK=32 from 410 toward 510 GB/s)?
- If ILP=16 fails with a register/launch error at 256 threads, retry with
  `THREADS="128"` (128×U=16 has the same in-flight bytes as 256×U=8 but
  fits registers).
- Append results to this file with a date section; paste them in chat too.
