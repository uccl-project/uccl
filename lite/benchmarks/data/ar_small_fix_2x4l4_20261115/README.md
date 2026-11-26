# Restore small-message AllReduce latency, November 15

> This directory retains only inputs and metadata needed by current figures or checks. [Complete original experiment](https://github.com/zhongjiechen/uccl-zhongjie-dev/tree/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/ar_small_fix_2x4l4_20261115) includes raw runs, removed scripts and source snapshots. Historical commands below require that archive.


The explicit `--host-gather 1` AllReduce path now uses the existing single-CTA
GPU AllReduce through **256 KiB**, followed by GPU ReduceScatter + host-packet
AllGather from **512 KiB** upward. The measured 256-KiB cutoff avoids the
GPU/host phase handoff for small messages without losing the hybrid gain at
1 MiB. The input extent is checked per invocation; all reductions remain on
GPUs. The optional CPU-router core and automatic serving dispatch are unchanged.
Small-only runs do not construct the host-gather buffer/worker engine.

## Measurement contract

Two hosts, mibura-sky-test-02 and mibura-sky-test-01; four NVIDIA L4 GPUs and
two IB rails each. GPU pairs 0/1 and 2/3 are NUMA groups, mapped to mlx5_0 and
mlx5_1. CUDA 12.9, Open MPI 4.1.7rc1, MSCCL++ build-ib-pcieccl. Eight MPI
ranks, one GPU per rank; FP32 in-place sum. N is input/output bytes per GPU.
Time is slowest-rank host enqueue-to-completion, excluding fill and full-output
checking. Every warmup and timed call is checked. Three rounds use twenty
warmups plus ten timed calls at each power-of-two size. Points are medians of
round means; UCCL Lite error bars show the round-mean range. Bus bandwidth is
`7N/(4t)` in decimal GB/s. The selection rule is fixed, not a best-variant-per-size
measured envelope.

`fixed` is this source, measured 1 KiB–1 GiB. `previous` is the exact binary
used for the preceding all-hybrid figure; `native` uses that same binary with
`--ar-auto 1`. Both controls are freshly remeasured at 1 KiB–1 MiB. Their maximum allocated
size is 1 MiB; fixed uses 1 GiB. Allocation/registration footprint therefore
differs, so the controls are diagnostic rather than an isolated switch-cost
comparison. The preceding full-range all-hybrid record has the same 1-GiB
maximum and is retained for comparison as well. Round order
is native, previous, fixed. The previous binary contains unrelated opt-in RS
experiments, none selected here; hashes identify it exactly. `comparison.csv`
records all medians and ranges. `summary.csv` combines the fresh fixed path
with the **unchanged recorded NCCL** rows from the preceding figure. No NCCL
benchmark is run for this fix. NCCL error bars remain omitted; ranges remain
in the CSV. Historical results are preserved separately.

[Archived figure: Fixed AllReduce, small-message latency and full-range throughput](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/ar_small_fix_2x4l4_20261115/results.svg)

## Results

The full-range fixed medians are **39.568 µs at 1 KiB**, **73.921 µs at
256 KiB**, **114.158 µs at 512 KiB**, **178.918 µs at 1 MiB**, and
**96.516 ms / 19.47 GB/s at 1 GiB**. The preceding all-hybrid full-range
record was 71.345 µs at 1 KiB and 84.034 µs at 256 KiB. These are separate
three-round records with the same maximum extent and timing contract, not
interleaved confidence intervals. The historical November 11 native 1-KiB
point was 38.398 µs; its configuration-specific result remains separate.

All 630 fixed timed calls and their warmups pass full-output checks. Both
fresh diagnostic controls also pass all 330 timed calls each. No correctness
checker or numerical tolerance was relaxed.

## Validation and reproduction

Both `multinode_allreduce` and `multinode_cpu_router` build using the Makefile.
The comparison and additional checks run on both four-GPU hosts. Additional
checks cover three consecutive in-place calls without refill, small-only
engine setup, sizes immediately below/at/above the 256-KiB switch, 1,536-byte
and 6,291,968-byte tails, a 131,200-byte host piece, and the 24-MiB flat-RS
interval. Ordinary native AR/AG/RS receive a separate 1-KiB–64-MiB regression
sweep. This is eager-only; graph replay and Compute Sanitizer are not claimed.

```bash
make BUILD=build/ar_fix \
  MSCCLPP_IB_BUILD=/home/zhongjie/third_party/mscclpp/build-ib-pcieccl \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx \
  multinode_allreduce multinode_cpu_router -j2
# Stage identical binaries at the same path on the second host.
bash benchmarks/data/ar_small_fix_2x4l4_20261115/run.sh
python3 benchmarks/data/ar_small_fix_2x4l4_20261115/summarize.py
python3 benchmarks/data/ar_small_fix_2x4l4_20261115/plot.py
```

Run from lite. The driver records the exact previous-control location and
refuses to overwrite comparison logs. Preserve this record and choose a new
output directory for reruns. `raw/` retains samples, error checks and selected
path names; `build.log` and the local/remote hashes identify this build.
