# 8xL4 compact AllReduce router (2026-11-11)

At 128 and 256 KiB complete input bytes per GPU, the packet router uses
four CPU workers per NUMA node and at most 16 transport blocks instead of
23 workers and 64 blocks. Both sizes retain 128 KiB packet buffers. This
setting is used for normal 8xL4 and the logical SHM testbed; UCCL-Lite's
host-packet route is the same in both.

Two process runs, ascending and descending, each compare the router with
the project-owned local path and NCCL 2.28.9. Each backend gets 20 warmups
and three alternating repetitions of ten timed calls at each size. All
**480 timed calls** passed full-output checks on all eight GPUs.

| Control | Ascending | Descending |
| --- | --- | --- |
| Local | [samples](local_run1.csv), [log](local_run1.log) | [samples](local_run2.csv), [log](local_run2.log) |
| NCCL | [samples](nccl_run1.csv), [log](nccl_run1.log) | [samples](nccl_run2.csv), [log](nccl_run2.log) |

The four matching `*.profile` files preserve topology and router-geometry
identities. The [summary](summary.csv) takes a median of repetition medians
within each run, then a median of run values. UCCL-Lite values in the
single-node figure come from the NCCL-paired runs.

| Complete input per GPU | Router | Local | NCCL |
| ---: | ---: | ---: | ---: |
| 128 KiB | 202.5 µs | 242.8 µs | 216.6 µs |
| 256 KiB | 231.9 µs | 320.5 µs | 308.2 µs |

The SHM figure reuses these two UCCL-Lite points and compares them with
the [archived NCCL SHM controls](../shm_8l4_20261111/README.md). Those
curves were measured in separate runs. `scripts/summarize_ar_compact_8l4.py`
regenerates this summary from the four sample and log files.

Reproduce from `experimental/lite` with `--gpus 8 --op allreduce --in-place
--begin 131072 --end 262144 --warmup 20 --iterations 10 --repeats 3
--calibrate PATH --samples PATH`; use `--fallback local` or `--fallback nccl`
and add `--reverse` for the second run. Keep distinct paths for each run.
