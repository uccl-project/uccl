# Four-L4 FP32 AllGather with per-source GPU readiness (2026-11-11)

This is the current single-node four-L4 AllGather result on
`mibura-sky-test-02`: FP32, out of place, complete output bytes per GPU.
The host-packet CPU router's GPU producer already published an `uploaded`
generation after writing each source tile to mapped host memory. The older
Copy path waited for all four sources before any consumer read. The new
`--packet-router-per-source` variant assigns consumer blocks to individual
sources: each group waits **only for its source's system-release flag** and
copies its source tile as soon as it is ready. A GPU-local per-source flag
lets one block poll the host flag and release its other blocks. All groups
still join before the consumer publishes reader credits, so a producer cannot
reuse a two-slot packet while any GPU reads it. Copy output slices have one
source owner each; the old reduction paths are unchanged. GPU progress
creates **no CPU routing worker** for this AllGather path; the configured
`workersPerNode=8` is retained as a geometry option but is not an active
CPU executor pool in this mode.

The FP32 packet router copies raw bits as pairs of 16-bit storage units; it
does not convert FP32 to BF16 or do BF16 arithmetic. Every output element is
checked as FP32 on every warmup and timed call, including scalar and large
tail cases. [Hardware/software versions and source/binary hashes](hardware_software.txt)
identify the exact build. The optional target is `make peer-ag-rs-packet`;
the ordinary peer AG/RS target does not link the router.

## Selected UCCL Lite path and paired NCCL control

The same harness alternates UCCL candidates and NCCL 2.28.7 default on the
same four GPUs. Each backend gets 20 warmups and three rotated rounds of ten
timed calls. The timer is the slowest-GPU per-call CUDA event; input fill
and complete output checks are outside timing. The [full raw log](full.log),
[2,070 timed samples](samples.csv), [all candidates](candidates.csv), and
[`selected.csv`](selected.csv) retain every tested size and negative result.
The UCCL path for each size is the fastest correct UCCL median, chosen
without looking at NCCL. NCCL is an external control, never a fallback.

| Complete output/GPU | Selected UCCL Lite path |
| ---: | --- |
| 1 KiB–2 MiB | Existing single-CTA `push-serial` |
| 4–8 MiB | Existing direct GPU `peer` path |
| 16 MiB–1 GiB | New per-source host-packet router |

| Complete output/GPU | Selected UCCL path | UCCL (µs) | Paired NCCL (µs) |
| ---: | --- | ---: | ---: |
| 1 KiB | push-serial | 14.46 | 27.82 |
| 64 KiB | push-serial | 17.83 | 33.76 |
| 1 MiB | push-serial | 72.62 | 100.64 |
| 2 MiB | push-serial | 130.17 | 144.47 |
| 4 MiB | peer | 241.45 | 234.59 |
| 8 MiB | peer | 408.20 | 414.67 |
| 16 MiB | packet-router | **692.76** | 775.95 |
| 32 MiB | packet-router | **1,280.46** | 1,524.31 |
| 64 MiB | packet-router | **2,408.13** | 3,018.81 |
| 1 GiB | packet-router | **36,354.60** | 49,246.41 |

At 16 MiB, the paired old peer path takes **739.03 µs**. At 1 GiB it
takes **45,560.06 µs**, versus **36,354.60 µs** for the per-source packet
router. The 4/8-MiB router candidates still lose to peer (253.35/424.84 µs
versus 241.45/408.20 µs), so they remain in `candidates.csv` and the
selected UCCL cutoff moves from 32 to **16 MiB**. AllReduce and
ReduceScatter main README figures retain their previous implementations.

## Direct-readiness ablation and correctness

[`progress_ablation.csv`](progress_ablation.csv) compares the new per-source
wait with an otherwise matching GPU-progress variant that waits for **all
four** upload flags before copying. The variants ran in separate paired
windows, each with its own NCCL and peer controls. Router medians are:

| Complete output/GPU | Wait all sources | Wait only own source |
| ---: | ---: | ---: |
| 4 MiB | 269.09 µs | **251.94 µs** |
| 8 MiB | 445.49 µs | **425.57 µs** |
| 16 MiB | 723.72 µs | **692.93 µs** |
| 1 GiB | 36.84 ms | **36.30 ms** |

Correctness logs include `tail_1008.log`, `tail_large.log`, the four-GPU
mixed-operation checks with delayed producer ranks 1 and 3, large and tail
CUDA-graph replays, a three-rank delayed-producer check, backend policy and
dispatcher checks, plus unchanged
CPU-progress and default peer smoke tests. `--packet-router-per-source`
requires GPU progress; unsupported combinations are rejected. The full
sweep and both tail runs checked complete FP32 output after every invocation.

Run `python3 summarize.py` here to validate the 2,070 timed calls, all
ablation logs and the 21 selected rows; it regenerates `samples.csv`,
`candidates.csv`, `selected.csv`, and `progress_ablation.csv`. Reproduce the
full paired sweep from `experimental/lite`:

```bash
make -j4 cpu-router peer-ag-rs-packet router-checks
./build/uccl_lite_peer_ag_rs_packet --packet-router-ag \
  --packet-router-per-source -b 1024 -e 1073741824 \
  -w 20 -n 10 -r 3
python3 scripts/plot_optimized_collectives.py
```
