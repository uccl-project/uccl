# Four-L4 BF16 CPU-router figures

The three archived BF16 figures show **our CPU-router
implementation at every one of the 21 sizes**, from 1 KiB to 1 GiB. NCCL
2.28.7 is only the paired external control. No plotted UCCL point is an
automatic fallback, and slower UCCL points remain visible. The figures
do not use `auto_selection.csv` or any backend-selection decision.
The current main README instead shows the earlier FP32 AR/RS figures and
the [FP32 AllGather per-source packet sweep](../fp32_packet_ag_per_source_4l4_20261111/README.md).

[`figure_summary.csv`](figure_summary.csv) has 63 rows and records the exact
UCCL path, dtype, device count, placement, timing method, NCCL configuration,
three-run range, and source case for each size. All plotted points are
**BF16, four GPUs, complete bytes per GPU, and slowest-GPU CUDA-event latency**
on `mibura-sky-test-02` with the same NCCL 2.28.7 library. The source runs
were freshly rebuilt after adding `--router-only`; [hardware and software
identity](hardware_software.txt) records the source/binary hashes. Each paired
run alternates forced UCCL router and a separately invoked NCCL control over
three rounds of ten timed calls; every call checks the complete output. Input
preparation and checks are outside timing.

Each operation also has a **fresh standalone `--router-only` replay** over
all 21 sizes. That mode does not initialize NCCL or another fallback. Its
fallback callback throws if invoked; `NCCL_DEBUG=INFO` produced no NCCL
initialization records. The three router-only logs contain 1,890 checked
timed calls. Their timings validate execution but are not substituted for
the paired curve. `scripts/build_cpu_router_4l4_figures.py` validates both
the 3,780 paired calls and all router-only calls before writing the CSV.
[`router_only_rejections.log`](router_only_rejections.log) confirms that the
mode rejects an explicit NCCL fallback and a disabled router.

| Figure | Placement | UCCL implementation and paired source | Warmups |
| --- | --- | --- | ---: |
| AllReduce | In place | `paired_allreduce_in.*`, 23 workers/NUMA, 64-block cap; the 128-KiB path is still CPU router | 5 |
| AllGather | Out of place | `paired_allgather_out.*`, CPU progress, 8 workers/NUMA, 248-block upper bound, 4-MiB packet cap; [geometry record](../cpu_router_ag_4l4_geometry_20261111/README.md) | 20 |
| ReduceScatter | Out of place | `paired_reducescatter_out.*`, 23 workers/NUMA, 64-block cap; 128 KiB uses the UCCL GPU-progress router variant | 5 |

The AR/RS and AG rows are from separate, correctly paired benchmark windows
on the same host and executable/NCCL library. They are not one interleaved
three-operation run. The older **FP32** direct-peer four-L4 figures remain
in [the detailed archive](../../../docs/peer-ag-rs.md); they are removed
from the main README because their dtype and implementation differ.

Representative losses and gains, in µs unless marked ms:

| Operation and full bytes/GPU | UCCL router | NCCL 2.28.7 |
| --- | ---: | ---: |
| AllReduce, 1 MiB | 444.93 | 153.68 |
| AllGather, 4 MiB | 311.30 | 241.66 |
| AllGather, 1 GiB | **37.44 ms** | 49.11 ms |
| ReduceScatter, 1 MiB | 159.23 | 94.24 |

The router remains the UCCL result even where NCCL wins. To regenerate the
CSV and SVGs from `experimental/lite`:

```bash
python3 scripts/build_cpu_router_4l4_figures.py
python3 scripts/plot_cpu_router_4l4_figures.py
```

For a fresh UCCL-only AllGather run, with no NCCL communicator initialized:

```bash
./build/uccl_lite_collectives --gpus 4 --router-only --op allgather \
  --begin 1024 --end 1073741824 --warmup 20 --iterations 10 --repeats 3 \
  --workers-per-node 8 --transport-blocks 248 \
  --large-ag-packet-bytes 4194304 \
  --samples build/new-router-only-ag-samples.csv
```
