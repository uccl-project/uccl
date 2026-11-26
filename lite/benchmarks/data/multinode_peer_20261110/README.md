# Two-node AllGather and ReduceScatter experiment

**Timing-scope correction:** these archived UCCL Lite runs used CUDA-event
timing, while the `nccl-tests` main time column used host timing. The
side-by-side ratios are not a matched timing comparison. The raw logs and
original summaries remain intact for provenance; use the checked
[refresh](../refresh_20261110/README.md) for a matched current comparison.

The accepted AG/RS comparisons are `paired_direct/` (1 KiB–64 MiB) and
`paired_direct_large/` (128 MiB–1 GiB). Their validated aggregates are
`summary.csv` and `summary_large.csv`. These rows use standard rank-ordered
float32 input and out-of-place output. The separate `paired_ar/` and
`summary_ar.csv` provide a current-source 32-MiB–1-GiB in-place AllReduce
comparison. The figure in
`../../../docs/assets/multinode-peer.svg` compares latency at 64 MiB from
`summary.csv` and 1 GiB from `summary_large.csv`, plus in-place AllReduce
from `summary_ar.csv`. The 512-KiB NCCL anomaly remains in the raw logs and
CSV but is outside that figure. There is no per-size best-variant selection.

Both runs used two nodes with four NVIDIA L4s each, PCIe Gen4 x16, NUMA pairs
`(0,1)` and `(2,3)`, and `mlx5_0,mlx5_1`. Host `mibura-sky-test-02` used
driver 595.58.03; host `mibura-sky-test-01` used driver 580.178.04. CUDA 12.9,
MSCCL++ source `52f659d`, and NCCL 2.28.7 were used. The NCCL baseline is
MPI-enabled nccl-tests 2.19.7 from source `717b683`, one GPU per MPI rank,
`-a 3` for slowest-rank CUDA-event time, default P2P/GDR, and both HCAs enabled.

The paired launcher alternated backend order over three repetitions. Each
repetition used five warmups and ten timed calls. UCCL Lite selected the same
AllGather ring throughout. The standard-input direct ReduceScatter ring used
one fixed rule: at least two pieces per shard, targeting at most 8 MiB per
piece (capped by the shard size). Its working partials have separate slots;
the standard input is never reordered or modified.
Each AG/RS result checked three changed input generations, including one after
the timed calls. NCCL reported zero wrong and out-of-bounds values. Neither
allocation nor validation is timed. The validator rejects missing ranks,
unmatched size sets, incorrect data, wrong NCCL version, and nonfinite timing.
The AR run checks one input generation per size before timing and uses the
in-place NCCL column; AG/RS use the out-of-place NCCL column.

The source and executable identities for the accepted runs were:

| Artifact | SHA-256 |
| --- | --- |
| `src/collectives/multinode_allreduce.cu` | `67ee225d315b5c9e9dde4bf36c7c3a7e911f91d4ab3f0c275a7efacdd2bcff95` |
| `uccl_lite_multinode_allreduce` | `a89b77ea6818678b18d389585b34bdd28189631c9800d36399f688e4ac329e43` |
| `all_gather_perf_mpi` | `4e55a364e90d70f930083a360b90acd38070ade3081e003b794efa87c617b430` |
| `reduce_scatter_perf_mpi` | `87929577993c979e9bc78ef812af8a8b9d05b0119a06f66fa46720147acd0dac` |
| `all_reduce_perf_mpi` | `268c7274668743fd36a8a4583382e09bb9624126f18b2ecd18a8e832f8035621` |
| `libmscclpp.so` | `596d650299d33b86fb18f68d9394ce01996ee45e0181f68c6e5e9069189bfe70` |
| `libnccl.so.2` | `ad951367f06580b8b4597ef97772a1eda6bf7c0162967bcfef7efdc127ac5449` |

The earlier root-level `allgather*_run*.txt` and `reducescatter*_run*.txt`
show flat-versus-ring exploration with host-clock timing. `paired/` also uses
host-clock timing. `paired_standard/`, `paired_standard_large/`, and
`summary*gpu_pack.csv` are fair standard-input GPU-pack experiments but are
superseded by the direct ring. `paired_gpu_event/`, `paired_large/`, and the
two `summary_*prepacked.csv` files are **excluded** from the accepted
comparison: that earlier RS prototype received an already rearranged input,
with layout conversion outside timing. They remain as an audit of the design
correction, not evidence that standard-layout ReduceScatter beats NCCL.

To regenerate the accepted summaries and figure:

```bash
python3 scripts/summarize_multinode_peer.py \
  benchmarks/data/multinode_peer_20261110/paired_direct \
  benchmarks/data/multinode_peer_20261110/summary.csv
python3 scripts/summarize_multinode_peer.py \
  benchmarks/data/multinode_peer_20261110/paired_direct_large \
  benchmarks/data/multinode_peer_20261110/summary_large.csv
python3 scripts/summarize_multinode_peer.py \
  benchmarks/data/multinode_peer_20261110/paired_ar \
  benchmarks/data/multinode_peer_20261110/summary_ar.csv --operations allreduce
python3 scripts/plot_multinode_peer.py
```
