# Eight-L4 GPU and SHM overview figures

> This directory retains only inputs and metadata needed by current figures or checks. [Complete original experiment](https://github.com/zhongjiechen/uccl-zhongjie-dev/tree/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/overview_8l4_20261114) includes raw runs, removed scripts and source snapshots. Historical commands below require that archive.


The Testbed 1/2 figures show **UCCL Lite**, **NCCL CE (tuned)**,
**NCCL SM (tuned)** and **NCCL (default)**, using the RTX PRO 6000 ×2 figure
layout. SM (tuned) is the tuned default NCCL implementation; CE (tuned) enables
its copy-engine-assisted transports. The two tuned curves come from
[independent calibration and validation](../nccl_tuned_testbeds12_20261115/README.md).
The original UCCL and default-NCCL curves retain their recorded source,
configuration and measurements below.

All six ordinary-call figures use **complete 1 KiB–1 GiB measurements**.
AR/AG retain [the full refresh](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/full_refresh_8l4_20261114/README.md);
RS uses the subsequent [ring recovery confirmation](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/rs_recovery_8l4_20261114/README.md)
at every size, including freshly paired NCCL controls. There are no mixed-run
tails within a curve or reused GPU/SHM UCCL curves. BF16; AllReduce in place, AllGather and
ReduceScatter out of place. Sizes denote complete AR/RS input or AG output per
GPU. Each figure has 1-KiB–1-MiB latency and 1-MiB–1-GiB bus-throughput panels;
both panels include 1 MiB.

## Measurement and selection

Every point comes from two fresh opposite-order sweeps: five warmups, three
rotated repetitions of ten checked timed calls per backend and size. The second
sweep reverses testbed, operation, size and initial backend order. All producer
streams finish before timing. The timer is the slowest-GPU CUDA-event interval,
including host submission gaps. Every output element and guard is checked on
every warmup and timed call. The base router is also measured and retained in
the original AR/AG dataset, but does not replace a losing current candidate.
RS pairs the previous fixed selection, the revised selection, and NCCL.

Both testbeds use the same fixed UCCL schedule:

| Collective | Complete bytes/GPU | UCCL implementation |
| --- | --- | --- |
| AR | 1 KiB–4 MiB | Tuned router: GPU reduction/progress through 8 KiB; compact CPU reduction thereafter |
| AR | 8 MiB–1 GiB | CPU NUMA reduction, regular 23-worker pool and 64-block cap, batch publication and parallel submission |
| AG | 1–128 KiB | Tuned packet router with GPU progress |
| AG | 256 KiB–1 MiB | Mini-host cooperative kernel, 16 blocks |
| AG | 2 MiB–1 GiB | Full per-source GPU-progress packet router, 64-block cap |
| RS | 1–128 KiB | GPU reduction/progress |
| RS | 256–512 KiB | CPU NUMA reduction, four active workers |
| RS | 1–8 MiB | Opt-in mapped-host BF16 ring, 16 CTAs, GPU reduction/progress |
| RS | 16 MiB–1 GiB | Existing CPU NUMA reduction |

This schedule was fixed before the full sweeps; it is not a per-size minimum or
an automatic-routing policy. The report retains regressions against the base
router and NCCL. The six original default-NCCL curves were paired with their UCCL samples
in the same experiment. The new tuned references have their own validation
runs, documented in the linked tuning report.

For the original UCCL/default measurements, GPU mode leaves NCCL P2P policy
at its default. SHM mode explicitly disables
CUDA peer access and NCCL P2P. The recorded initialization probes show 16 SHM
channel edges in **both** modes on this host, so these curves must not be
interpreted as measured P2P-versus-SHM transport throughput. UCCL uses mapped
host memory in both modes. See the raw transport evidence in the full refresh.

## Resident figure

`collectives-8l4-resident-small.svg` remains a separate earlier experiment. Its
1–64-KiB host-blocking resident measurements and NCCL graph controls come from
[the compact transfer study](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/transfer_4to8_20261114/README.md). No resident
number is inserted into the freshly measured ordinary-call overviews, and no
resident result is extrapolated to 1 GiB.

## Reproduction

[summary.csv](summary.csv) contains exactly **126 points**, including
implementation labels and per-collective source paths. AR/AG measurements
remain tied to their earlier source; they are not revalidation of this revision.
To reconstruct the historical base dataset, first run
`python3 scripts/summarize_full_8l4.py` from the Lite root; it rejects missing
runs, samples, or size points. Then run:

```bash
python3 scripts/summarize_rs_recovery.py
python3 benchmarks/data/overview_8l4_20261114/build.py
python3 benchmarks/data/overview_8l4_20261114/plot.py --preview
```

`build.py` rejects historical or incomplete inputs. For the current four-series figures, keep the new tuning dataset’s validated
`summary.csv` alongside this dataset and run `plot.py --preview`.

`plot.py` writes six SVGs
under `docs/assets/` and PNG/PDF previews under `build/figures/`. Throughput is
decimal GB/s, `bytes / latency_us / 1000`, scaled by 14/8 for AR and 7/8 for
AG/RS. Source and figure hashes are recorded in `provenance.json`, and this
dataset is covered by `checksums.json`. Earlier figures and measurements remain
in Git history and their original datasets.
