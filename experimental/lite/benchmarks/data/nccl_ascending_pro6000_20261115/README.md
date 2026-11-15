# NCCL memcpy tuning with ascending-size sweeps

This follow-up uses the requested **small-to-large** test order for all three
collectives. Every process creates its own communicator, starts at 1 KiB,
and visits every power of two through 1 GiB while retaining that communicator.
It never creates a fresh communicator at a selected large-size boundary.
The resulting purple reference curve is distinct from the green NCCL curve
validated with both ascending and descending size orders.

This study does not overturn the earlier largest-first stalls or establish
that they are fixed. It measures the explicitly different ascending execution
sequence. The original UCCL, default-NCCL and two-order tuned references remain
visible in the README figures.

## Results

All 30 calibration processes and all 12 independent final processes passed.
The final runs contain **25,200 checked timed invocations**, plus warmups.
At **1 GiB complete tensor per GPU**:

| Collective | Fresh default | Ascending memcpy reference | Speedup |
| --- | ---: | ---: | ---: |
| AllReduce | 32.42 GB/s | **50.71 GB/s** | 1.564× |
| AllGather | 28.57 GB/s | **49.17 GB/s** | 1.721× |
| ReduceScatter | 30.72 GB/s | **48.39 GB/s** | 1.575× |

The chosen bulk settings are:

```bash
NCCL_P2P_USE_CUDA_MEMCPY=1
NCCL_MIN_NCHANNELS=8
NCCL_MAX_NCHANNELS=8
NCCL_BUFFSIZE=16777216  # AllReduce: 16 MiB
# NCCL_BUFFSIZE=8388608 # AllGather / ReduceScatter: 8 MiB
```

These settings are fixed within each complete ascending run. The plotted
small-size default fallback is chosen from calibration, using separate full
default sweeps; no setting is changed on a live communicator. Do not interpret
the composite reference curve as one runtime-switching communicator.

All six complete selection traces passed coverage checks. Tuned bulk paths
log `P2P/CUMEM/CE`, while default paths use direct P2P. Large collectives still
use Ring/Simple GPU kernels: this is the P2P transport's memcpy path, not
the separate zero-CTA collective API. See [transport.json](transport.json).

At 1 GiB the new NCCL values are comparable to the previously recorded UCCL
peer reference. UCCL was not rerun in this measurement window, so small
cross-window differences are not a paired superiority claim.

## Method

- Host: `aws-pro6000`; two RTX PRO 6000 Blackwell Server Edition GPUs,
  PCIe Gen5 x16, same PCIe switch (`PIX`), Xeon Platinum 8559C, one NUMA node.
- Driver 595.91.07, CUDA runtime 13.0/compiler 13.0.88, NCCL 2.29.7+cuda13.2.
- BF16, out of place, one process with two GPUs. Complete AR input, AG output
  or RS input bytes per rank define size. AR bus bandwidth is `bytes/time`;
  AG/RS use `0.5*bytes/time`, in decimal GB/s.
- The unchanged NCCL-only reference from the preceding study is reused, with
  ordinary cudaMalloc buffers in every configuration here. No symmetric-window
  registration or `NCCL_CTA_POLICY=2` is enabled in this follow-up.
- Eager slowest-rank CUDA-event timing includes submission gaps. Allocation,
  communicator creation, input generation and full-output/input/guard checking
  are outside the interval. All buffers retain 1-GiB capacity.

The candidate family is default NCCL, `NCCL_P2P_USE_CUDA_MEMCPY=1` alone,
and memcpy transport with eight channels and 8/16/32-MiB collective buffers.
Each candidate receives two separate ascending calibration runs per operation,
with distinct seeds, three warmups and 3×3 timed calls per size. Candidate and
operation order is shuffled; **size order is always ascending**. All 30
calibration processes passed correctness.

Calibration freezes one bulk configuration per operation using geometric-mean
latency over 16 MiB–1 GiB. The plotted schedule uses that configuration only
at sizes of at least 16 MiB where both calibrations show a gain of at least
5% over default. Otherwise it uses the default control. This rule is fixed
before final measurement and does not select the best final result per size.

Final measurement independently runs both default and the selected bulk
configuration over the **entire** 1 KiB–1 GiB grid, twice per operation.
Each size receives 100 warmups and five rounds of twenty checked timed calls.
Configuration/operation order swaps between repeats, but neither repeat
reverses the sizes. The plotted point is the median of the two run medians;
shading is their observed range, not a confidence interval.

The plotted reference is a calibrated default/memcpy comparison assembled
from these full sweeps. It is not an online environment-variable switch or
a claim that a newly created large-only communicator will work. The complete
fixed-memcpy results, including small-message slowdowns, are retained separately
in [fixed_config_cases.csv](fixed_config_cases.csv).

## Artifacts

- [run.py](run.py): calibration, final measurement and ascending transport traces.
- [select.py](select.py): freezes configurations and per-size default fallback.
- [analyze.py](analyze.py): reconstructs medians and verifies ascending sample order.
- [check_traces.py](check_traces.py): verifies full size coverage and actual
  default/memcpy transport selection.
- [candidates.json](candidates.json), [schedule.json](schedule.json), and
  [selection.json](selection.json): settings, decisions and freeze timestamp/hash.
- [cases.csv](cases.csv): plotted reference and fresh default bandwidths.
- [fixed_config_cases.csv](fixed_config_cases.csv): full chosen memcpy curves.
- `*_runs.json` and `*_hashes.json`: commands, settings, timestamps, outcomes
  and executable/source/library identity. The shared node lock is held and
  GPU occupancy is checked before every process.

The executable and its source are unchanged from the
[NCCL-only preparation probe](../nccl_tuned_pro6000_20261115/README.md).
No new CUDA compilation or production-source change was required.
Remote outputs are under
`/home/xinze/dev/pro6000-integrated-b0ac7b4a-20261115-022114-retry1/source/experimental/lite/build/nccl_ascending_20261115/`.
Historical UCCL numbers are not rerun or substituted into these fresh NCCL results.

Raw calibration, final samples and transport traces are packaged in
[measurements.tar.gz](measurements.tar.gz). To verify offline, copy this
directory to a temporary location, extract the archive there, then run
`python3 analyze.py` and `python3 check_traces.py`. The directly reviewable
run manifests record exact commands and outcomes. [checksums.json](checksums.json)
records evidence hashes; `figures.json` identifies the updated four-curve SVGs.
