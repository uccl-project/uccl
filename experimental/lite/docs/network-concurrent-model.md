# Concurrent GPU/NIC flow measurements

The network probe now measures complete flow cohorts in addition to isolated
paths. A cohort records every directed GPU/NIC endpoint, payload bytes per flow,
and the duration observed by every sender on every timed call. It does not
assign the resulting aggregate rate to an individual NIC or CPU interconnect.

`NetworkProfile` format v2 retains the isolated observations and adds cohorts,
their control-software identities, and both timing matrices:

- Sender time covers CPU posting through sender CQ completion.
- Cohort time includes an MPI start barrier, posting/CQ completion and an MPI
  completion barrier. The reported sample is the maximum rank duration for that
  complete call. This window contains all transfers even if sender starts are
  staggered; summing bytes over the largest individual sender duration would not
  provide that guarantee.
- Receiver RDMA visibility, input generation and complete GPU input/output/guard
  verification remain outside timing. Every warmup and timed call is checked.
- Caller CPU affinity and NUMA memory policy are restored after MSCCL++ setup,
  before measurements. The original memory policy is included in live transport
  identity. The new isolated-path timing scope identifies this placement
  protocol explicitly. Legacy records can still be read/reported, but their old
  placement scope is not reused as current operational evidence.

The timed MPI implementation/configuration is recorded as producer provenance.
It does not require the consuming runtime to link MPI. These are measurements
of a controlled transfer cohort, including its control overhead, not forecasts
of GPU-driven collective latency.

## Probe placements

The probe generates the physical planner's placement, rotations through each
host's active NIC list, and uniform NIC choices per host. Duplicate complete
placements are removed. Candidate order rotates between repetitions. Setup and
teardown occur outside timing; all sender/cohort samples are retained, including
partial samples if later registration fails. Failed groups cannot be selected.

On the current two-host/four-L4-per-host topology this produces eight groups,
each with eight bidirectional flows. Per-flow sizes are 1 MiB and 64 MiB. Each
group has three repetitions, two warmups and five timed calls per size and
repetition. The output contains 240 timed cohort calls and 1,920 sender plus
1,920 cohort durations, alongside the 1,440 isolated timed calls.

```bash
make calibrate-network network-profile-report \
  MSCCLPP_IB_BUILD=/home/zhongjie/third_party/mscclpp/build-ib-pcieccl \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx

python3 -B benchmarks/probes/run_network.py \
  --hosts 10.10.55.2,10.10.55.1 --out build/network-cohorts-new \
  --mpirun /usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
  --iface ibp55s0f0 --hcas mlx5_1,mlx5_0 --peer-hcas mlx5_0,mlx5_1 \
  --ranks-per-host 4 --wait-seconds 600

build/uccl_lite_network_profile_report \
  build/network-cohorts-new/network.profile --groups > build/network-groups.csv
```

The GPU-idle gate and process monitor described in the network-calibration guide
still apply. They do not reserve the CPU or network fabric.

## Planning contract

The planner first constructs the existing isolated-path/shared-bound prior.
Concurrent data is usable only if the exact baseline flow set and the exact
per-flow byte extent were observed. It does not interpolate group measurements,
reuse a subset, infer a larger group, or relabel a different NIC assignment.
Both live host identities must match.

Measured alternatives must cover every declared peer edge in both directions,
use one consistent NIC per endpoint and remain in the current physical candidate
set. A replacement must beat the measured baseline median by more than 3%, with
its complete cohort-sample range below the baseline range. The lowest admitted
median wins. Without a measured baseline, the prior remains advisory; isolated
observations never become concurrent evidence.

The root broadcasts the selected plan and its concurrent-evidence status. The
profile consensus digest covers every sender/cohort value and the control
identities. Changing only concurrent data on one rank rejects preparation.
Enqueue continues to use the prepared transport without new measurement or
routing decisions.

Groups outside the currently modeled physical candidate set remain diagnostic
evidence. The model still omits local collective traffic, computation overlap
and GPU-driven protocol costs. Whole-backend automatic admission, broader
algorithms/layouts and native fallback are separate unfinished work.

## Recorded results and checks

`build/network-concurrent-calibration/` contains the current profile, manifest,
GPU process observations, raw verification log, `paths.csv` and `groups.csv`.
Hardware is two hosts with four NVIDIA L4s each, dual Xeon Silver 4410Y NUMA
domains and two advertised 400-Gb/s HCAs per host. Payload is raw uint32 data;
there is no NCCL baseline in this probe.

At 64 MiB per flow, the recorded median maximum-cohort durations were:

| Placement | Cohort duration | Aggregate useful payload rate |
| --- | ---: | ---: |
| Physical topology placement | 6.834 ms | 78.56 GB/s |
| Other topology rotations | 7.827–8.018 ms | 66.96–68.59 GB/s |
| Uniform NIC choice per host | 13.036–13.200 ms | 40.67–41.18 GB/s |

Ranges retain all configurations in each category; this is not a best-variant
curve. The runtime correctly retained the topology placement. Its cohort cost
is substantially larger than its isolated-path prior, demonstrating why the
single-flow measurements alone were insufficient. These numbers include the
timed control barriers and are not pure NIC capacity or a collective speedup.

CPU checks cover max-per-call aggregation, flow-order invariance, exact extent,
missing/different flows, stale endpoints, partial negative records, persistence,
legacy parsing and concurrent-choice reversal on synthetic evidence. Real ABI
runs are recorded under `build/network-groups-final-{matched,isolated,subset,
legacy}/`: 126 calls/rank, two lifetimes, two real four-L4 hosts and opposite HCA
enumeration. The matched case also rejects one rank's different concurrent
samples. Isolated-only and subset fixtures do not gain concurrent status; legacy
placement-scope records remain historical evidence.
