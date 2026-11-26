# Automatic direct model calibration

When resource probing is enabled, the prepared distributed selector calibrates
the first constructed direct executor on private buffers. The resulting model
is reused for that runtime's exact requests. Calibration is outside timed
candidate blocks and never runs from enqueue. The existing end-to-end
correctness and paired-performance rules still decide admission.

## Measurement boundaries

Calibration measures the actual implementation primitives:

| Primitive | Timed boundary | Logical service bytes |
| --- | --- | --- |
| Scalar-aligned input staging | Copy submission through caller-stream completion | Two times payload |
| Self-slot copy | Copy submission through caller-stream completion | Two times payload |
| Gathered output | All rank-slot copies through stream completion | `2 * ranks * payload` |
| Source-ordered reduction | Existing reduction kernel through stream completion | `(ranks + 1) * payload` |
| Each directed IPC/IB flow | Host write submission through sender flush | Payload |
| Entry | The shared argument/operation/stream agreement path on an idle stream | Control time |
| Publication/retirement | A complete status agreement with ready participants | Control time |
| Required owner visibility | The actual GPU RDMA visibility call | Control time |

The scalar size and prepared chunk size are sampled. Every point retains nine
timed observations from three repetitions, each with two warmups and three
timed calls. All private source, receive-slot, output and guard bytes are
verified after every operation. Directed flows are isolated; local GPU
operations run on the participating GPUs. Receiver visibility is outside a
directed-flow timer and measured separately. Unsupported/failed operations
are errors, not fabricated rates.

## Service tables, not forced bandwidth fits

Initial four-GPU measurements showed nonpositive linear slopes for some copy
services: small host-completed operations can be slower than larger ones.
Those observations are retained in `build/direct-calibration-verified/`.
Forcing a positive bandwidth fit would misrepresent them.

The model instead interpolates **positive operation durations** inside the
observed size interval, without requiring monotonicity or extrapolating beyond
coverage. An operation's service resource includes its measured dispatch and
synchronization cost. Additional copy/flush/reduction overhead is zero because
it is already included, rather than because it is assumed free. Gather output
is represented by one measured copy-batch service. Inconsistent traffic factors,
mixed timing scopes, missing coverage and identity/dtype mismatches make the
estimate unavailable.

GPU memory uses the maximum observed local operation service envelope for this
working set. This is **not a physical DRAM bandwidth or peak measurement**;
GPU cache behavior and computation are folded into the scoped operation data.
Operational and GPU-memory demands are constraints on the same task, not added
durations. They feed the existing shared-resource estimator.

Known directed PCIe/NUMA cuts and NIC rates are attached as capacity bounds,
using host-scoped identities. Unknown shared cuts are listed explicitly in
`unmodeled_resources`; isolated end-to-end flows do not identify their separate
capacities. The global-drain and fluid-sharing assumptions remain advisory.

## Runtime reporting and limits

Direct candidate reports now include `resource_model_identity`,
`resource_model_observations`, `resource_model_reason`, unmodeled resources and
the existing estimate/serial/dependency/resource-load fields. Reconstructed
executors must match the calibration's per-rank configuration identities.
`probeResources=false` skips this calibration. A cached caller decision does
not construct direct or produce a new model. A positive cached candidate can
still calibrate a fresh model when probing is enabled.

The model is held in memory, not persisted as another source of performance
admission. It does not change chunk geometry or replace current candidate
timings. Smaller/larger cache footprints, other alignments, scheduler load,
unmeasured shared cuts and control-tail overlap can produce prediction error.
The independent prediction-versus-execution observations are retained so those
limits remain visible.

`make direct-calibration-check` builds a client that saves raw primitive samples,
checks post-calibration collectives and compares model estimates with separately
timed whole calls. The full-buffer oracle and the runtime's admission thresholds
are unchanged.

## Recorded prediction checks

The final calibration checks in `build/direct-calibration-release/` use the
two recorded Xeon Silver 4410Y hosts, each with four NVIDIA L4 GPUs, CUDA 12.9,
and MSCCL++ IB/IPC transports. The network runs use one selected HCA per host.
Each row below covers nine exact operation/count points: AR/AG/RS at 257
elements, one 64-KiB chunk, and one scalar beyond a chunk. FP32 and BF16 use
their respective scalar sizes. These are fixed implementations and chunk sizes;
there is no best-variant-per-size curve.

The comparison baseline is an independently timed **whole direct call**, not
NCCL: two warmups followed by three maximum-rank host-completion samples,
with input production and the external barrier outside the timer. Error is
`100 * (prediction / median(observed) - 1)`. The nine-sample primitive
calibration protocol is described above.

| Layout | Dtype | Median absolute error | Signed error range |
| --- | --- | --- | --- |
| One host, four GPUs | FP32 | 0.96% | -3.16% to +3.78% |
| Two hosts, 1+1 GPUs | BF16 | 12.78% | -31.13% to +12.50% |
| Two hosts, 3+1 GPUs | FP32 | 14.56% | -8.80% to +41.59% |
| Two hosts, 4+4 GPUs | FP32 | 8.69% | -14.95% to +12.02% |

An additional isolated-view 2+2 BF16 run passes under Compute Sanitizer with
zero device errors. Its instrumented timings are stored separately and are
not included in this performance table. Only handled MSCCL++ capability-probe
API reports are suppressed.

Earlier table-model runs in `build/direct-calibration-network/` have materially
different errors, including +61.24% at one 1+1 point. Both sets of observations
remain available; this repeat does not establish an accuracy bound. The model
remains advisory, and these checks claim no gain over NCCL.
