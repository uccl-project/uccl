# Persistent distributed preparation

Distributed eager preparation can now reuse matched resource measurements and
exact-request decisions across runtime and process lifetimes. Configure
`NetworkPreparationOptions::planCache`, the unified
`PreparationOptions::networkPlanCache`, or `UCCL_LITE_NETWORK_PLAN_CACHE` before
construction. The C extension uses the environment setting. The default is
disabled; no cache file is created unless configured.

Rank zero reads and updates the snapshot and distributes validated contents
through the existing bootstrap. Other ranks need the same configured path
string, but do not access that path. No shared filesystem, MPI dependency in
the core, or filesystem operation at enqueue is required. The optional snapshot
contains the automatically measured network profile and all retained exact
request records. It is not a standalone `UCCL_LITE_NETWORK_PROFILE` file.

## Matching and revalidation

The context binds participating hosts/GPU placement, topology and software
identity, caller baseline implementation/configuration, capacity, transport
identity and UCCL/CUDA execution or instrumentation settings. The cache pathname
itself is excluded from execution identity. Every record additionally binds
candidate kind, operation, dtype, exact extent and every rank's alias/alignment.
Hierarchy also binds its execution permutation and NIC rails. Direct binds
per-rank prepared transport/topology identities. A direct winner is reconstructed
and its identity checked before checkpoint execution; mismatches trigger fresh
qualification. Live hierarchy preflight runs before its reuse. An
explicitly supplied resource profile remains authoritative.

| Prepared result | Work at a new runtime's first matching prepare |
| --- | --- |
| No record or changed context/map | Existing three-round ABBA qualification |
| Cached network winner | Fresh guarded A–B–B–A checkpoint, two warmups and three timed calls/block |
| Cached caller decision | One guarded caller block, two warmups and three timed calls; no network candidate initialization |
| Correct candidate loses its checkpoint comparison | Full three-round qualification, then replace the record |
| Candidate fails checkpoint correctness | Evict that record and retain the caller; do not retry and promote it during that prepare |
| Baseline fails correctness or cache is malformed | Collective prepare error before admitting application execution |

Warm network reuse requires both the original complete evidence and a fresh
>3% median improvement with nonoverlapping timing ranges. Current checkpoint
samples never masquerade as three fresh rounds. The main sample arrays retain
the original 18 candidate/18 baseline measurements when `fromCache=true`;
`validationCandidateSamples` and `validationBaselineSamples` retain the new
checkpoint separately. Direct has independent `fromCache`, `cacheStatus`,
`validationSamples` and `validationBaselineSamples` fields. The common report
marks the whole decision cached only if every eligible candidate was reused
and no fresh pool validation was needed. Each candidate also exposes its own
cache status. The C report adds `from_cache`, `cache_status`,
`validation_samples_us` and `validation_baseline_samples_us`.

Timing is maximum-rank host enqueue-through-backend completion, including
network send retirement. Setup, rendezvous, input generation and full-buffer
verification remain outside timing. Alternative resources are absent during
caller comparison blocks and restored afterward only for retained winning
handles. Subsequent in-memory hits revalidate identity without repeating the
checkpoint. Owned distributed graphs retain native NCCL and do not reuse eager
network admission.

Cached caller decisions deliberately avoid searching for newly emerging gains;
clear/disable the snapshot or use a new context to force a fresh search. This
keeps cached fallback free of candidate initialization. The cache is a prepared
decision reuse mechanism, not continuous adaptation to changing load.

## Persistence and failure boundaries

The CPU-only format is versioned, bounded to 64 MiB and 4096 records, and retains
raw samples and negative decisions. Updates lock the file, merge only the same
context, write and sync a private temporary file, and atomically replace the
snapshot. A full snapshot retires the oldest qualification when a new one is
added. Different contexts replace the prior context's records. Malformed files
are rejected without silently overwriting them. Write failures are reported
collectively; caller identity is revalidated before publishing new evidence.

`encodeNetworkProfile`/`decodeNetworkProfile` share the existing validated
resource-file format with the embedded profile. File reads require bounded
regular files and reject extent changes during reading. Existing v1/v2 resource
profiles, both MPI/bootstrap cohort scopes, and negative observations remain
compatible. No application pointers or communicator handles are persisted.

Hierarchy-only snapshots retain the V1 plan format and its FP32/two-host/four-GPU
contract. Snapshots containing direct evidence use V2, which stores candidate
kind, dtype, rank extent and per-rank configuration identities explicitly.
Direct supports FP32/BF16 and 2–32-rank scalar extents; old V1 files remain
readable. Candidate kind is part of the key, so updates and eviction cannot
replace another implementation's evidence for the same workload.

A snapshot records candidate-specific decisions, not a final combined-pool
winner. Both candidates compete using fresh checkpoint samples when cached;
combined-resource validation still runs when needed. Global/request fallback-only
preparation and local selection do not newly load this distributed cache. It does not add standard NCCL group or external
capture forwarding.
Previously prepared instances still enforce their immutable live context even
when a later request asks for fallback-only execution.

## Validation

`make network-cache-runtime-check planner-check nccl-prepared-api-check` builds
the checks. CPU tests cover exact requests/layouts, decisions inconsistent with
their samples, truncated/unknown formats, embedded resource roundtrips,
concurrent writers, context replacement, exact eviction and bounded retention.

`build/network-cache-final/` contains separate MPI process launches on two real
four-L4 hosts, with reversed GPU ownership and interleaved logical ranks. The
other host has no `state/` cache directory. Controlled scheduling-delay fixtures
exercise positive hits and performance drift; they are correctness tests, not
NCCL speedup measurements. They confirm:

- Cold qualification invokes the native baseline 30 times; positive reuse
  invokes it 10 times and retains 6/6 fresh checkpoint samples.
- Cached native reuse invokes the baseline 5 times, retains 0/3 checkpoint
  samples and initializes no network candidate.
- Performance drift triggers the checkpoint plus full qualification (40 native
  calls) and returns to native in the fixture.
- Changed configuration and execution maps miss. Embedded resource identity
  mismatch, malformed files and an intentionally incorrect baseline reject.
  Forced fallback ignores a malformed cache.
- Full-buffer independent scalar references, guards, in-memory reuse and native
  communicator ownership pass in every applicable mode.

The real shared C-interface run checks 36 requests over two lifetimes. Twelve
eligible second-lifetime requests reuse persisted native decisions; unsupported
extents remain native. Automatic resource probing runs once, not once per
lifetime. Four-rank local and eight-rank fallback regressions also pass.
Hardware is the documented dual-NUMA Xeon Silver 4410Y/L4 setup; baseline is
NCCL 2.28.7. Data remains local calibration/test evidence, separate from the
historical eight-GPU performance figures.

`make direct-cache-check` builds the direct runtime checker and runs the shared
CPU format checks. V2 tests cover typed scalar tails, variable rank counts,
coexisting candidates, per-kind eviction, malformed identities and decisions
inconsistent with their evidence. A negative direct hit validates only the
caller and does not assert a newly measured direct time or initialize direct.

A direct writer without a resource profile preserves an existing hierarchy
calibration in the same context. Absence means “no update,” not “erase”; a new
context still replaces the old context and its resources. The regression suite
checks this interleaving as well as candidate-specific update/eviction.

Direct checkpoints keep original 18/18 evidence separate from fresh 6/6 samples.
A changed prepared configuration triggers full qualification before publishing
new evidence. Cached caller decisions take one five-call checked baseline block
(three timed observations) and create no direct executor. Candidate competition
uses fresh checkpoint samples for both implementations. In the ranking fixture,
artificially tiny historical direct samples cannot override a faster current
hierarchy measurement.

Current mixed/direct evidence is in `build/direct-cache-release/`: 36 runtime
cases pass, including BF16/FP32 variable-host cold/warm/refresh/error cases,
current-sample candidate ranking, C API reporting, hierarchy V1 compatibility,
standard ABI pool checks and strict V4 regressions. Two-host/four-rank cold and
warm BF16 memchecks report zero device errors (known capability-probe API reports
alone are suppressed). Source/binary fingerprints are in
`build/direct-cache-artifacts.sha256`. Hardware is the recorded dual-NUMA Xeon
Silver 4410Y/L4 setup. The scheduling and synthetic-history fixtures test policy
behavior; they do not establish a performance gain over NCCL.
