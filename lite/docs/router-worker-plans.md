# Evidence-gated CPU worker plans

The optional `RouterPlanDispatcher` chooses among explicit
`RouterConfiguration` objects. `routerWorkerConfigurations()` generates
NUMA-local CPU sets from the participating devices and permitted CPU budget;
unknown placement produces no candidate. Exact CPU IDs, worker counts and
other options enter the existing dispatcher identity.

Selection requires the same exact workload, hardware/software/placement
identity and caller-fallback identity as `BackendPolicy`. Among admitted
configurations it chooses the smallest measured router median. Forced mode is
not a shortcut through this selector: use individual forced dispatchers to
collect correctness and calibration evidence first.

Worker-scoped host-copy profiles can supply an advisory partial-reduction
memory-traffic estimate. It excludes arithmetic, cross-NUMA merging, GPU
transfers and synchronization. It cannot admit a backend or override measured
collective results. AllGather's CPU path publishes readiness rather than
copying the payload, so it deliberately has no memcpy-based CPU prior.

`prepare()` constructs only the selected configuration and caches the decision
for the exact call. Configuration changes occur at an idle boundary, with
capture rejected before tearing down the preceding implementation. Preparing a
fallback releases any preceding active router; an initial/unprepared fallback
never creates router buffers or workers. `enqueue()` uses prepared choices and
never evaluates the resource model or starts a router.

```bash
make router-plan-check router-worker-calibrate
./build/uccl_lite_router_plan_check
./build/uccl_lite_router_plan_cuda_check
./build/uccl_lite_router_worker_calibrate build/workers.profile \
  build/host-copy-payload-isolated.profile
```

The calibration executable is a local four-L4 BF16 AllReduce experiment at
1 MiB, out-of-place, comparing 1/2/4 workers per NUMA node with the prepared
peer executor. It uses three ABBA rounds, two warmups and three checked calls
per block. Timing is maximum-GPU event time per call; validation is outside the
timed interval. Router objects/workers are absent during both baseline blocks.
All raw calls are retained in the companion CSV, and every configuration,
including regressions, is retained in the profile. The profile is tied to the
exact executable/dispatcher/fallback identities; it is not portable evidence
for a differently linked application.

The current local runs rejected all three router configurations and retained
the peer fallback. This is a useful negative result, not a reason to bypass the
admission gate. Synthetic-timing tests separately exercise admitted execution,
configuration switching, capture rejection/recovery and fallback teardown.
