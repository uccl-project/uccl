# AR 512-KiB boundary and persistent host AG experiment

The zero-regression target is unchanged and unfulfilled. All measurements here
use two hosts with four NVIDIA L4s each, FP32 SUM/copy, three unchanged nccl-tests
sweeps over 1 KiB–1 GiB and both layouts, 20 warmups and 30 timed iterations.
Each row is a median of three maximum-rank host enqueue/completion batch means.
These are fixed builds, with no best-per-size envelope. Comparison against the
fresh original UCCL reference remains diagnostic because harness conventions
differ; it does not establish matched admission.

## AR boundary

The direct-input AR experiment selects the existing medium algorithm at exactly
512 KiB, only if preparation allocated the large scratch layout. It retains
that algorithm's single-bank reuse and drain rules; it does not extend the
medium double-buffer policy. Smaller preparations retain the existing UCCL path.
The optional boundary ABI adds 512-KiB calls without removing previous cases:
144 calls/rank over two lifetimes pass, including offsets, per-rank layout
variations, streams, allocation recycling and operation transitions.

512-KiB AR out/in medians are 110.79/109.86 us versus fresh UCCL's 115.06 us.
The preceding fixed build measured about 145–148 us. Other full-curve results
remain in the CSV; 1-GiB AR still exceeds the UCCL target. No native NCCL is
loaded by candidate ranks and every size executes UCCL without fallback.

## First persistent AG: correct but rejected on performance

The cooperative grid splits four upload blocks from 28 consumer blocks. All
actors are resident. Producers wait for reader ACKs and outgoing retirement,
write mapped host rows, issue system fences and publish upload only after all
producer blocks complete. Consumers handle disjoint output elements, consume
local/remote rows in order and publish ACKs only after all consumer blocks
complete. Per-slot CUDA atomic counters reset before publication; existing
MSCCL++ receive credits and CPU-only RDMA progress remain unchanged. No CUDA
calls are made by the network worker, and the caller stream owns completion.

The component passes 54 calls/rank over two lifetimes, including delayed ranks,
misalignment, packet tails, mixed layouts, stream changes and allocation reuse.
The strict mixed ABI passes all 144 calls/rank. Three complete nccl-tests AG
sweeps pass numerical and route checks, but 1-GiB out/in are 70.804/72.654 ms
versus 41.922 ms reference; 8-MiB is 544.68/545.25 us versus 355.37 us.
The experiment is not admitted. It requires a per-block phase timeline before
further tuning; fewer CUDA launches alone did not improve its performance.


## Phase timeline and follow-ups

An instrumented build using the [FlashInfer profiler macros](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/profiler.cuh)
passes the 144-call ABI and a checked 1-GiB nccl-tests job (2 warmups, 3 timed
iterations). The saved buffers contain the final in-place invocation on all
eight GPUs. Header identity, raw command, decoder, per-block statistics and a
rank-0 plot are retained in timeline/. Binary traces remain under build/.
These instrumented timings are diagnostic and are not scored performance.

Each phase span uses the issuing thread's timestamps. Publication spans include
fences, block synchronization and completion counters; phase wall time also
includes start skew. Ranks 0–3 average roughly 74 us/packet in remote wait,
while ranks 4–7 average 62–65 us in local wait. The first eight consumer blocks
usually finish last; 64 vector tiles split over 28 consumers give those blocks
three tiles and the remainder two. This indicates imbalance and synchronization
cost, not a pure copy-bandwidth measurement.

Changing only the consumer fence to device scope preserves host payload system
fences and system-release ACKs. It passes 54 component and 144 ABI calls/rank,
but 1-GiB AG is still 68.960/69.206 ms (out/in), so it is unadmitted.
A separate resident control block publishes upload/reader flags after observing
completion counters, keeping the same four producer/28 consumer blocks. It also
passes 54 component and 144 ABI calls/rank. Three complete sweeps give
57.372/57.500 ms at 1 GiB, still slower than both the CE candidate and UCCL.
The next screen uses 16 consumers to divide the 64 tiles evenly. No failed
variant is enabled as the default replacement.

A component staging attempt ran out of remote disk space before MPI started.
Only completed-run duplicate binaries with exact local SHA-256 matches were
removed; the retry passed. All source, logs and local binaries were preserved.


## Balanced consumers: full fixed-library result

The [16-consumer candidate](uccl_persistent_consumers16/README.md) retains four
uploaders and one publisher, and divides all 64 packet tiles evenly. Its 54-call
component and 144-call ABI checks pass. The same library passes nine full
nccl-tests sweeps (three per operation). At 1 GiB, AG is 38.149/38.235 ms versus
41.922 ms reference; AR is 92.180/92.247 ms versus 96.445 ms. RS remains
54.162/53.874 ms versus 53.693 ms. These are diagnostics, not matched admission.
Across the complete matrix, 22/126 rows remain slower (AG 5, AR 6, RS 11).

Two additional fixed-policy screens are rejected. Extending direct-input AR
down to 256 KiB gives 80.39/80.15 us there versus 74.55 us reference, so retain
only the 512-KiB change. Reducing the persistent path's source packet to 64 KiB
for complete outputs through 32 MiB gives 8-MiB AG 731.74/745.16 us versus
355.37 us reference; 1-GiB timing is unchanged because its packet policy remains
256 KiB. Both experiments pass the 144-call ABI; the packet experiment also
passes 54 component calls/rank. Their full negative curves remain preserved.
