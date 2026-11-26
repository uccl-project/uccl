# UCCL Lite reference preservation: active requirement

The acceptance baseline is the README UCCL Lite multi-node implementation,
not native NCCL. The target is zero latency regression at every 1-KiB–1-GiB
AG/AR/RS size and both NCCL layouts, while the replacement executes only UCCL
Lite. Native fallback remains forbidden. The earlier 5%-versus-NCCL result
is not proof of this requirement.

`comparison.csv` diagnoses the previous replacement against the recorded
README curves. It has 61/126 slower rows (AG 25, AR 22, RS 14), with worst
ratios 1.4722, 1.2977 and 1.0142 respectively. This is historical diagnostic
information, not paired admission: layouts, process counts and timing differ.
No mismatched measurement is silently treated as a comparable baseline.

Reference provenance:
- AG: the README's combined CE/SM implementation, original binary SHA
  3bb71d255b4edfeeddcd00a389fea88f9150f546327460c392e06c94d93cb658;
  four MPI processes/two GPUs each, full output bytes per GPU.
- AR: exact recorded source recovered from commit 39ce6ad9, with its exact
  shared host-gather header; GPU small path and GPU RS + host-packet gather.
- RS: exact recorded source recovered from commit 833e4ac6, using the tuned
  standalone size policy and 96-CTA/512-thread/8-MiB-target configuration.

`source_identity.json` records full source hashes. Frozen sources and Makefile
build recipes are under `build/uccl_readme_reference/`; current implementation
files were not substituted into those reference builds. Fresh reproduction
uses three complete sweeps with original warmup/iteration settings. AG's
optional NCCL control is disabled; the strict shim satisfies its optional
NCCL dependency so native NCCL is not loaded. AR/RS have no NCCL dependency.
This reproduces original configurations; matched adapter admission is still
required and must retain the fast UCCL paths.

The goal is active and unfulfilled. Experimental shared-host/CE AG, cooperative
medium AG and hybrid AR paths now exist and pass the stated correctness checks.
They are not admitted as zero-regression replacements. The latest complete
[single-library diagnostic](cooperative_rs8_candidate/README.md) still has
51/126 slower rows (AG 16, AR 22, RS 13); its cross-harness comparison is not
matched admission. Native-NCCL-relative success is not a substitute.

[Active implementation and validation work](../../../docs/uccl-reference-no-regression.md).

## Fresh reference and first component

All nine reference jobs completed with per-call checking. Three-round 1-GiB
medians: AG 41.922 ms, AR 96.445 ms, RS 53.693 ms. See `fresh_reference/`.
The independent-GPU-process shared packet storage passes 48 cases/rank over two
lifetimes on both four-GPU hosts; see `shared_packets/`. Network exchange,
stream integration and the no-regression gate are not established by that check.


The subsequent [hybrid producer correction](hybrid_graph_rs8/README.md)
passes mixed ABI and a full AR-only timing follow-up. It improves 8-MiB AR
but does not meet the UCCL target, and has no new full-matrix admission.


The newest complete [persistent candidate](ar512_persistent/uccl_persistent_consumers16/README.md)
passes all nine nccl-tests jobs with one binary, and improves 1-GiB AG/AR over
the fresh UCCL reference. Its diagnostic still has 22/126 slower rows (5/6/11
for AG/AR/RS), and matched admission remains unproven. The goal is still active.


The newest [fused combined candidate](ar512_persistent/uccl_fused_combined/README.md)
passes all nine full nccl-tests jobs with one binary. Its diagnostic has seven
slower rows (AG 5, AR 2, RS 0). This supersedes the older full-matrix counts as
progress evidence, but zero-regression and matched admission remain unfulfilled.


The latest [384-thread AR candidate](ar512_persistent/uccl_ar256_threads384/README.md)
has a complete fixed-binary matrix with six slower AG diagnostic rows and none
for AR or RS. Correctness/native-absence checks pass; matched zero-regression
admission and the remaining AG work are still unfulfilled.


The latest [warp-publisher candidate](ar512_persistent/uccl_warp_publisher/README.md)
has four slower diagnostic rows, all AG at 8/16 MiB. All 84 AR/RS rows are no
higher than reference in this complete fixed-binary matrix. Correctness checks
pass, but remaining AG work and matched zero-regression admission are unfinished.


The [GPU readiness-gate candidate](ar512_persistent/uccl_gpu_gates8m/README.md)
passes all 126 batch-timing diagnostic medians. Final admission remains unproven:
the aligned per-call/full-retirement comparison is underway and already exposes
small-message latency gaps. The original goal is still active, not complete.
