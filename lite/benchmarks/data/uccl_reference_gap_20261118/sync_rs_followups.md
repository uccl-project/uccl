# Synchronization and RS output follow-ups

The zero-regression goal remains active. All timings below use two four-L4
hosts, FP32 SUM/copy, three unchanged nccl-tests sweeps over 1 KiB–1 GiB and
both layouts, 20 warmups and 30 timed iterations. Each row is a median of
maximum-rank host enqueue/completion batch means. Policies are fixed builds,
not per-size minima. Comparisons against the original UCCL harness remain
diagnostic; matched admission has not been established.

Increasing persistent uploaders from four to eight preserves numerical checks
but slows 1-GiB AG to 39.459/38.990 ms versus the 38-ms four-uploader control.
It does not improve the medium gaps and is rejected. Skipping redundant waits
only when CUDA stream IDs match (so recycled handles remain distinct) also
fails to improve performance: 8-MiB AG is 358.16/357.35 us. Both pass 60 component
and 144 mixed ABI calls/rank. The stream-ID optimization is not retained.

Removing only the cooperative kernel's final post-ACK grid barrier retains
the pre-ACK reader barrier and CUDA kernel completion. This passes 60 component
and 144 mixed ABI calls/rank, but 8-MiB AG remains 357.12/355.81 us versus
355.37 us reference; no complete performance success is claimed.

RS alternating banks extended to 32 MiB pass the 144-call ABI and three full
RS sweeps. 16-MiB out/in are 910.53/907.36 us versus 913.60 us reference.
32-MiB is 1824.88/1817.81 us versus 1824.8509 us: the tiny positive out-of-place
ratio remains a failure, not a rounded pass. The first staging attempt ran out
of remote disk space; the dependent measurement attempt failed without usable
timings. Both failed directories are preserved. After reclaiming only completed
binary copies with exact local hashes, abi_retry and the full sweeps passed.

## Redirecting and fusing RS output

The own-half final local RS result can be written to actual caller output;
only the opposite half is sent over IB. In-place safety follows because the
last own contribution is read before that element is overwritten, and future
pieces are disjoint. The per-call descriptor still supplies actual pointers,
including rotated and recycled allocations. Unaligned output uses the existing
client load/store helpers.

The partial-output experiment keeps the final remote-add kernel, and passes
144 mixed ABI calls/rank plus three full RS sweeps. 1-GiB out/in improve to
53.831/53.557 ms versus 53.693 ms reference. A second experiment moves the
remote wait before the final own-half add and fuses both additions into one
kernel. It preserves (own + local-peer-partial) + remote arithmetic order,
all local forwarding, the remote-send buffer and final send retirement. Its
144-call ABI and three RS sweeps pass. 1-GiB becomes 53.458/53.375 ms; remaining
slower rows in that isolated build are 16 MiB (both layouts) and 32 MiB out.
No native NCCL is loaded or executed in any candidate.

## Whole-library and hybrid evidence

The [combined partial-output build](ar512_persistent/uccl_combined_partial/README.md)
passes 60 component, 144 mixed ABI calls/rank and nine full nccl-tests jobs.
Its complete diagnostic has 14/126 slower rows (AG 6, AR 5, RS 3). The noisy
128-KiB AR out-of-place result is preserved as a failed row.

A separate build extends hybrid AR down to 1 MiB, reusing already-fast RS and
cooperative AG while leaving the 512-KiB direct-input policy in place. Its
144-call ABI and three full AR sweeps pass. Out/in at 1, 2 and 4 MiB are
142.91/143.07, 243.33/243.24 and 442.56/442.46 us, versus UCCL references
180.02, 306.73 and 495.50 us. These separate wins must be validated together
in a new fixed binary; they are not a synthetic acceptance curve.
