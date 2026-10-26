# Standalone GIN contract

The device API publishes commands through a mapped host FIFO. Each channel's
CPU proxy submits payload writes and ordered updates, then publishes completion
for the device to observe. The implementation reuses the
[rail-primitives source at 29e7e7ca](https://github.com/DanielDanyang/uccl-danyang/tree/29e7e7ca868590fb3a70bc96ebf42271983ab9b6/experimental/uccl_gin).

## Ownership and completion

| Resource | Owner and lifetime |
|---|---|
| Symmetric payload window | The host Context owns its allocation. Proxies borrow it without freeing it. |
| Device resource bundle | Copied into kernel arguments; its handles refer to device-resident queues. |
| Channel FIFO and queue hints | Owned by Context/proxy setup; retain the proxy-major channel layout. |
| `put_value` source bytes | Private proxy staging remains registered until its write completes. |

Publish each command before waiting for its completion. Payload source storage
must remain valid until the corresponding completion or drain. Context teardown
drains/stops its proxies and releases their registrations and handles before
releasing its payload allocation.

`put`, `put_tail_add`, `put_value` and `red_add_rel` share the rail command
encoding. Payloads larger than the 24-bit command length are split into aligned
chunks. Tail signals follow payload completion. Ordered updates to one counter
must use a consistent lane; a queue inflight limit alone does not prove the
per-counter sequence bound.

`quiet` drains the selected lane. Scalar `flush` covers the queues for one
caller. Cooperative group completion and the typed NCCL-EP adapter are separate
increments; their actual contract belongs with their implementation.

## Platform and topology limits

- The CUDA transport defaults to EFA and requires the modern RDMA SDK selected
  by the build CI. Device/NIC selection and shared GPU runtime mappings remain
  on the mainline implementations.
- `PLATFORM=rocm GFX=gfx942` selects the HIP/RoCE build branch. It covers the
  put/quiet path; the EFA ordered-atomic protocol does not establish a native
  RoCE atomic backend. This PR has no AMD GPU/network execution qualification.
- The standalone backend implements rail operations. LSA operations and
  unsupported device primitives fail explicitly.

## Validation boundary

The local fixtures use the production FIFO and command encoding with a test
consumer that supplies completions. They check fields, duplicate/missing
commands, queue wrapping/reuse and normal drain. They qualify that queue path;
actual NIC receiver visibility, distributed source lifetime and distributed
Context teardown require separate network tests.

[Build and API entry point](README.md) ·
[Local test commands](tests/local/README.md) ·
[Completed native qualification](tests/local/MAIN_INTEGRATION_RESULTS.md)
