# SM-budget experiment design (2026-09-04 revision)

## Principle

The project's claim is *fewer SMs than native NCCL*, so no measured shim
configuration may use more SM blocks than native NCCL uses coll channels in
the same test. For every collective and placement we therefore:

1. measure the native channel budget `B` (coll channels at that placement);
2. sweep the shim's persistent-worker block count `b <= B`;
3. pick the best `b*` (highest busbw; ties go to the smaller `b`, i.e.
   fewer SMs);
4. report the final matrix at `b*` against native at its `B` channels.

SM-resident shim work (the data path) is `b` persistent worker blocks, so
`b` is the SM count reported. AllToAll is a pure CE/IPC path: the persistent
worker is out of the data path and the measurement is reported as **0 SM**,
regardless of implementation details (no block sweep).

## Budget source

`B` = native NCCL coll channels, captured per placement from
`NCCL_DEBUG=INFO` (`nchannels`) on the same binaries used for the suite.
Channels are a communicator property in NCCL, so the same `B` applies to
AllReduce, ReduceScatter, and AllGather at that placement.

- B300 (NVSwitch, native 2.29.7+cuda13.2): `B = 32` at S2/S4/S8
  (measured 2026-09-04, `NCCL_DEBUG=INFO`).
- L40S (PCIe Gen5 + ConnectX-6, native 2.31.2): `B` measured per placement
  S2/S4/S8 and X4/X8/X16 (2+2/4+4/8+8 across node5/node6).

Measured 2026-09-04: L40S `B = 4` at S2/S4, `B = 2` at S8 and at all
cross-node placements X4/X8/X16.

## Collective classification

| collective | shim data path | SM count | sweep? |
|---|---|---:|---|
| AllReduce | fused reduce+copy on worker (`FUSE_REDUCE_COPY=1 FUSE_AG_COPY=1`) | `b` | yes |
| ReduceScatter | worker reduce (+CE/IPC ring puts) | `b` | yes |
| AllGather | CE/IPC ring puts + local publish copy (worker) | `b` | yes (light; expected flat) |
| AllToAll | CE/IPC puts, self-slice on user stream | **0** | no |

## Phase 1 — SM budget sweep

For each platform, placement, and collective in {AllReduce(fused),
ReduceScatter, AllGather}:

- candidate blocks `b` in `{1,2,4,8,16,24,28,32}` clipped to `B`
  (L40S candidate set clipped to measured `B`: `{1,2,4}` where `B=4`,
  `{1,2}` where `B=2`);
- primary size 256 MiB (bandwidth/SM-saturation regime), check size 16 MiB;
- nccl-tests `-n 10 -w 2`, validation on, one run per candidate (sweep),
  then medians of 3 at the top two candidates;
- selection: `b* = min{ b : busbw(b) >= 0.97 * max_busbw }` at 256 MiB.

Expected shape (to verify): bandwidth rises to a knee well below `B`, then
flattens — the "fewer SMs" story. AllGather is expected to be flat because
only the local publish copy touches the worker.

## Phase 2 — final matrices (all at `b*`, medians of 3, 0 wrong)

### B300 (S2/S4/S8)

- AllReduce fused@`b*` vs native (sizes 1M..256M, f4)
- ReduceScatter @`b*` vs native (1M..256M)
- AllGather CE @`b*` vs native (1M..256M)
- AllToAll CE (0 SM) vs native (1M..256M)
- AllToAll rotation A/B at 256 MiB (`UK_CCL_A2A_ROTATE` 1 vs 0)

### L40S (S2/S4/S8/X4/X8/X16)

- AllReduce unfused CE/IPC @`b*` vs native (1M..256M)
- ReduceScatter @`b*` vs native (1M..256M)
- AllGather CE @`b*` vs native (1M..256M)
- AllToAll CE (0 SM) vs native (1M..256M)
- AllToAll rotation A/B at 256 MiB (same-node placements)

Cross-node shim runs add `UK_CCL_RDMA_FUSED_MODE=proxy` and the
`-x LD_LIBRARY_PATH` / `-x UK_CCL_UNBIND=1` environment of the L40S
procedure.

## Mechanism add-on

B300 host latency decomposition at 1M/4M AllReduce fused@`b*`
(`UK_CCL_DEBUG=3`, HostProf) to attribute the small-message floor.

## Reporting

Every table lists the shim config's `b` (or "0 SM (CE)" for AllToAll) next
to native's `B`; run-to-run medians of 3; all cells validated.
