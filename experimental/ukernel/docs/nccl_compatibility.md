# NCCL compatibility

What the ukernel NCCL shim supports, how it matches native NCCL
semantics, and how it plugs in as a drop-in. See also
[`perf_test_procedure.md`](perf_test_procedure.md) for building/running
perftests against the shim.

## Drop-in ABI

- Install prefix (`build/nccl/` by default) ships `libnccl.so.2`
  (native libnccl's SONAME) + `libnccl.so` dev symlink; the ROCm build
  additionally ships `librccl.so.1` + `librccl.so` (RCCL's SONAME).
  Perftests/frameworks built against a normal NCCL/RCCL install resolve
  the shim at runtime via `LD_LIBRARY_PATH` — nothing is compiled
  against the shim itself.
- NCCL 2.19+ ABI symbols used by modern perftests are provided:
  `ncclMemAlloc`/`ncclMemFree` wrap `cudaMalloc`/`cudaFree`
  (`hipMalloc`/`hipFree` on ROCm), `ncclCommRegister`/`ncclCommDeregister`
  are no-ops (the shim never moves buffers), and custom pre-mul-sum
  ops are unsupported (`ncclRedOpCreatePreMulSum` returns
  `ncclInvalidUsage`, `ncclRedOpDestroy` is a no-op).

## Supported APIs

```text
ncclGetUniqueId
ncclCommInitRank
ncclCommInitAll            (deprecated NCCL path; one comm per device)
ncclAllReduce              (ring + opt-in binary tree)
ncclAllGather
ncclReduceScatter
ncclAllToAll               (in-place only; shim extension — upstream
                            NCCL has no ncclAllToAll, nccl-tests builds
                            the exchange from ncclSend/ncclRecv)
ncclCommDestroy / ncclCommAbort / ncclCommFinalize
ncclCommCount / ncclCommUserRank
ncclCommGetAsyncError
ncclGetErrorString
ncclGetVersion
ncclGroupStart / ncclGroupEnd
ncclMemAlloc / ncclMemFree
ncclCommRegister / ncclCommDeregister
ncclRedOpDestroy
```

## In-place semantics

Match NCCL:

- AllReduce supports both placements (`sendbuff == recvbuff`, and
  out-of-place);
- AllGather / ReduceScatter detect NCCL's in-place form (sendbuff
  pointing inside recvbuff, and vice versa) and run the in-place
  algorithm variant;
- AllToAll supports out-of-place (`sendbuff != recvbuff`, preferred —
  pure IPC puts, no staging) and in-place (runs a staged variant for
  correctness).

## Unsupported APIs

`ncclBroadcast`, `ncclReduce`, `ncclSend`, `ncclRecv`, and custom
reduction ops (`ncclRedOpCreatePreMulSum`) return `ncclInvalidUsage`.
The shim exposes no `ncclBarrier` — upstream NCCL has none, and
extensions like that would break the drop-in contract.
Of the stock nccl-tests binaries, only `all_reduce_perf` (both
placements), `all_gather_perf`, and `reduce_scatter_perf` pass;
`broadcast_perf` / `reduce_perf` / `alltoall_perf` / `sendrecv_perf`
fail by design (their APIs are not implemented).

## Algorithm notes

- Binary-tree AllReduce is opt-in via `UK_CCL_TREE_THRESHOLD_BYTES`
  (default 0 = never). With `nranks == 2` the tree degenerates to the
  ring's shape, so the crossover can only be calibrated on a
  larger-rank environment.
- Version reporting: `ncclGetVersion` returns the value baked in
  `include/nccl.h` (`NCCL_VERSION_CODE`, currently 2.9.0+ukernel), not
  the native library's version.
