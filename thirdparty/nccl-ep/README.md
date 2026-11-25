# NCCL-EP with UCCL-GIN

This licensed source snapshot and its local adapter are described in
[SOURCE.md](SOURCE.md). The native C API is declared in
[include/nccl_ep.h](include/nccl_ep.h); this integration has no Python wrapper.

## Build

```sh
make -C thirdparty/nccl-ep CUDA_HOME=/usr/local/cuda SM=120 \
  NCCL_DIR=/path/to/nccl NCCL_EP_USE_UCCL_GIN=ON
```

Use installed NCCL 2.30+ headers/libraries. Set `NCCL_EP_USE_UCCL_GIN=OFF` for the
native NCCL backend. `CMAKE_FLAGS` accepts
`-DCMAKE_PREFIX_PATH=/path/to/private/deps` for ibverbs/libnuma/libnl. EFA is the
UCCL transport default; select the RDMA interface with `UCCL_GIN_IFNAME`.
`NCCL_EP_BUILD_EXAMPLES=ON` builds the C++ EP test/benchmark and requires MPI;
the library itself bootstraps through the NCCL communicator.

## Integration contract

- UCCL handles HT rail operations; LSA/NVLink and the EP communicator retain
  their NCCL/CUDA implementation. The vendored LL code is outside this UCCL
  adapter's validation scope.
- Distributed groups require equal-size contiguous node rank layouts and
  signal IDs within the supported indexed range. Unsupported layouts fail at
  setup. A single-rank HT fixture requires an LSA1 runtime build.
- Tensor arrays carry typed token, top-k-index and probability buffers. Keep
  their storage and routing handle alive until completion. Group teardown
  drains transport resources before releasing the borrowed window.

[Adapter ownership and completion](../../experimental/uccl_gin/MAIN_DESIGN.md)
explains these contracts. Local device checks use real NCCL headers without a
network communicator; [native qualification](../../experimental/uccl_gin/tests/local/MAIN_FIX_RESULTS.md)
and complete-model local-copy timing have separate execution scopes.

The complete original API reference is available at the
[pinned imported source](https://github.com/NVIDIA/nccl/blob/1933fdd6360a8bfccaa0166bd71bce363d32e5b6/contrib/nccl_ep/README.md).
Its Python examples and reference performance are upstream material, not this
adapter's execution results. [License](LICENSE.txt) and
[third-party notices](ThirdPartyNotices.txt) remain with the source.
