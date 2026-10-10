# NCCL-EP with UCCL-GIN

This is the source snapshot described in [SOURCE.md](SOURCE.md).
For a standalone build against installed NCCL 2.30+:

```sh
make -C thirdparty/nccl-ep CUDA_HOME=/usr/local/cuda SM=120 \
  NCCL_DIR=/path/to/nccl NCCL_EP_USE_UCCL_GIN=ON
```

Set `NCCL_EP_USE_UCCL_GIN=OFF` for the native NCCL path. `CMAKE_FLAGS` accepts
`-DCMAKE_PREFIX_PATH=/path/to/private/deps` for ibverbs/libnuma/libnl headers and
libraries. EFA is enabled by default. The caller must provide a supported RDMA
interface through `UCCL_GIN_IFNAME`. Examples require `NCCL_EP_BUILD_EXAMPLES=ON`
and an MPI development installation; the library itself uses NCCL bootstrap.

Local device checks are built from `experimental/uccl_gin/tests/local` with real
NCCL device headers. They link no NCCL runtime and create no network communicator.
[Native test results](../../experimental/uccl_gin/tests/local/MAIN_FIX_RESULTS.md)
will distinguish those fixtures from distributed/model E2E.

The upstream NCCL-EP API guide is retained in [API_GUIDE.md](API_GUIDE.md).
