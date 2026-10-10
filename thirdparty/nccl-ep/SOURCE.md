# NCCL-EP source

This directory reuses the NCCL-EP snapshot and UCCL adapter from
[DanielDanyang/uccl-danyang at 29e7e7ca](https://github.com/DanielDanyang/uccl-danyang/tree/29e7e7ca868590fb3a70bc96ebf42271983ab9b6/thirdparty/nccl-ep).
NVIDIA and DeepSeek copyright notices, Apache-2.0 licensing and third-party
notices are retained. Local changes wire host resources through kernel parameters,
fix cooperative completion and typed signals, add single-rank HT validation and
make the build work outside an NCCL source checkout. The imported Python ctypes
wrapper is omitted; this integration builds the native C/CUDA library.

`NCCL_EP_USE_UCCL_GIN=ON` selects UCCL for HT rail operations; intra-node work
retains NCCL/CUDA primitives. The host adapter bootstraps over the EP NCCL
communicator and registers the existing GIN payload window with UCCL. It does not
require MPI_COMM_WORLD to match an EP subgroup. Only contiguous, equal-size node
rank groups are supported. The 13-bit atomic byte-offset protocol bounds the
indexed signal range; unsupported topologies return ncclInvalidUsage at setup.
Distributed RDMA execution still needs a suitable testbed. Native local fixtures
check kernel behavior with test completions and cannot qualify network E2E.
