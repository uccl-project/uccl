# NCCL-EP source

This directory reuses NVIDIA's NCCL-EP source from
[NCCL v2.30.4-1, commit 1933fdd6](https://github.com/NVIDIA/nccl/tree/1933fdd6360a8bfccaa0166bd71bce363d32e5b6/contrib/nccl_ep),
with the local UCCL adapter. The imported public header and original API guide
match that pinned snapshot; the guide is linked from [README.md](README.md).
NVIDIA and DeepSeek copyright notices, Apache-2.0 licensing and third-party
notices are retained. Local changes wire host resources through kernel parameters,
fix cooperative completion and typed signals, add single-rank HT validation and
make the build work outside an NCCL source checkout. HT metadata preprocessing
decodes the globally packed expert bitmap, and backward combine returns weights
in the original top-k slots. The imported Python ctypes wrapper is omitted; this
integration builds the native C/CUDA library.

`NCCL_EP_USE_UCCL_GIN=ON` selects UCCL for HT rail operations; intra-node work
retains NCCL/CUDA primitives. The host adapter bootstraps over the EP NCCL
communicator and registers the existing GIN payload window with UCCL. It does not
require MPI_COMM_WORLD to match an EP subgroup. Only contiguous, equal-size node
rank groups are supported. The 13-bit atomic byte-offset protocol bounds the
indexed signal range; unsupported topologies return ncclInvalidUsage at setup.
Distributed RDMA execution still needs a suitable testbed. Native local fixtures
check kernel behavior with test completions and cannot qualify network E2E.
