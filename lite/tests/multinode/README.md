# Explicit two-host network library check

The optional `libuccl_lite_multinode.so` exposes `MultinodeCommunicator` through
`include/uccl_lite/collectives.h`. Its implementation and retained FP32 network
kernels are private to `src/multinode/`. The library links MSCCL++ and CUDA,
not MPI or NCCL. The test harness uses MPI only for the lifetime-ID exchange
and process coordination.

Build using the desired IB-enabled MSCCL++ build and MPI installation:

```bash
make multinode-library build/multinode_communicator_check \
  build/multinode_communicator_check_static \
  MSCCLPP_IB_BUILD=$HOME/third_party/mscclpp/build-ib-pcieccl \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx
```

Stage the same shared library, test binaries and MSCCL++/CUDA dependencies at
identical absolute paths on both hosts. Then:

```bash
HOSTS=node0:4,node1:4 SOCKET_IFNAME=ibp55s0f0 \
  make multinode-library-check \
  MSCCLPP_IB_BUILD=$HOME/third_party/mscclpp/build-ib-pcieccl \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx \
  MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun

HOSTS=node0:4,node1:4 SOCKET_IFNAME=ibp55s0f0 \
  MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
  bash scripts/run_multinode_library.sh --static
```

The explicit runner uses the existing testbed's `host-no-atomic` IB mode and
`mlx5_0,mlx5_1` HCA map. Override `IBV_MODE`, `HCA_DEVICES`, `BUILD_DIR` or
`TIMEOUT_SECONDS` when appropriate. No network backend is enabled in the NCCL
interposer by this check or by GPU count.

Each rank performs 90 FP32 AR/AG/RS calls over two communicator lifetimes,
full extents 128 B–1 MiB, changing size/family and alternating streams. Odd
ranks use NCCL in-place layouts; each rank has a different pointer alignment.
Checks compare every output element to an analytic signed/index-dependent
oracle and verify guards. One producer is delayed. Unprepared enqueue,
unsupported counts and graph capture must reject without launching work.
Transport setup is explicit and outside any timing; construction alone does
not allocate GPU or network resources. All ranks must call prepare and enqueue
in the same order with matching operations/counts and equal capacity.

The current backend uses retained single-CTA serial-local kernels and stable
registered buffers. An entry handshake protects prior output copies before
buffer reuse. New API timings must include the handshake and D2D input/output
copies; historical kernel-only measurements cannot establish performance.
There is no automatic routing/admission policy, graph execution or in-process
network fault recovery. The optional NCCL ABI integration and exact two-host calibration are described
in [the interposer checks](../nccl_dropin/README.md); the raw library itself
does not select routes.

The explicit double-buffer constructor additionally reserves two independent
banks. Each retained kernel reaches an eight-rank progress point: reaching
call i+2 proves every peer entered i+1 after finishing its output for i, so
bank i is reusable even across changing families/sizes. The original constructor
keeps the synchronized single-bank behavior.

`enqueueOrdered(call, completion)` is an integration entry point. Its caller
supplies stream ordering and records the same borrowed completion event after
each call, including any external staging. That event remains alive through
destruction. This path fuses caller input/output movement into the serial-local
kernels, including natural four-byte FP32 alignment, and avoids duplicate
completion events/copy submissions. The ordinary `enqueue()` retains internal
events and explicit copies. Do not mix ordering modes on one instance.

`build/multinode_communicator_check_double` and
`build/multinode_communicator_check_ordered` exercise the two-bank modes.
The retained standalone kernels instantiate the original non-client-I/O path.
