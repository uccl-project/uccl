# Runtime-visible accelerator topology regression

`get_gpu_cards()` must preserve the GPU runtime's ordinal-to-PCI mapping.
On a P800 CUDA-compatible runtime, the runtime enumerated a device with PCI
vendor `0x1d22`, but the vendor filter discarded it. Automatic P2P NIC selection
then indexed an empty vector and terminated with SIGSEGV. Selecting a NIC with
`UCCL_P2P_RDMA_DEV` bypassed that path.

The fix resolves each runtime-returned BDF directly, retaining its ordinal even
when sysfs resolution is restricted. P2P also checks the ordinal before indexing.
It changes neither the distance strategy nor the explicit NIC override.

## CPU regression

```sh
make -C p2p test_gpu_topology
./p2p/test_gpu_topology
```

The test creates temporary sysfs fixtures for P800, NVIDIA, AMD and Cambricon
vendors. It checks mixed vendors, visibility reordering, a missing sysfs entry,
and an empty runtime enumeration. It requires C++17 and no GPU or NIC.

The exact formatted test source passed with GCC 9.4 on 2026-10-14. The modified
C++ files were formatted with clang-format 14.0.6.

## Native comparison on P800

Base: `0578e0b819831f35c4ad17008e65e2159746ef80`. Both builds used the same
environment and a separate GCC 9 build compatibility shim that supplies a
`filesystem::path` hasher using `filesystem::hash_value`. That shim is outside
this topology change. The runtime provided CUDA device enumeration, BDF lookup
and host/device copy interfaces; the NIC was Mellanox mlx5.

The same native probe called `uccl_engine_create`, registered a 1 MiB **host**
buffer, obtained engine metadata, deregistered it, and destroyed the engine.
One physical accelerator was exposed as runtime ordinal zero in each run.

| Build | NIC selection | Result |
| --- | --- | --- |
| Base plus common build shim | Automatic | SIGSEGV, exit 139, before registration completed |
| Base plus common build shim | Explicit RDMA NIC | Registration, metadata and cleanup passed, exit 0 |
| Topology fix plus common build shim | Automatic | Registration, metadata and cleanup passed, exit 0 |

An additional two-node test using an explicit identical NIC choice completed
nine host-memory writes with the topology build: 1 KiB, 1 MiB and 16 MiB, with
three changing payloads per size and receiver-side SHA256 verification.
Both processes exited zero. This is a finite transport correctness check, not a
bandwidth measurement.

The native comparison establishes automatic topology discovery on the tested
P800 runtime. Direct GPU-memory RDMA, CUDA IPC on NVIDIA/AMD hardware, Graph
replay, NIXL plugin integration and model-level acceleration require their own
qualification. A separate blocking C API accept race was observed during later
transfer tests; this change does not alter the accept implementation.
