# P800 topology, build and device-transfer qualification

The P800 CUDA-compatible runtime reports PCI vendor `0x1d22`. Resolve each
runtime-returned BDF directly to preserve device ordinals, including visibility
remapping and restricted sysfs. P2P checks the ordinal before indexing; NIC
distance ranking and `UCCL_P2P_RDMA_DEV` overrides retain their existing behavior.
The topology cache uses `filesystem::hash_value` for GCC 9 compatibility.

## Build

Use the P800 vendor's matching CUDA-compatible **headers and libraries**, not
NVIDIA runtime libraries. The tested layout has headers under `/usr/local/cuda`
and runtime libraries under `/usr/local/xcudart/lib`. Keep that library directory
first at build and run time. Other installations can override `CUDA_INC` and
`CUDA_LIB` independently. No separate native XPU backend is introduced here.

Install the normal P2P prerequisites: C++17 compiler, Python development headers,
`nanobind`, RDMA/verbs and EFA headers, libfabric headers, zlib and libelf
development packages. Some Mellanox development packages omit `efadv.h`, and
Ubuntu 20.04's libfabric headers predate the APIs used by the current CXI code.
The qualification build uses EFA headers from rdma-core v47.0 and libfabric
v1.22.0 headers in a separate prefix. `EFA_HOME` and `LIBFABRIC_HOME` select that
prefix; do not replace the host's driver libraries or a running environment.
If the Python environment lacks `python-config`, set `PYTHON_CONFIG` to the
matching interpreter's `python3-config`.

```sh
: "${UCCL_BUILD_DEPS:?Prefix containing include/infiniband/efadv.h and include/rdma/fabric.h}"
export LD_LIBRARY_PATH=/usr/local/xcudart/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
make -C p2p -j2 all PYTHON=python3 \
  CUDA_HOME=/usr/local/cuda CUDA_LIB=/usr/local/xcudart/lib \
  EFA_HOME="$UCCL_BUILD_DEPS" LIBFABRIC_HOME="$UCCL_BUILD_DEPS"
# Install into the isolated Python environment used for the test.
make -C p2p install PYTHON=python3 CUDA_LIB=/usr/local/xcudart/lib \
  EFA_HOME="$UCCL_BUILD_DEPS" LIBFABRIC_HOME="$UCCL_BUILD_DEPS"
# Check immediate dynamic-symbol resolution in the same isolated environment.
# Importing the extension does not construct an endpoint or run a device test.
LD_BIND_NOW=1 python3 -c 'from uccl import p2p; print(p2p.__file__)'
```

Standalone CPU checks need no accelerator, SDK or Python package installation:

```sh
make -C p2p test_gpu_topology PYTHON=/nonexistent/python
./p2p/test_gpu_topology
python3 -m unittest discover -s p2p/tests -p test_p800_device_transfer_control.py
```

The C++ test covers mixed PCI vendors, reordered visibility, missing sysfs,
empty enumeration and equivalent path keys. The Python tests exercise protocol
completion, full-buffer corruption rejection, staging directions and cleanup
using host-memory doubles; they are not device/RDMA validation. Injected faults
include swapped, repeated and rotated blocks, stale payloads, dropped writes
and a corrupted final byte. They also cover writes into either guard, ignored
source/destination offsets, and omitted staging copies. Each must fail without
a PASS result on either applicable path.
The runtime tests also model pageable H2D copies that return before device
completion, checking initialization ordering and completion-error propagation.

## Cross-node device test

Run only on available, assigned devices. Select one local device on each of two
nodes using the vendor runtime's visibility settings. Both peers must use the
same build, runtime library and test options. Set `UCCL_P2P_TRANSPORT=rdma` and,
if desired, `UCCL_P2P_RDMA_DEV` to a routable data NIC on each node. The probe uses
network `Endpoint.connect`/`write`, not the intra-node IPC shortcut. Provide XPU
and RDMA device permissions, readable PCI sysfs, a sufficient locked-memory
limit, and network reachability for the rendezvous and UCCL-selected ports.
Set `UCCL_P2P_COMPRESS_STRATEGY=none` on both peers. The probe rejects enabled
or unknown compression settings before importing UCCL or initializing the
runtime: DietGPU builds can otherwise select a different data path and allocate
compression buffers. Unset/empty, `none`, `off` and `0` are accepted (case
insensitive). This qualification measures uncompressed RDMA payload bytes.

Start the receiver, then the sender (replace `RECEIVER_IP`):

```sh
# Receiver node
UCCL_P2P_TRANSPORT=rdma UCCL_P2P_COMPRESS_STRATEGY=none \
  timeout -k 5s 600s python3 p2p/tests/p800_device_transfer.py \
  --role receiver --gpu 0 --path host-staged \
  --runtime-lib /usr/local/xcudart/lib/libcudart.so
# Sender node
UCCL_P2P_TRANSPORT=rdma UCCL_P2P_COMPRESS_STRATEGY=none \
  timeout -k 5s 600s python3 p2p/tests/p800_device_transfer.py \
  --role sender --peer RECEIVER_IP --gpu 0 --path host-staged \
  --runtime-lib /usr/local/xcudart/lib/libcudart.so
```

`host-staged` explicitly copies **device -> host -> UCCL RDMA -> host -> device**
in the probe. It is application staging, not transparent engine support or
GPU-direct RDMA. `direct` registers device allocations on both nodes and fails
if device registration/transfer is unsupported; it never silently stages.
Repeat with `--path direct` on both peers to test that separate path, then swap
sender/receiver to cover both directions. Native calls require the outer timeout.

`mapped-alias` is a third, explicit vendor-runtime experiment. It queries
`cudaPointerGetAttributes` for the allocated device pointer, requires device
memory on the selected ordinal and a distinct, non-null `hostPointer`, and
uses that alias for registration, advertisement and RDMA addresses. Buffer
initialization and full verification still use the original device allocation.
There are no application D2H/H2D staging copies in the measured transfer path.
This is not an automatic fallback, a change to UCCL's address handling, or proof
of the runtime's internal physical data path. Run it with `--path mapped-alias`
on both peers; missing aliases or attribute-query failures fail the test.

Defaults are 1 KiB, 1 MiB and 16 MiB, with two warmups and 20 measured transfers
per size. Each transfer uses a deterministic SHAKE256 byte stream keyed by size
and sequence, poisons the destination and checks every byte copied back from
the destination device before acknowledging success. The pattern varies across
block positions, so swapping or repeating whole blocks is also detectable.
The rendezvous plan includes the protocol version; both peers must use the
same version of this probe.
Each registered allocation includes 256-byte guards on both sides of the
payload, and only the interior range is advertised and transferred. Distinct
source/destination guards and poisoned payloads are reset each iteration. The
probe verifies the entire device allocation on both peers and, for staging,
the host allocation too. This detects writes outside the requested range and
incorrect use of a registration's base address. Staging copies move only the
payload; the guards are excluded from the transferred byte count.
JSON results include the verified count and decimal GB/s. `transport_GBps`
times serial blocking UCCL writes; `device_path_GBps` also includes source D2H
and destination H2D for staging. Allocation, registration, initialization,
verification and TCP control messages are excluded. This is basic single-flow
bandwidth, not a saturated or concurrent throughput measurement.
Every H2D copy explicitly waits with `cudaDeviceSynchronize`: pageable host
memory can be staged before the final device copy has completed under the
[CUDA-compatible API contract](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).
This ensures initialization finishes before buffers are used by RDMA. The
destination H2D completion wait is inside `device_path_seconds`; it is not
deferred to the later verification read. Protocol v6 requires both peers to
enforce the uncompressed path and rejects earlier peers. Each result includes
its protocol version, `transport=rdma` and `compression=none`. CPU checks alone do
not establish the runtime's native synchronization behavior; see the bounded
native observations below.

## Qualification status

Historical native tests establish runtime ordinal discovery and host-buffer
registration/writes. Detailed environment information and original evidence are
linked in [PR #1072's discussion](https://github.com/uccl-project/uccl/pull/1072).
Raw logs, old patches and manifests are intentionally outside the source tree.

Native device-buffer results now distinguish explicit host staging, the original
device-address registration failure, and a vendor mapped-alias path. They do not
establish general P800 platform support. Hardware-direct RDMA, DMA-BUF device
registration, intra-node IPC/XPULink, graphs, NIXL and real-model concurrency remain
unqualified. The PR retains its topology scope.

### Native device-buffer observation, 2026-10-25

Two P800 OAM nodes, one assigned device per node (`ordinal 0`, PCI `0000:23:00.0`),
XPU driver/runtime `5.24.0.0` / `5.24.0`, vendor CUDA-compatible libraries,
`mlx5_1` RoCE v2, GID index 3, MTU 4096. Both peers used the same extension
(SHA256 `9f59206f816baeebb4c5ceb8feb998d19584177de82e8a9095622f70f4dbe22b`).
The 103 non-test files under `p2p/` and `include/` matched
`6a91aa0292a17236846a7f2e176dd8f3f39de16c`; later changes are probe/tests/docs only.

Each row below passed 22 complete device-allocation/guard checks (2 warmups,
20 measured). This gives **132 verified transfers per path, 264 total**.
All eight peers exited 0, deregistered/freed their buffers, and were removed.
Results are single-flow serial timing with verification excluded, not saturated
bandwidth or model throughput. A and B denote the two physical hosts.

| Path | Direction | Payload | UCCL transport GB/s | Device path GB/s |
| --- | --- | --- | ---: | ---: |
| host-staged | A → B | 1 KiB | 0.04976 | 0.01514 |
| host-staged | A → B | 1 MiB | 15.77564 | 2.95510 |
| host-staged | A → B | 16 MiB | 25.58361 | 2.94164 |
| host-staged | B → A | 1 KiB | 0.04731 | 0.01382 |
| host-staged | B → A | 1 MiB | 15.68690 | 2.94705 |
| host-staged | B → A | 16 MiB | 26.79201 | 2.94036 |
| mapped-alias | A → B | 1 KiB | 0.04975 | 0.04869 |
| mapped-alias | A → B | 1 MiB | 16.11146 | 15.97898 |
| mapped-alias | A → B | 16 MiB | 25.53461 | 25.43198 |
| mapped-alias | B → A | 1 KiB | 0.04888 | 0.04781 |
| mapped-alias | B → A | 1 MiB | 15.30240 | 15.14439 |
| mapped-alias | B → A | 16 MiB | 25.39466 | 25.28660 |

Host staging used protocol v5 (probe SHA256
`68ef45182e9f29937b655b669b6a6adf6d49e8309aa58c5dce43a376bde0a767`).
Mapped-alias results are from the protocol v6 probe in this tree (SHA256
`afaeaf7b54c3d8a876d96f23ddfa0254ad40e33f46de3872790ef1c2c4150cdc`).
An earlier private alias prototype also passed 132 transfers; it is excluded
from this table and count. Full original logs and independent SHA/member
readback receipts are retained outside the source tree.

The original `direct` path failed on **both nodes before any transfer**:
registration of the first 1536-byte allocation returned `EFAULT / Bad address`.
A subsequent one-node registration-only diagnostic also failed at 64 KiB and
2 MiB. `kunlun_peermem` was loaded on both nodes. The runtime reported device
memory on ordinal 0 with distinct device and host-alias addresses; registering
the latter succeeded. This does not establish the root cause or justify
transparent address substitution in the engine. The reverse direct run and
larger direct transfers were not executed after this registration gate failed.
No direct-path bandwidth is reported.

To reproduce the mapped-alias rows, use the commands above with
`--path mapped-alias`, `UCCL_P2P_RDMA_DEV=mlx5_1`,
`UCCL_P2P_RDMA_GID_INDEX=3`, and the same socket interface/visible device on each
assigned node. This validates the explicit alias path only: the runtime may
have internal mappings or copies that this probe does not instrument.
