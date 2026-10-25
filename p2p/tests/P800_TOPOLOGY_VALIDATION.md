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
deferred to the later verification read. Protocol v5 requires both peers to
enforce the uncompressed path and rejects earlier peers. Each result includes
its protocol version, `transport=rdma` and `compression=none`. These CPU checks do
not establish the P800 runtime's native synchronization behavior.

## Qualification status

Historical native tests establish runtime ordinal discovery and host-buffer
registration/writes. Detailed environment information and original evidence are
linked in [PR #1072's discussion](https://github.com/uccl-project/uccl/pull/1072).
Raw logs, old patches and manifests are intentionally outside the source tree.

P800 device-buffer transfer correctness and bandwidth remain **pending**. The
probe and CPU checks do not establish platform support. GPU-direct RDMA,
DMA-BUF device registration, intra-node IPC/XPULink, graphs, NIXL and real-model
concurrency/performance are also unqualified. Publish actual device results and
the observed transfer path before expanding the PR's supported-platform claim.
