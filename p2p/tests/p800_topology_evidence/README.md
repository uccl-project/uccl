# P800 topology validation supplement for UCCL #1072

Prepared from preserved historical evidence and a new finite native retest on 2026-10-21. A repository copy accompanies the Makefile/CI follow-up; the native libraries tested below are bound to the earlier commit explicitly listed here.

## Environment

- P800 OAM, PCI vendor `0x1d22`; one accelerator visible at runtime ordinal zero via `CUDA_VISIBLE_DEVICES=7` and `CUDA_DEVICE_ORDER=OAM_ID`.
- Historical SMI and the new retest both report driver **5.24.0.0** and XPU-RT **5.24.0**.
- Retest compiler: GCC **9.4.0**, Ubuntu **20.04.6 LTS**.
- CUDA-compatible libraries: `/usr/local/xcudart/lib`; headers: `/usr/local/cuda/include`.
- Retest API queries returned `cudaRuntimeGetVersion=12090` and `cuDriverGetVersion=10020`, both rc=0. These are the compatibility layer's reported integers, not NVIDIA driver package versions.
- mlx5 RDMA NICs. Explicit selection uses `mlx5_1`; automatic selection is recorded in each log.

## Exact source and build boundary

The historical baseline is `0578e0b819831f35c4ad17008e65e2159746ef80`.
The new matched baseline is `27a0abb2744c7d3f2b6391760eae1217b07b31b4`, the main parent of PR merge head `c912f851a53b4ee0b86affdc9d9b249a4543b886`. The native candidate library was built from that exact PR head plus the GCC 9 path-hash compatibility shim.

The two new builds use the same `PathHashCompat` implementation (`filesystem::hash_value`). The baseline has two affected maps; the candidate has one because this PR removes the old ordinal map. The separate patches in `patches/` show this explicitly. No transport, IPC, model or scheduler fix is folded into that shim.

Use separate clean checkouts of the two commits, apply the corresponding shim, and reproduce the recorded commands in `build-commands.sh`. Existing dependencies must include CUDA-compatible headers/libraries, Python/nanobind, RDMA/EFA and libfabric headers, and zlib/libelf development files. The proprietary P800 runtime is an external prerequisite; it is not distributed by this packet.

The new retest also found a test-target defect: `-I../include/util` shadows the system `<endian.h>` with the project's C++ header on this GCC 9/glibc environment. Removing that unnecessary include search path makes `make -C p2p test_gpu_topology` pass. `patches/cpu-makefile-fix.patch` records the minimal repair used during the native test preparation. The accompanying Makefile/CI follow-up also skips Python/nanobind discovery for this standalone host test and makes CI run the documented Makefile target. These changes do not alter either tested native library; their hashes and source commit remain those listed in `results.json`. The initial CPU failure and repaired test output are retained.

## Preparing the two source trees

Set `EVIDENCE` to this evidence directory, `ROOT` to a new writable build directory,
and `LAB` to the existing dependency installation (containing `.venv/` and `deps/`).
From a UCCL checkout containing both commits:

```sh
mkdir -p "$ROOT"
git worktree add --detach "$ROOT/baseline" 27a0abb2744c7d3f2b6391760eae1217b07b31b4
git worktree add --detach "$ROOT/candidate" c912f851a53b4ee0b86affdc9d9b249a4543b886
git -C "$ROOT/baseline" apply "$EVIDENCE/patches/baseline-gcc9-shim.patch"
git -C "$ROOT/candidate" apply "$EVIDENCE/patches/candidate-gcc9-shim.patch"
for src in "$EVIDENCE"/source/*.cc.txt; do
  name=$(basename "$src" .txt)
  cp "$src" "$ROOT/$name"
done
export ROOT LAB
bash "$EVIDENCE/build-commands.sh"
```

The commands retain the observed Python 3.10 shared-library path in the probe
link flags; adapt that path if the supplied runtime environment differs.
For each arm, run `ldd "$ROOT/uccl_capi_probe"` under the same `LD_LIBRARY_PATH`
used below and require `libuccl_p2p.so` to resolve to that arm. The probe has a
baseline RUNPATH; omitting this check can silently test the wrong library.

## Probe source and invocation

`source/uccl_capi_probe.cc.txt` is the recovered original probe, unchanged. It creates the engine, registers a 1 MiB **host** vector, obtains metadata, deregisters the region and destroys the engine. Registration success, metadata success, cleanup marker and exit code are required for a successful cell.

Run the same binary with the selected build's library first in the dynamic-library search path. The retest's `ldd` records verified the actual library selected for both arms.

```sh
env -u UCCL_P2P_RDMA_DEV \
  CUDA_VISIBLE_DEVICES=7 CUDA_DEVICE_ORDER=OAM_ID \
  XPU_USE_DEFAULT_CTX=1 CUDA_ENABLE_P2P_NO_UVA=1 \
  UCCL_P2P_DISABLE_IPC=1 NCCL_SOCKET_IFNAME=eth0 \
  UCCL_P2P_TRANSPORT=rdma \
  LD_LIBRARY_PATH="/usr/local/xcudart/lib:${ROOT}/${ARM}/p2p" \
  bash -c 'ulimit -c 0; exec "$@"' probe \
  timeout -k 3 25 "${ROOT}/uccl_capi_probe"
```

For explicit selection add `UCCL_P2P_RDMA_DEV=mlx5_1` to the environment. The isolated mapping probe in `source/topology_probe.cc.txt` also prints runtime and topology counts without indexing the returned topology vector. Under the same environment, replace the final executable with `${ROOT}/topology-${ARM}` to run it. Set `ARM=baseline` or `ARM=candidate`; run the automatic and explicit cases three times each (12 C API invocations total).

## Observed results

| Validation | Baseline | Candidate |
|---|---|---|
| Historical automatic NIC | exit 139, before registration | exit 0, registration/metadata/cleanup |
| Historical explicit NIC | exit 0 | not part of the historical three-cell comparison |
| New automatic NIC, 3 repetitions | **3/3 exit 134, `std::bad_alloc`, before registration** | **3/3 exit 0**, registration/metadata/cleanup |
| New explicit NIC, 3 repetitions | **3/3 exit 0** | **3/3 exit 0** |
| New isolated mapping probe | runtime count 1, topology count 0 | runtime count 1, topology count 1 |

The new baseline's observed crash signature differs from the historical SIGSEGV. The first retest expected exit 139 and therefore stopped on exit 134; that failed attempt is preserved. The follow-up added the direct topology-count diagnostic and froze a three-repeat matrix accepting either observed abort/segfault failure before registration for the broken baseline. It did not turn the original failed attempt into a pass. The evidence demonstrates the empty topology and the fixed path; there is no native stack-trace claim for the new abort.

In the historical auxiliary two-node run, the topology build completed 9/9 host-memory writes (1 KiB, 1 MiB, 16 MiB, each with three changing payloads). That is a separate **historical** finite correctness result. The accompanying staged-GPU attempt timed out; it is not a GPU-direct success. The new retest does not rerun those transfers.

## Evidence and scope

Historical and new logs are separated under `logs/`. Internal IPs in metadata are replaced with `<NODE_IP>`; excerpts are explicitly labeled. `manifest.json` records original and distributed file hashes and each transformation. Complete unmodified original archives remain private and were independently read back with every member SHA checked (33 files for the initial retest, 55 for the follow-up).

The owned component lock was acquired on its original inode and released. The retained K3 service PID/starttime and nonce matched before/after; zero K3 requests and zero service restarts were issued. This is topology/host-registration validation. It does not qualify GPU-direct RDMA, NVIDIA/AMD IPC, graph replay, NIXL integration or model TPOT/throughput.

Frozen probe sources and logs use a `.txt` suffix to preserve their original bytes independently of source formatting tools. The manifest records the distribution filenames and original SHA256 values.
