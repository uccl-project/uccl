# AGENTS.md

## Project Identity

**UCCL-Lite** is an efficient communication library for low-end, consumer-level
GPUs.  It targets highly heterogeneous environments where NVLink and GPUDirect
RDMA are unavailable — think GeForce RTX 4090/5090, or Unified Memory devices
like GB10.

The codebase is derived from **MSCCLPP**.  The primary restructuring goal is to
split code by responsibility into clearer layers for independent building,
reading, and future evolution — not to rewrite functional semantics.

When working in this repository, assume:

- Upstream semantics and existing interface compatibility matter more than
  locally "cleaner" renames.
- Directory reorganization is the main change surface; confirm necessity before
  moving code across directories.
- `MSCCLPP_*` macros, the `mscclpp` namespace, `nccl.h`-style interfaces, and
  header conventions exist for upstream and caller compatibility — do not
  casually unify them into new names.

## Target Scenario

- Consumer-grade GPUs without GPUDirect RDMA, possibly without direct PCIe P2P.
- Cover GPUs like RTX 4090/5090 and Unified Memory devices like GB10.
- All inter-node data movement must stage through host memory.
- The entire send/recv data path is **SM-free** — no GPU kernels are launched
  for data movement.  See `doc/p2p-design.md` for full architecture details.

## Short-Term Goals

- Implement all inter-node collective communication primitives on top of the
  current MSCCLPP architecture, using host-memory staging.

## Directory Structure

- `core/`
  - Fundamental communication and runtime facilities.
  - Contains communicator, connection, context, IB/NUMA, socket, proxy,
    memory channel, GPU IPC, logging, error handling, etc.
  - Build artifacts: `build/libmint.so`, `build/libmint.a`.
- `collective/`
  - Collective algorithms and execution layer built on top of `core/`.
  - Contains algorithm selector, execution plan, executor, and various
    allgather/allreduce CUDA implementations.
  - Build artifacts: `build/libmint_collective.so`, `build/libmint_collective.a`.
- `nccl/`
  - NCCL compatibility / adaptation layer.
  - Built via its own `nccl/Makefile`, which depends on `libmint` /
    `libmint_collective` from the top-level build.
  - Build artifacts: `nccl/build/`.
- `doc/`
  - Design documents and architecture notes.
- `thirdparty/`
  - Vendored dependencies (currently `nlohmann/json.hpp`).
- `build/`
  - Local build output.  Treat as generated — never edit by hand.

## Building

Top-level `Makefile` covers `core/` and `collective/`:

```bash
make
make core
make collective
make clean
```

`nccl/Makefile` covers the NCCL compatibility layer and triggers top-level
dependency builds automatically:

```bash
make -C nccl
make -C nccl clean
```

Default toolchain and conventions:

- `CXX ?= g++`
- `NVCC ?= nvcc`
- `CUDA_HOME ?= /usr/local/cuda`
- `CUDA_ARCH ?= 80 90 100`
- Code compiles as **C++17**

If you change `.cu` files, exported symbols, header include paths, or library
dependencies, at least re-run the corresponding `make` target.

## Modification Guidelines

- When modifying `core/`: check impact on `collective/` and `nccl/` include
  paths, symbol visibility, public types, and error codes.
- When modifying `collective/`: keep algorithm implementations and execution
  framework cleanly separated — do not push new low-level facilities into
  algorithm files.
- When modifying `nccl/`: treat it as a compatibility layer, not an
  independent runtime.
- When adding files, place them in the semantically closest directory:
  - Low-level transport, device abstraction, environment, public runtime
    utilities -> `core/`
  - Collective algorithms, selector, plan, executor -> `collective/`
  - NCCL API wrappers, type mappings, compatibility shims -> `nccl/`
- Do not modify vendored code under `thirdparty/` unless strictly necessary.
- Do not hand-edit anything under `build/`.

## Upstream Compatibility Notes

- When you see `MSCCLPP_*` macros, the `mscclpp` namespace, or `nccl.h`-style
  interfaces, preserve them as-is.
- Even though library names have been reorganized to `libmint*`, do not
  blanket-replace upstream naming in source code with `mint`.
- If a change only "unifies naming" but widens the diff against upstream, it
  is generally not worth making.

## Validation

After any change, run at least the minimal validation matching the change
scope:

- Changed `core/`: `make core`
- Changed `collective/`: `make collective`
- Changed cross-layer interfaces or includes: `make`
- Changed `nccl/`: `make -C nccl`

If the build depends on local CUDA/IB tooling that is temporarily unavailable,
explicitly state unverified items in the commit message.

## External Libraries

- NCCL and nccl-tests are available at `/home/yangz/nfs/zhongjie/nccl` and
  `/home/yangz/nfs/zhongjie/nccl-tests`.

## Additional Conventions

- The repository is driven almost entirely by `.cc/.cpp/.cu/.hpp/.h` files and
  Makefiles.  Do not introduce a new build system unless explicitly required.
- Prefer small incremental refactors that keep the build green over large
  cross-directory migrations.
- If you find code with obvious upstream naming remnants, understand its
  compatibility purpose before deciding whether to touch it.

## Debugging

- When running test scripts, set `timeout` to at most **15 seconds**.
