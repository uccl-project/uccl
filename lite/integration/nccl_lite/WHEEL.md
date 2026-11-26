# UCCL Lite NCCL wheel

This Linux x86-64 wheel contains a generic launcher, the UCCL Lite NCCL
proxy, and opt-in SGLang hooks. Installation is not tied to EP8. Its only
accelerated profile currently validated is single-node eight-L4 (SM 89),
BF16 EP8. It is not a complete `libnccl.so.2` replacement.

## Build

From the repository root, with CUDA 12.9 and NCCL headers available:

```bash
CUDA_HOME=/usr/local/cuda-12.9 \
NCCL_HOME="$PWD/lite/build/nccl-2.28.9" \
python -m pip wheel --no-deps ./lite/integration/nccl_lite -w ./lite/integration/nccl_lite/build/wheels
```

Building the wheel invokes the existing Makefile and bundles the resulting
`libuccl_lite_nccl_compat.so`. Installing the wheel does not compile CUDA code.
The tested 0.1.1 build is also checked into `wheels/` for direct installation.
The wheel needs CUDA 12 runtime and cuBLAS libraries on the destination host;
it does not bundle them or NCCL. The wheel has a Linux platform tag, not a
manylinux portability claim.

## Install and run

Install into the same Python environment that runs SGLang or PyTorch:

```bash
python -m pip install ./lite/integration/nccl_lite/build/wheels/uccl_lite_nccl-0.1.1-py3-none-linux_x86_64.whl
uccl-lite-nccl --real-nccl /absolute/path/to/libnccl.so.2 \
  -- python -m sglang.launch_server YOUR_SGLANG_ARGS
```

The default `--mode compatible` loads the proxy into PyTorch processes but
forwards their NCCL calls to the real library. SGLang's direct PyNccl path
loads the real library. It does not set EP-specific packet, CPU-affinity, or
SGLang hook options, so this is the mode to use for EP1/2/4/8 and other
configurations when acceleration has not been validated. It needs the
application's real NCCL library and does not provide a universal NCCL ABI.
It does not interpret `--ep-size`: if the application runs that EP
configuration with its own NCCL, compatible mode uses the same NCCL path.

On the validated single-node eight-L4 profile, enable the packet path:

```bash
uccl-lite-nccl --mode packet --profile l4-ep8 \
  --real-nccl /absolute/path/to/libnccl.so.2 \
  -- python -m sglang.launch_server YOUR_SGLANG_ARGS
```

For the validated opt-in AG+gate and row-ready RS combination, use
`--mode sglang-ag-ready-rs --profile l4-ep8`. `--mode nccl` executes the
command unchanged, which is useful as a control on any host.
Accelerated modes require an explicit profile so installing the wheel cannot
silently enable the L4 kernels elsewhere. The launcher also checks that
eight NVIDIA L4 GPUs are visible; the profile selection asserts that the
rest of the single-node topology and software match the tested setup.
The `--` separates launcher options from the application command. Other
modes are `sglang-ag-gate`,
`sglang-ready-combine`, `sglang-ready-rs`, and `sglang-rs-sum2`.

`--real-nccl` should point to the exact NCCL binary used by the application's
PyTorch ProcessGroup. If omitted, the launcher uses
`UCCL_LITE_REAL_NCCL`, then the installed `nvidia-nccl-cu12` Python wheel.
In compatible mode, the launcher preloads the proxy but disables its UCCL
path and points SGLang PyNccl at the real NCCL library. In accelerated modes,
it sets `LD_PRELOAD`, `SGLANG_NCCL_SO_PATH`, packet settings, and, for SGLang
modes, `PYTHONPATH` to the bundled hooks. It changes no installed SGLang
source or model weights. Existing settings are inherited by
the launched process, except settings chosen explicitly by the mode scripts.

The proxy accelerates eligible eight-rank BF16 AllGather, ReduceScatter SUM,
and AllReduce SUM calls. Other supported NCCL calls forward to the real
library. Calls during CUDA Graph capture, multi-node routing, and arbitrary
concurrent streams are outside the validated boundary. SGLang hooks are
version-pinned to the installed GLM-4.5-Air-FP8 path and should be used only
after matching the numerical and E2E checks in `fusion_ag_rs_8l4_20261112.md`.

## Compatibility checks

On the eight-L4 test host, the installed wheel's default compatible mode
passed PyTorch BF16 and FP32 AG/RS/AR checks with 2, 4, and 8 processes.
GLM-4.5-Air-FP8 SGLang servers with EP2 and EP4 reached readiness and each
completed an eight-token generation. These checks do not prove every model,
EP size, NCCL version, or hardware configuration. The host needed its usual
NCCL network settings (`NCCL_NET=Socket`, `NCCL_NET_PLUGIN=none`) and the
SGLang environment's `bin` directory on `PATH` for `ninja`.
