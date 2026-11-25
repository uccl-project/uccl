#!/usr/bin/env python3
"""Host-only contract test for the frozen MCCL v11 plugin ABI."""

import pathlib
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
MCCL_DIR = ROOT / "collective" / "mccl"

FAKE_PLUGIN = r"""
#include "mccl_net_abi.h"

static mcclResult_t init(void**, uint64_t, mcclNetCommConfig_v11_t*,
                         mcclDebugLogger_t, mcclProfilerCallback_t) {
  return mcclSuccess;
}
static mcclResult_t devices(int*) { return mcclSuccess; }
static mcclResult_t properties(int, mcclNetProperties_v11_t*) {
  return mcclSuccess;
}
static mcclResult_t listen_fn(void*, int, void*, void**) { return mcclSuccess; }
static mcclResult_t connect_fn(void*, int, void*, void**,
                               mcclNetDeviceHandle_v11_t**) {
  return mcclSuccess;
}
static mcclResult_t accept_fn(void*, void**,
                              mcclNetDeviceHandle_v11_t**) {
  return mcclSuccess;
}
static mcclResult_t reg_mr(void*, void*, size_t, int, void**) {
  return mcclSuccess;
}
static mcclResult_t reg_mr_dmabuf(void*, void*, size_t, int, uint64_t, int,
                                  void**) {
  return mcclSuccess;
}
static mcclResult_t dereg_mr(void*, void*) { return mcclSuccess; }
static mcclResult_t isend(void*, void*, size_t, int, void*, void*, void**) {
  return mcclSuccess;
}
static mcclResult_t irecv(void*, int, void**, size_t*, int*, void**, void**,
                          void**) {
  return mcclSuccess;
}
static mcclResult_t iflush(void*, int, void**, int*, void**, void**) {
  return mcclSuccess;
}
static mcclResult_t test(void*, int*, int*) { return mcclSuccess; }
static mcclResult_t close_comm(void*) { return mcclSuccess; }

extern "C" const mcclNet_v11_t mcclNetPlugin_v11 = {
    "fake",
    init,
    devices,
    properties,
    listen_fn,
    connect_fn,
    accept_fn,
    reg_mr,
    reg_mr_dmabuf,
    dereg_mr,
    isend,
    irecv,
    iflush,
    test,
    close_comm,
    close_comm,
    close_comm,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};
"""


def run(*args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, **kwargs)


def main():
    with tempfile.TemporaryDirectory(prefix="uccl-mccl-abi-") as tmp:
        tmpdir = pathlib.Path(tmp)
        plugin_src = tmpdir / "fake_plugin.cc"
        plugin_so = tmpdir / "libfake_mccl.so"
        probe = tmpdir / "mccl_abi_probe"
        plugin_src.write_text(FAKE_PLUGIN)

        run(
            "c++",
            "-std=c++17",
            "-fPIC",
            "-shared",
            "-I",
            str(MCCL_DIR),
            str(plugin_src),
            "-o",
            str(plugin_so),
        )
        run(
            "c++",
            "-std=c++17",
            "-I",
            str(MCCL_DIR),
            str(MCCL_DIR / "mccl_plugin_probe.cc"),
            str(MCCL_DIR / "mccl_abi_probe.cc"),
            "-ldl",
            "-o",
            str(probe),
        )
        result = run(str(probe), str(plugin_so))
        assert "ABI v11 is loadable" in result.stdout

        missing_symbol = tmpdir / "libmissing_symbol.so"
        run("c++", "-x", "c++", "-fPIC", "-shared", "-", "-o", str(missing_symbol),
            input="extern \"C\" int not_mccl;\n")
        rejected = subprocess.run(
            (str(probe), str(missing_symbol)), capture_output=True, text=True
        )
        assert rejected.returncode == 1
        assert "probe failed" in rejected.stderr


if __name__ == "__main__":
    main()
