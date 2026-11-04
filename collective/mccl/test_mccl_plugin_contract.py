#!/usr/bin/env python3
"""Host-only checks for the MUSA MCCL plugin boundary."""

import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "collective" / "rdma" / "mccl_plugin_musa.cc"
TRANSPORT = ROOT / "collective" / "rdma" / "transport.cc"
MAKEFILE = ROOT / "collective" / "rdma" / "Makefile.musa"
RDMA_MAKEFILE = ROOT / "collective" / "rdma" / "Makefile"


def main():
    source = PLUGIN.read_text()
    transport = TRANSPORT.read_text()
    makefile = MAKEFILE.read_text()

    required_source_tokens = (
        "mcclNetPlugin_v11",
        "MCCL_PTR_MUSA",
        "mcclPluginIflush",
        "can_register_musa",
    )
    for token in required_source_tokens:
        assert token in source, token
    assert "gpuMemcpyHostToDevice" in transport
    assert "ncclNetPlugin_v8" not in source
    assert "NCCL_PTR_CUDA" not in source
    assert "libmccl-net-uccl.so" in makefile
    assert "-DUCCL_USE_MUSA" in makefile
    # MCCL_PTR_DMABUF must not be advertised from the NIC dma_buf_support
    # attribute alone; the MUSA dmabuf export path is not verified, so the
    # capability stays closed (manual §7.5).
    assert "MCCL_PTR_DMABUF" not in source
    assert "dma_buf_support" not in source

    musa_build = subprocess.run(
        (
            "make",
            "-f",
            "Makefile.musa",
            "-n",
            "all",
            "MUSA_HOME=/missing/musa",
        ),
        check=True,
        capture_output=True,
        text=True,
        cwd=MAKEFILE.parent,
    )
    assert "mccl_plugin_musa.cc" in musa_build.stdout

    cuda_build = subprocess.run(
        ("make", "-n", "-f", str(RDMA_MAKEFILE), "build"),
        check=True,
        capture_output=True,
        text=True,
        cwd=RDMA_MAKEFILE.parent,
    )
    assert "mccl_plugin_musa" not in cuda_build.stdout


if __name__ == "__main__":
    main()
