"""Build the standalone UCCL-GIN Python extension (_uccl_gin).

Mirrors ep/setup.py's approach: a torch CUDAExtension that auto-hipifies the
sources on ROCm (torch.version.hip) so the copied transport substrate — including
the mscclpp FIFO layer that is full of raw cuda* calls — compiles on AMD without a
hand-written hipify step.

Scope on ROCm (RoCE, non-EFA): only put + quiet are functional; the inter-node
atomic primitives are EFA-shaped and not supported yet. See AMD_SUPPORT_PLAN.md.

Usage:
    # NVIDIA: prefer the Makefile (also builds the standalone microbench exe).
    # AMD/ROCm:
    python setup.py build_ext --inplace      # produces python/uccl_gin/_uccl_gin*.so
"""

import os
import subprocess
import sysconfig
from glob import glob
from pathlib import Path

import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PROJECT_ROOT = Path(os.path.dirname(__file__)).resolve()
IS_ROCM = torch.version.hip is not None

NUM_PROXY_THS = int(os.getenv("UCCL_GIN_NUM_PROXY_THS", "4"))


def _mpi_flags():
    """Return (include_dirs, library_dirs, libraries) for MPI.

    Prefer an explicit UCCL_GIN_MPI_HOME (a self-contained MPI install with
    headers + libs, e.g. /opt/cluster-test on boxes lacking libopenmpi-dev),
    then fall back to `mpicxx --showme`. Validate that mpi.h actually exists at
    the resolved include dir, since some boxes ship the mpicxx wrapper but not
    the dev headers.
    """
    mpi_home = os.getenv("UCCL_GIN_MPI_HOME")
    if mpi_home:
        inc = str(Path(mpi_home) / "include")
        libdir = str(Path(mpi_home) / "lib")
        return [inc], [libdir], ["mpi"]

    mpicxx = os.getenv("MPICXX", "mpicxx")
    inc, libdirs, libs = [], [], []
    try:
        comp = subprocess.check_output([mpicxx, "--showme:compile"]).decode().split()
        link = subprocess.check_output([mpicxx, "--showme:link"]).decode().split()
    except Exception:
        return inc, libdirs, libs
    for tok in comp:
        if tok.startswith("-I"):
            inc.append(tok[2:])
    for tok in link:
        if tok.startswith("-L"):
            libdirs.append(tok[2:])
        elif tok.startswith("-l"):
            libs.append(tok[2:])
    if inc and not any(Path(d, "mpi.h").exists() for d in inc):
        raise RuntimeError(
            "mpicxx reports include dirs %s but none contain mpi.h. Install MPI "
            "dev headers or set UCCL_GIN_MPI_HOME to a self-contained MPI." % inc)
    return inc, libdirs, libs


def main():
    # NOTE: tests/microbench.cu is the NVIDIA Makefile executable (NCCL-coupled),
    # NOT part of this extension. Only the put/quiet smoke kernel is.
    sources = (
        glob("transport/*.cpp")
        + glob("transport/*.cc")
        + ["context.cpp", "bindings.cpp", "tests/put_quiet_smoke.cu"]
    )

    mpi_inc, mpi_libdirs, mpi_libs = _mpi_flags()
    python_inc = sysconfig.get_paths()["include"]

    include_dirs = [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "transport"),
        str(PROJECT_ROOT / ".." / ".." / "include"),
        python_inc,
    ] + mpi_inc

    library_dirs = list(mpi_libdirs)
    libraries = ["ibverbs", "nl-3", "nl-route-3", "numa"] + mpi_libs

    common_defs = [
        f"-DUCCL_NUM_PROXY_THS={NUM_PROXY_THS}",
    ]

    cxx_flags = ["-O3", "-std=c++17", "-fvisibility=hidden",
                 "-Wno-unused-result", "-Wno-unused-variable"] + common_defs
    nvcc_flags = ["-O3", "-std=c++17"] + common_defs

    if IS_ROCM:
        rocm_home = os.getenv("ROCM_HOME", os.getenv("ROCM_PATH", "/opt/rocm"))
        gfx = os.getenv("UCCL_GIN_GFX", "gfx942")
        include_dirs.append(str(Path(rocm_home) / "include"))
        library_dirs.append(str(Path(rocm_home) / "lib"))
        libraries.append("amdhip64")
        # NCCL-GIN reference path off; non-EFA RC path (no -DEFA / SOFTWARE_ORDERING).
        rocm_defs = ["-DUCCL_GIN_WITH_NCCL_GIN=0", "-D__HIP_PLATFORM_AMD__"]
        cxx_flags += rocm_defs
        nvcc_flags += rocm_defs + [f"--offload-arch={gfx}"]
    else:
        cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
        efa_home = os.getenv("EFA_HOME", "/opt/amazon/efa")
        include_dirs.append(str(Path(cuda_home) / "include"))
        library_dirs.append(str(Path(cuda_home) / "lib64"))
        libraries += ["cudart", "cuda"]
        defs = ["-DEFA", "-DUCCL_GIN_WITH_NCCL_GIN=0"]
        if Path(efa_home).exists():
            include_dirs.append(str(Path(efa_home) / "include"))
            library_dirs.append(str(Path(efa_home) / "lib"))
            libraries.append("efa")
        cxx_flags += defs
        nvcc_flags += defs

    header_files = []
    for inc in include_dirs:
        header_files += glob(str(Path(inc) / "**" / "*.h"), recursive=True)
        header_files += glob(str(Path(inc) / "**" / "*.hpp"), recursive=True)
        header_files += glob(str(Path(inc) / "**" / "*.cuh"), recursive=True)

    print("=" * 60)
    print(f"UCCL-GIN extension build — Platform: {'ROCm' if IS_ROCM else 'CUDA'}")
    print(f"  sources: {len(sources)}  libs: {libraries}")
    print("=" * 60)

    setuptools.setup(
        name="uccl_gin",
        version="0.0.1",
        ext_modules=[
            CUDAExtension(
                name="uccl_gin._uccl_gin",
                sources=sources,
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=libraries,
                extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
                depends=header_files,
            )
        ],
        packages=["uccl_gin"],
        package_dir={"uccl_gin": "python/uccl_gin"},
        cmdclass={"build_ext": BuildExtension},
    )


if __name__ == "__main__":
    main()
