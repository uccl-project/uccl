import os
import subprocess
import setuptools
from glob import glob
import torch

from pathlib import Path

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PROJECT_ROOT = Path(os.path.dirname(__file__)).resolve()

if __name__ == "__main__":

    cxx_flags = [
        "-O3",
        "-Wno-deprecated-declarations",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
        "-Wno-reorder",
        "-Wno-attributes",
        "-Wno-unused-result",
        "-Wno-unused-function",
    ]
    nvcc_flags = ["-O3", "-Xcompiler", "-O3"]
    sources = glob("./src/*.cu") + glob("./src/*.cpp") + glob("./src/*.cc")
    libraries = ["ibverbs", "glog"]
    include_dirs = [PROJECT_ROOT / "include", PROJECT_ROOT / ".." / "include"]
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = []

    if torch.version.cuda:
        if int(os.getenv("DISABLE_SM90_FEATURES", 0)):
            # Prefer A100
            os.environ["TORCH_CUDA_ARCH_LIST"] = os.getenv(
                "TORCH_CUDA_ARCH_LIST", "8.0"
            )

            # Disable some SM90 features: FP8, launch methods, and TMA
            cxx_flags.append("-DDISABLE_SM90_FEATURES")
            nvcc_flags.append("-DDISABLE_SM90_FEATURES")

        else:
            # Prefer H800 series
            os.environ["TORCH_CUDA_ARCH_LIST"] = os.getenv(
                "TORCH_CUDA_ARCH_LIST", "9.0"
            )

            # CUDA 12 flags
            nvcc_flags.extend(
                ["-rdc=true", "--ptxas-options=--register-usage-level=10"]
            )

        device_arch = os.environ["TORCH_CUDA_ARCH_LIST"]
    else:
        # Disable SM90 features on AMD
        cxx_flags.append("-DDISABLE_SM90_FEATURES")
        nvcc_flags.append("-DDISABLE_SM90_FEATURES")

        cxx_flags.append("-DDISABLE_AGGRESSIVE_ATOMIC")
        nvcc_flags.append("-DDISABLE_AGGRESSIVE_ATOMIC")

        device_arch = os.getenv("TORCH_CUDA_ARCH_LIST", "gfx942")
        os.environ["PYTORCH_ROCM_ARCH"] = device_arch

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if device_arch.strip() != "9.0":
        assert int(os.getenv("DISABLE_AGGRESSIVE_PTX_INSTRS", 1)) == 1
        os.environ["DISABLE_AGGRESSIVE_PTX_INSTRS"] = "1"

    # Disable aggressive PTX instructions
    if int(os.getenv("DISABLE_AGGRESSIVE_PTX_INSTRS", "1")):
        cxx_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")
        nvcc_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")

    # Put them together
    extra_compile_args = {
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args["nvcc_dlink"] = nvcc_dlink

    # Summary
    print("Build summary:")
    print(f" > Sources: {sources}")
    print(f" > Includes: {include_dirs}")
    print(f" > Libraries: {library_dirs}")
    print(f" > Compilation flags: {extra_compile_args}")
    print(f" > Link flags: {extra_link_args}")
    print(f" > Arch list: {device_arch}")
    print(f' > Platform: {"ROCm" if torch.version.hip else "CUDA"}')
    print()

    # noinspection PyBroadException
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        revision = "+" + subprocess.check_output(cmd).decode("ascii").rstrip()
    except Exception as _:
        revision = ""

    setuptools.setup(
        name="ep",
        version="0.0.1" + revision,
        ext_modules=[
            CUDAExtension(
                name="ep",
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                sources=sources,
                libraries=libraries,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
