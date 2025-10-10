import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PROJECT_ROOT = Path(os.path.dirname(__file__)).resolve()

# -------- Supported GPU ARCHS --------
SUPPORTED_GPU_ARCHS = ["gfx942", "gfx950"]


def get_offload_archs():
    def _get_device_arch():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found!")
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0].lower()

    gpu_archs = os.environ.get("GPU_ARCHS", None)

    arch_list = []
    if gpu_archs is None or gpu_archs.strip() == "":
        arch_list = [_get_device_arch()]
    else:
        for arch in gpu_archs.split(";"):
            arch = arch.strip().lower()
            if arch == "native":
                arch = _get_device_arch()
            if arch not in arch_list:
                arch_list.append(arch)

    macro_arch_list = []
    offload_arch_list = []
    for arch in arch_list:
        if arch in SUPPORTED_GPU_ARCHS:
            offload_arch_list.append(f"--offload-arch={arch}")
        else:
            print(f"[WARNING] Ignoring unsupported GPU_ARCHS entry: {arch}")
    return offload_arch_list, macro_arch_list


def get_common_flags():
    arch = platform.machine().lower()
    extra_link_args = [
        "-Wl,-rpath,/opt/rocm/lib",
        f"-L/usr/lib/{arch}-linux-gnu",
    ]

    cxx_flags = [
        "-O3",
        "-fvisibility=hidden",
        "-std=c++20",
        "-DDISABLE_SM90_FEATURES",
    ]

    nvcc_flags = [
        "-O3",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT16_OPERATORS__",
        "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT162_OPERATORS__",
        "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
        "-fno-offload-uniform-block",
        "-mllvm",
        "--lsr-drop-solution=1",
        "-mllvm",
        "-enable-post-misched=0",
        "-mllvm",
        "-amdgpu-coerce-illegal-types=1",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
        "-mllvm",
        "-amdgpu-function-calls=false",
        "-std=c++20",
        "-DDISABLE_SM90_FEATURES",
    ]

    # Device Archs
    offload_arch_list, macro_arch_list = get_offload_archs()
    cxx_flags += macro_arch_list
    nvcc_flags += macro_arch_list
    nvcc_flags += offload_arch_list

    # Max Jobs
    max_jobs = int(os.getenv("MAX_JOBS", "64"))
    nvcc_flags.append(f"-parallel-jobs={max_jobs}")

    return {
        "extra_link_args": extra_link_args,
        "extra_compile_args": {
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    }


def all_files_in_dir(path, name_extensions=None):
    all_files = []
    for dirname, _, names in os.walk(path):
        for name in names:
            suffix = Path(name).suffix.lstrip(".")
            if name_extensions and suffix not in name_extensions:
                continue
            all_files.append(Path(dirname, name))
    return all_files


def build_torch_extension():

    # Link and Compile flags
    extra_flags = get_common_flags()

    # CPP
    ep_src_dir = Path(PROJECT_ROOT / "src")
    # sources = [ep_src_dir / "ep_runtime.cu",
    #            ep_src_dir / "common.cpp",
    #            ep_src_dir / "layout.cu",
    #            ep_src_dir / "intranode.cu",
    #            ep_src_dir / "uccl_ep.cc",
    #            ep_src_dir / "proxy.cpp",
    #            ep_src_dir / "uccl_bench.cpp",
    #            ep_src_dir / "peer_copy_manager.cpp",
    #            ep_src_dir / "peer_copy_worker.cpp",
    #            ep_src_dir / "uccl_proxy.cpp",
    #            ep_src_dir / "py_cuda_shims.cu",
    #            ep_src_dir / "internode_ll.cu"]

    sources = all_files_in_dir(ep_src_dir, ["cpp", "cu", "cc"])
    return CUDAExtension(
        name="ep_cpp",
        sources=sources,
        libraries=["ibverbs", "glog"],
        library_dirs=["/usr/lib/x86_64-linux-gnu/"],
        include_dirs=[
            Path(PROJECT_ROOT / "include"),
            Path(PROJECT_ROOT / ".." / "include",
                 "/usr/include",
                 PROJECT_ROOT / ".." / "include"),
        ],
        **extra_flags,
    )


if __name__ == "__main__":

    torch_ext = build_torch_extension()
    setup(
        name="ep_cpp",
        version="0.0.0",
        package_data={"ep": ["lib/*.so"]},
        ext_modules=[build_torch_extension()],
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    )
