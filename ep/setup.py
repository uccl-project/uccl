import os
import subprocess
import setuptools
from glob import glob
import torch
import shutil
import site

from pathlib import Path

from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools.command.install import install

PROJECT_ROOT = Path(os.path.dirname(__file__)).resolve()


class CustomInstall(install):
    """Custom install command that installs .so file to INSTALL_DIR"""

    def run(self):
        # Run the standard build first
        self.run_command("build_ext")

        # Get the install directory
        python_site_packages = site.getsitepackages()[0]
        install_dir = os.getenv(
            "INSTALL_DIR", os.path.join(python_site_packages, "uccl")
        )
        os.makedirs(install_dir, exist_ok=True)

        # Find the built .so file
        build_lib = self.get_finalized_command("build_ext").build_lib
        so_files = list(Path(build_lib).glob("ep*.so"))

        if not so_files:
            raise RuntimeError(f"Could not find built .so file in {build_lib}")

        so_file = so_files[0]
        dest_path = os.path.join(install_dir, so_file.name)

        # Copy the .so file to the install directory
        print(f"Installing {so_file.name} to {install_dir}")
        shutil.copy2(so_file, dest_path)
        print(f"Installation complete. Module installed as: {dest_path}")


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
    libraries = ["ibverbs", "glog", "nl-3", "nl-route-3", "numa"]
    include_dirs = [PROJECT_ROOT / "include", PROJECT_ROOT / ".." / "include"]
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = []

    if torch.version.cuda:
        # EFA (Elastic Fabric Adapter) Detection
        efa_home = os.getenv("EFA_HOME", "/opt/amazon/efa")
        has_efa = os.path.exists(efa_home)
        if has_efa:
            print("EFA detected, building with EFA support")
        else:
            print("EFA not detected, building without EFA")

        # Architecture Detection
        arch = os.uname().machine
        cpu_is_arm64 = arch == "aarch64"

        # GPU Detection
        gpu_name = ""
        gpu_is_hopper = False
        try:
            gpu_query = (
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("ascii")
                .strip()
                .split("\n")[0]
            )
            gpu_name = gpu_query
            gpu_is_hopper = "GH200" in gpu_name
        except Exception:
            print("Warning: Could not detect GPU name via nvidia-smi")

        # GH200 (Grace Hopper) Detection
        has_gh200 = cpu_is_arm64 and gpu_is_hopper
        if has_gh200:
            print(
                f"GH200 detected (GPU: {gpu_name}, CPU: {arch}), building with GH200 support"
            )
        else:
            print("GH200 not detected, building without GH200 support")

        # Add EFA flags if detected
        if has_efa:
            cxx_flags.append("-DEFA")
            nvcc_flags.append("-DEFA")
            include_dirs.append(Path(efa_home) / "include")
            library_dirs.append(Path(efa_home) / "lib")
            libraries.append("efa")

        # Add GH200 flags if detected
        if has_gh200:
            cxx_flags.append("-DUSE_GRACE_HOPPER")
            nvcc_flags.append("-DUSE_GRACE_HOPPER")

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

    # Convert Path objects to strings for include_dirs and library_dirs
    include_dirs = [str(d) for d in include_dirs]
    library_dirs = [str(d) for d in library_dirs]

    # Summary
    print("\n" + "=" * 60)
    print("Build Summary")
    print("=" * 60)
    print(f" > Platform: {'ROCm' if torch.version.hip else 'CUDA'}")
    if torch.version.cuda:
        print(f" > Architecture: {arch}")
        if gpu_name:
            print(f" > GPU: {gpu_name}")
        print(f" > EFA Support: {'Yes' if has_efa else 'No'}")
        print(f" > GH200 Support: {'Yes' if has_gh200 else 'No'}")
    print(f" > Device Arch: {device_arch}")
    print(f" > Sources: {len(sources)} files")
    print(f" > Include Dirs: {include_dirs}")
    print(f" > Library Dirs: {library_dirs}")
    print(f" > Libraries: {libraries}")
    print(f" > CXX Flags: {cxx_flags}")
    print(f" > NVCC Flags: {nvcc_flags}")
    print(f" > Link Flags: {extra_link_args}")
    print("=" * 60 + "\n")

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
        cmdclass={
            "build_ext": BuildExtension,
            "install": CustomInstall,
        },
    )
