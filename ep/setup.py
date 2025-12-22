import os
import subprocess
import setuptools
from glob import glob
import shutil
import site
from pathlib import Path

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools.command.install import install

PROJECT_ROOT = Path(os.path.dirname(__file__)).resolve()


class CustomInstall(install):
    """Custom install command that installs .so file to INSTALL_DIR"""

    def run(self):
        super().run()
        # Get the install directory for the .so file
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
        # Add CUDA library directory to library_dirs
        cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
        library_dirs.append(str(Path(cuda_home) / "lib64"))

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
        detected_compute_cap = None
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

            # Auto-detect compute capability
            compute_cap_query = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=compute_cap",
                        "--format=csv,noheader",
                    ],
                    stderr=subprocess.DEVNULL,
                )
                .decode("ascii")
                .strip()
                .split("\n")[0]
            )
            detected_compute_cap = compute_cap_query.strip()
            print(f"Detected GPU compute capability: {detected_compute_cap}")
        except Exception as e:
            print(f"Warning: Could not detect GPU info via nvidia-smi: {e}")

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

        # Use auto-detected compute capability if available
        if detected_compute_cap:
            default_arch = detected_compute_cap
        else:
            # Fallback to 9.0 if detection failed
            default_arch = "9.0"

        if int(os.getenv("DISABLE_SM90_FEATURES", 0)):
            # Force A100 architecture
            default_arch = "8.0"
            # Disable some SM90 features: FP8, launch methods, and TMA
            cxx_flags.append("-DDISABLE_SM90_FEATURES")
            nvcc_flags.append("-DDISABLE_SM90_FEATURES")
        else:
            # For SM90 and above, add register usage optimization
            if float(default_arch) >= 9.0:
                nvcc_flags.extend(["--ptxas-options=--register-usage-level=10"])

        # Set architecture environment variable before creating CUDAExtension
        device_arch = os.getenv("TORCH_CUDA_ARCH_LIST", default_arch)
        os.environ["TORCH_CUDA_ARCH_LIST"] = device_arch
    else:
        device_arch = os.getenv("TORCH_CUDA_ARCH_LIST", "gfx942")

        for arch in device_arch.split(","):
            nvcc_flags.append(f"--offload-arch={arch.lower()}")

        # Disable SM90 features on AMD
        cxx_flags.append("-DDISABLE_SM90_FEATURES")
        nvcc_flags.append("-DDISABLE_SM90_FEATURES")

        if int(os.getenv("DISABLE_AGGRESSIVE_ATOMIC", 0)):
            cxx_flags.append("-DDISABLE_AGGRESSIVE_ATOMIC")
            nvcc_flags.append("-DDISABLE_AGGRESSIVE_ATOMIC")

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    # Only enable aggressive PTX instructions for SM 9.0+ (H100/H800/B200)
    try:
        arch_version = float(device_arch.strip())
        if arch_version < 9.0:
            os.environ["DISABLE_AGGRESSIVE_PTX_INSTRS"] = "1"
        else:
            # Enable aggressive PTX instructions for SM 9.0+
            os.environ.setdefault("DISABLE_AGGRESSIVE_PTX_INSTRS", "0")
    except (ValueError, AttributeError):
        os.environ.setdefault("DISABLE_AGGRESSIVE_PTX_INSTRS", "1")

    # Apply aggressive PTX instruction flag
    if int(os.getenv("DISABLE_AGGRESSIVE_PTX_INSTRS", "0")):
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
        name="uccl_ep",
        version="0.0.1" + revision,
        packages=["uccl_ep"],
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
