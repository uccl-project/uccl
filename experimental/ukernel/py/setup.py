import os
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

try:
    import nanobind
except ImportError as exc:
    raise RuntimeError(
        "nanobind is required for ukernel_ccl Python bindings. "
        "Install it with: pip install nanobind"
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
UCCL_ROOT = ROOT.parent.parent
CUDA_HOME = Path("/usr/local/cuda")
ROCM_HOME = Path(os.environ.get("ROCM_HOME", "/opt/rocm"))
USE_ROCM = getattr(torch.version, "hip", None) is not None
BUILD_CCL_ON_ROCM = os.environ.get("UKERNEL_BUILD_CCL_ON_ROCM", "0") == "1"
RDMACM_SO = Path("/usr/lib/x86_64-linux-gnu/librdmacm.so.1")
GDRCOPY_INCLUDE_DIR = Path(
    os.environ.get("GDRCOPY_INCLUDE_DIR", "/usr/local/include")
)
GDRCOPY_LIBDIR = os.environ.get("GDRCOPY_LIBDIR", "").strip()
RDMA_STATIC = (
    UCCL_ROOT / "collective" / "rdma" / ("librdma_hip.a" if USE_ROCM else "librdma.a")
)
NANOBIND_ROOT = Path(nanobind.__file__).resolve().parent


def rel(path: Path) -> str:
    return str(path.resolve())


sources = [
    rel(ROOT / "py" / "ukernel_ccl.cpp"),
    rel(NANOBIND_ROOT / "src" / "nb_combined.cpp"),
    rel(ROOT / "src" / "ccl" / "backend" / "device_backend.cc"),
    rel(ROOT / "src" / "ccl" / "backend" / "transport_backend.cc"),
    rel(ROOT / "src" / "ccl" / "executor.cc"),
    rel(ROOT / "src" / "ccl" / "executor_real.cc"),
    rel(ROOT / "src" / "ccl" / "plan.cc"),
    rel(ROOT / "src" / "ccl" / "selector.cc"),
    rel(ROOT / "src" / "ccl" / "topology.cc"),
    rel(ROOT / "src" / "transport" / "communicator.cc"),
    rel(ROOT / "src" / "transport" / "memory" / "mr_manager.cc"),
    rel(ROOT / "src" / "transport" / "memory" / "ipc_manager.cc"),
    rel(ROOT / "src" / "transport" / "memory" / "shm_manager.cc"),
    rel(ROOT / "src" / "transport" / "oob" / "oob.cc"),
    rel(ROOT / "src" / "transport" / "oob" / "oob_shm.cc"),
    rel(ROOT / "src" / "transport" / "oob" / "oob_socket.cc"),
    rel(ROOT / "src" / "transport" / "adapter" / "tcp_adapter.cc"),
    rel(ROOT / "src" / "transport" / "adapter" / "uccl_adapter.cc"),
    rel(ROOT / "src" / "transport" / "adapter" / "ipc_adapter.cc"),
    rel(ROOT / "src" / "transport" / "util" / "utils.cc"),
    rel(ROOT / "src" / "device" / "fifo" / "c2d_fifo.cc"),
    rel(ROOT / "src" / "device" / "fifo" / "d2c_fifo.cpp"),
    rel(ROOT / "src" / "device" / "fifo" / "sm_fifo.cc"),
    rel(ROOT / "src" / "device" / "worker.cc"),
    rel(ROOT / "src" / "device" / "persistent_kernel_ops.cu"),
]

p2p_sources = [
    rel(ROOT / "py" / "ukernel_p2p.cpp"),
    rel(NANOBIND_ROOT / "src" / "nb_combined.cpp"),
    rel(ROOT / "src" / "transport" / "communicator.cc"),
    rel(ROOT / "src" / "transport" / "memory" / "mr_manager.cc"),
    rel(ROOT / "src" / "transport" / "memory" / "ipc_manager.cc"),
    rel(ROOT / "src" / "transport" / "memory" / "shm_manager.cc"),
    rel(ROOT / "src" / "transport" / "oob" / "oob.cc"),
    rel(ROOT / "src" / "transport" / "oob" / "oob_shm.cc"),
    rel(ROOT / "src" / "transport" / "oob" / "oob_socket.cc"),
    rel(ROOT / "src" / "transport" / "adapter" / "tcp_adapter.cc"),
    rel(ROOT / "src" / "transport" / "adapter" / "uccl_adapter.cc"),
    rel(ROOT / "src" / "transport" / "adapter" / "ipc_adapter.cc"),
    rel(ROOT / "src" / "transport" / "util" / "utils.cc"),
]

include_dirs = [
    rel(ROOT / "include"),
    rel(ROOT / "src" / "transport"),
    rel(ROOT / "src" / "device"),
    rel(ROOT / "src" / "device" / "fifo"),
    rel(ROOT / "src" / "ccl"),
    rel(NANOBIND_ROOT / "include"),
    rel(NANOBIND_ROOT / "ext" / "robin_map" / "include"),
    rel(UCCL_ROOT),
    rel(UCCL_ROOT / "collective" / "rdma"),
    rel(UCCL_ROOT / "include"),
    str(GDRCOPY_INCLUDE_DIR),
]
if USE_ROCM:
    include_dirs.append(str(ROCM_HOME / "include"))
else:
    include_dirs.append(str(CUDA_HOME / "include"))

library_dirs = ["/usr/local/lib", "/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"]
runtime_library_dirs = []
if USE_ROCM:
    library_dirs.append(str(ROCM_HOME / "lib"))
    runtime_library_dirs.append(str(ROCM_HOME / "lib"))
else:
    library_dirs.append(str(CUDA_HOME / "lib64"))
    runtime_library_dirs.append(str(CUDA_HOME / "lib64"))
if GDRCOPY_LIBDIR:
    library_dirs.append(GDRCOPY_LIBDIR)
    runtime_library_dirs.append(str(Path(GDRCOPY_LIBDIR).resolve()))

common_cxx_args = [
    "-O3",
    "-std=c++17",
    "-Wall",
    "-Wno-unused-function",
    "-Wno-sign-compare",
    "-Wno-reorder",
    "-Wno-unused-variable",
    "-Wno-unused-label",
    "-Wno-unused-but-set-variable",
    "-Wno-stringop-overread",
    "-Wno-narrowing",
    "-pthread",
    "-fPIC",
    "-DUKERNEL_ENABLE_TMA=0",
]
if USE_ROCM:
    common_cxx_args.extend(["-D__HIP_PLATFORM_AMD__", "-DUSE_ROCM=1", "-DHIPBLAS_V2"])

cuda_nvcc_args = [
    "-O3",
    "-std=c++20",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-DKITTENS_HOPPER",
    "-DUKERNEL_ENABLE_TMA=0",
    "-gencode",
    "arch=compute_80,code=sm_80",
    "-gencode",
    "arch=compute_86,code=sm_86",
    "-gencode",
    "arch=compute_89,code=sm_89",
]

ExtensionCls = CppExtension if USE_ROCM else CUDAExtension

common_libraries = [
    "gflags",
    "z",
    "ibverbs",
    "nl-3",
    "nl-route-3",
    "pthread",
    "numa",
]
if USE_ROCM:
    common_libraries.extend(["amdhip64", "elf", "dl"])
else:
    common_libraries.append("rdmacm")
    common_libraries.extend(["cudart", "cuda", "gdrapi"])

extra_link_args = []
if USE_ROCM and RDMACM_SO.exists():
    extra_link_args.append(str(RDMACM_SO.resolve()))

ext = ExtensionCls(
    name="ukernel_ccl._C",
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args=common_cxx_args if USE_ROCM else {
        "cxx": common_cxx_args,
        "nvcc": cuda_nvcc_args,
    },
    library_dirs=library_dirs,
    libraries=common_libraries,
    extra_objects=[str(RDMA_STATIC.resolve())],
    extra_link_args=extra_link_args,
    runtime_library_dirs=runtime_library_dirs,
)

p2p_ext = ExtensionCls(
    name="ukernel_p2p._C",
    sources=p2p_sources,
    include_dirs=include_dirs,
    extra_compile_args=common_cxx_args if USE_ROCM else {
        "cxx": common_cxx_args,
        "nvcc": cuda_nvcc_args,
    },
    library_dirs=library_dirs,
    libraries=common_libraries,
    extra_objects=[str(RDMA_STATIC.resolve())],
    extra_link_args=extra_link_args,
    runtime_library_dirs=runtime_library_dirs,
)

ext_modules = [p2p_ext]
if not USE_ROCM or BUILD_CCL_ON_ROCM:
    ext_modules.insert(0, ext)


setup(
    name="ukernel-ccl",
    version="0.1.0",
    packages=["ukernel_ccl", "ukernel_p2p"],
    package_dir={"ukernel_ccl": "ukernel_ccl", "ukernel_p2p": "ukernel_p2p"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
