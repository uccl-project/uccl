from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
GDRCOPY_ROOT = UCCL_ROOT / "thirdparty" / "gdrcopy"
RDMA_STATIC = UCCL_ROOT / "collective" / "rdma" / "librdma.a"
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
    rel(ROOT / "src" / "transport" / "memory" / "memory_manager.cc"),
    rel(ROOT / "src" / "transport" / "oob" / "oob.cc"),
    rel(ROOT / "src" / "transport" / "oob" / "oob_shm.cc"),
    rel(ROOT / "src" / "transport" / "oob" / "oob_socket.cc"),
    rel(ROOT / "src" / "transport" / "request.cc"),
    rel(ROOT / "src" / "transport" / "adapter" / "tcp_adapter.cc"),
    rel(ROOT / "src" / "transport" / "adapter" / "uccl_adapter.cc"),
    rel(ROOT / "src" / "transport" / "adapter" / "ipc_adapter.cc"),
    rel(ROOT / "src" / "transport" / "utils.cc"),
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
    rel(ROOT / "src" / "transport" / "memory" / "memory_manager.cc"),
    rel(ROOT / "src" / "transport" / "oob" / "oob.cc"),
    rel(ROOT / "src" / "transport" / "oob" / "oob_shm.cc"),
    rel(ROOT / "src" / "transport" / "oob" / "oob_socket.cc"),
    rel(ROOT / "src" / "transport" / "request.cc"),
    rel(ROOT / "src" / "transport" / "adapter" / "tcp_adapter.cc"),
    rel(ROOT / "src" / "transport" / "adapter" / "uccl_adapter.cc"),
    rel(ROOT / "src" / "transport" / "adapter" / "ipc_adapter.cc"),
    rel(ROOT / "src" / "transport" / "utils.cc"),
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
    rel(GDRCOPY_ROOT / "include"),
    str(CUDA_HOME / "include"),
]

ext = CUDAExtension(
    name="ukernel_ccl._C",
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={
        "cxx": [
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
        ],
        "nvcc": [
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
        ],
    },
    library_dirs=[
        str(CUDA_HOME / "lib64"),
        str(GDRCOPY_ROOT / "src"),
    ],
    libraries=[
        "cudart",
        "cuda",
        "gflags",
        "z",
        "ibverbs",
        "nl-3",
        "nl-route-3",
        "pthread",
        "rdmacm",
        "gdrapi",
        "numa",
    ],
    extra_objects=[str(RDMA_STATIC.resolve())],
    runtime_library_dirs=[
        str(CUDA_HOME / "lib64"),
        str((GDRCOPY_ROOT / "src").resolve()),
    ],
)

p2p_ext = CUDAExtension(
    name="ukernel_p2p._C",
    sources=p2p_sources,
    include_dirs=include_dirs,
    extra_compile_args={
        "cxx": [
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
        ],
        "nvcc": [
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
        ],
    },
    library_dirs=[
        str(CUDA_HOME / "lib64"),
        str(GDRCOPY_ROOT / "src"),
    ],
    libraries=[
        "cudart",
        "cuda",
        "gflags",
        "z",
        "ibverbs",
        "nl-3",
        "nl-route-3",
        "pthread",
        "rdmacm",
        "gdrapi",
        "numa",
    ],
    extra_objects=[str(RDMA_STATIC.resolve())],
    runtime_library_dirs=[
        str(CUDA_HOME / "lib64"),
        str((GDRCOPY_ROOT / "src").resolve()),
    ],
)


setup(
    name="ukernel-ccl",
    version="0.1.0",
    packages=["ukernel_ccl", "ukernel_p2p"],
    package_dir={"ukernel_ccl": "ukernel_ccl", "ukernel_p2p": "ukernel_p2p"},
    ext_modules=[ext, p2p_ext],
    cmdclass={"build_ext": BuildExtension},
)
