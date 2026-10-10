"""Python helpers for the standalone UCCL-GIN prototype."""

# On ROCm/CUDA the _uccl_gin extension is built with torch's CUDAExtension and
# links libtorch (c10/torch_hip). torch must be imported first so those shared
# libs are loaded before the extension's module init runs; otherwise importing
# _uccl_gin segfaults. Best-effort: skip if torch is absent (e.g. a non-torch
# Makefile build on NVIDIA).
try:
    import torch  # noqa: F401
except ImportError:
    pass

try:
    from ._uccl_gin import Context, mpi_finalize, mpi_rank, mpi_world_size
except ImportError:
    Context = None  # type: ignore[assignment]
    mpi_finalize = None  # type: ignore[assignment]
    mpi_rank = None  # type: ignore[assignment]
    mpi_world_size = None  # type: ignore[assignment]

__all__ = ["Context", "mpi_finalize", "mpi_rank", "mpi_world_size"]
