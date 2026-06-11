"""Python helpers for the standalone UCCL-GIN prototype."""

try:
    from ._uccl_gin import Context, mpi_finalize, mpi_rank, mpi_world_size
except ImportError:
    Context = None  # type: ignore[assignment]
    mpi_finalize = None  # type: ignore[assignment]
    mpi_rank = None  # type: ignore[assignment]
    mpi_world_size = None  # type: ignore[assignment]

__all__ = ["Context", "mpi_finalize", "mpi_rank", "mpi_world_size"]
