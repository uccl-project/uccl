import os
import sys
import resource
from pathlib import Path

try:
    from . import _rocm_init
except ImportError:
    pass
else:
    _rocm_init.initialize()
    del _rocm_init

def has_efa():
    infiniband = Path("/sys/class/infiniband/")
    try:
        if infiniband.is_dir():
            return any("rdmap" in child.name for child in infiniband.iterdir())
        return False
    except (OSError, PermissionError):
        return False
is_efa = has_efa()


def set_files_limit():
    """
    Raise file descriptor soft limit to hard limit.

    This allows more concurrent sockets for high-performance communication,
    similar to NCCL's approach. Should be called early in program initialization.

    Returns:
        tuple: (new_limit, old_limit) on success, or (None, None) on failure
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f"[uccl] File descriptor limit raised: {soft} -> {hard}",
                  file=sys.stderr)
            return (hard, soft)
        else:
            print(f"[uccl] File descriptor limit already at maximum: {hard}",
                  file=sys.stderr)
            return (hard, hard)

    except (ValueError, OSError) as e:
        print(f"[uccl] Failed to set file descriptor limit: {e}",
              file=sys.stderr)
        return (None, None)

if not is_efa:
    try:
        from . import p2p
        from . import collective
        from . import transfer
    except ImportError:
        pass

__version__ = "0.0.1.post4"


def nccl_plugin_path():
    """Returns absolute path to the NCCL plugin .so file"""
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
    return os.path.join(lib_dir, "libnccl-net-uccl.so")


def rccl_plugin_path():
    """Returns absolute path to the RCCL plugin .so file"""
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
    return os.path.join(lib_dir, "librccl-net-uccl.so")


def efa_plugin_path():
    """Returns absolute path to the EFA plugin .so file"""
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
    return os.path.join(lib_dir, "libnccl-net-efa.so")


def efa_nccl_path():
    """Returns absolute path to the EFA NCCL .so file"""
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
    return os.path.join(lib_dir, "libnccl-efa.so")
