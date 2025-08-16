import os

is_efa = "rdmap" in " ".join(os.listdir("/sys/class/infiniband/"))
if not is_efa:
    try:
        from . import p2p
        from . import collective
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
