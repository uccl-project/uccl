import os
import sys
import importlib
import warnings
from pathlib import Path

try:
    from . import _rocm_init
except ImportError:
    pass
else:
    _rocm_init.initialize()
    del _rocm_init

_plugins_registered = False

def has_efa():
    infiniband = Path("/sys/class/infiniband/")
    try:
        if infiniband.is_dir():
            return any("rdmap" in child.name for child in infiniband.iterdir())
        return False
    except (OSError, PermissionError):
        return False
is_efa = has_efa()

if not is_efa:
    try:
        from . import p2p
        from . import collective
    except ImportError:
        pass

try:
    from . import ep
except ImportError:
    pass

__version__ = "0.1.0.post6"


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


def _discover_uccl_plugins() -> None:
    """Discover and import UCCL plugin modules.

    Plugins are advertised through the ``uccl_plugins`` entry-point group.
    Each entry point provides the public attribute name exposed on ``uccl``
    and the Python module path to import.

    During startup, UCCL imports every discovered plugin module and binds it
    onto the ``uccl`` package using the entry-point name. Plugin modules should
    keep import-time work lightweight, because all installed plugins are loaded
    eagerly during discovery.
    """
    # Read plugin registrations advertised via Python entry points.
    from importlib.metadata import entry_points

    global _plugins_registered

    plugin_modules = set()

    if not _plugins_registered:
        # Collect the public plugin name and its backing module path.
        for entry_point in entry_points(group="uccl_plugins"):
            plugin_modules.add((entry_point.name, entry_point.value))

        # Import each plugin module and expose it on the uccl package.
        for sub_module_name, plugin_module_name in plugin_modules:
            plugin_module = None
            try:
                plugin_module = importlib.import_module(plugin_module_name)                    
                fqn = f"{__name__}.{sub_module_name}"
                # Check if the module is already loaded and if it is the same module
                if fqn in sys.modules and sys.modules[fqn] is not plugin_module:
                    raise RuntimeError(
                        f"UCCL plugin registration error: module alias "
                        f"{fqn} already exists"
                    )
                sys.modules[fqn] = plugin_module
                setattr(sys.modules[__name__], sub_module_name, plugin_module)
            except ModuleNotFoundError:
                warnings.warn(f"UCCL plugin configuration error: Plugin module {plugin_module_name} "
                               "does not exist")
            except ImportError:
                warnings.warn(f"UCCL plugin configuration error: Plugin module {plugin_module_name} "
                                 "could not be loaded")

        _plugins_registered= True


_discover_uccl_plugins()
