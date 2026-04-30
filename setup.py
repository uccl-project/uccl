import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from distutils import log

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext


def _is_freethreaded():
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def get_version():
    """Parse version from uccl/__init__.py without importing."""
    with open("uccl/__init__.py", "r") as f:
        content = f.read()
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE
    )
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string in uccl/__init__.py")


# Nanobind stable ABI requires Python >= 3.12.  On 3.12+ we emit a
# cp312-abi3 wheel via the _abi3_stub extension; on older Pythons the
# stub still exists (to force a platform-specific wheel tag) but without
# the limited-API flag, producing a cpXY-cpXY version-specific wheel.
_use_limited_api = not _is_freethreaded() and sys.version_info >= (3, 12)

VERSION = get_version()

def _detect_rocm() -> str:
    try:
        import torch  # noqa: F401

        return "1" if getattr(torch.version, "hip", None) else "0"
    except Exception:
        return "0"


def _detect_rocm_arch() -> str:
    env_arch = os.environ.get("ROCM_ARCH_LIST") or os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env_arch:
        return env_arch
    try:
        output = subprocess.check_output(
            ["rocminfo"], stderr=subprocess.DEVNULL, text=True
        )
        matches = re.findall(r"Name:\s+(gfx[0-9a-z]+)", output, re.IGNORECASE)
        if matches:
            # Preserve order while deduplicating
            seen = []
            for item in matches:
                if item not in seen:
                    seen.append(item)
            return ",".join(seen)
    except Exception:
        pass
    return ""

# Single package "uccl" for all backends (vLLM-style).
# Variants are distinguished by PEP 440 local version identifiers in the
# wheel filename (e.g. uccl-0.1.0+cu13, uccl-0.1.0+cu12.efa).
# The default cu12 build has no local version and is published to PyPI;
# all other variants are distributed via GitHub Releases.
#
# Install from PyPI:   pip install uccl                (CUDA 12 default)
# Install from GitHub: pip install uccl-0.1.0+cu13-... (download .whl)

abi3_ext = Extension(
    "uccl._platform_tag_stub",
    sources=["uccl/_platform_tag_stub.c"],
    py_limited_api=_use_limited_api,
    define_macros=[("Py_LIMITED_API", "0x030C0000")] if _use_limited_api else [],
)


class MakeExtension(Extension):
    """Extension wrapper that drives external `make` builds."""

    def __init__(self, name, sourcedir=".", make_target="all", make_args=None, env=None):
        super().__init__(name, sources=[])
        self.sourcedir = Path(sourcedir).resolve()
        self.make_target = make_target
        self.make_args = make_args or []
        self.env = env or {}


class build_ext(_build_ext):
    """Custom build_ext that invokes make before compiling C extensions."""

    def run(self):
        make_exts = [ext for ext in self.extensions if isinstance(ext, MakeExtension)]
        for ext in make_exts:
            self.build_make_extension(ext)
        # Drop make-only extensions so the base class doesn't expect artifacts.
        self.extensions = [ext for ext in self.extensions if not isinstance(ext, MakeExtension)]
        super().run()

    def build_make_extension(self, ext: MakeExtension):
        cmd = ["make"]
        if self.parallel:
            cmd.extend(["-j", str(self.parallel)])
        cmd.extend(ext.make_args)
        if ext.make_target:
            cmd.append(ext.make_target)
        env = os.environ.copy()
        env.setdefault("PYTHON", sys.executable)
        env.update(ext.env)
        log.info("running `%s` in %s", " ".join(cmd), ext.sourcedir)
        subprocess.check_call(cmd, cwd=str(ext.sourcedir), env=env)


_build_jobs = max(os.cpu_count() or 1, 1)
_rocm_flag = _detect_rocm()
_rocm_arch_list = _detect_rocm_arch() if _rocm_flag == "1" else ""


make_ext = MakeExtension(
    name="uccl.make",
    sourcedir=Path(__file__).parent,
    make_target="all",
    make_args=[
        f"PYTHON={sys.executable}",
        f"BUILD_JOBS={_build_jobs}",
        f"BULD_JOBS={_build_jobs}",
        f"ROCM_DETECTED={_rocm_flag}",
        *( [f"ROCM_ARCH_LIST={_rocm_arch_list}"] if _rocm_arch_list else [] ),
        *( [f"HIP_HOME={os.environ['HIP_HOME']}"] if os.environ.get("HIP_HOME") else [] ),
        *( [f"CONDA_LIB_HOME={os.environ['CONDA_LIB_HOME']}"] if os.environ.get("CONDA_LIB_HOME") else [] ),
    ],
)

setup(
    name="uccl",
    version=VERSION,
    author="UCCL Team",
    description="UCCL: Ultra and Unified CCL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/uccl-project/uccl",
    packages=find_packages(include=["uccl", "uccl.*", "deep_ep"]) + ["uccl.ep"],
    package_dir={
        "uccl": "uccl",
        "uccl.ep": "ep/python/uccl_ep",
    },
    ext_modules=[make_ext, abi3_ext],
    package_data={
        "uccl": [
            "lib/*.so",
            "p2p*.so",
            "lib/*.a",
            "collective.py",
            "utils.py",
        ],
        "uccl.ep": [
            "ep_cpp*.so",
        ],
    },
    license="Apache-2.0",
    install_requires=["intervaltree"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.12",
    options={"bdist_wheel": {"py_limited_api": "cp312"}} if _use_limited_api else {},
    extras_require={
        "rocm": [],
    },
    cmdclass={
        "build_ext": build_ext,
    },
)
