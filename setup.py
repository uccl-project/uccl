import os
import re
import shutil
import subprocess
import sys
import sysconfig
import importlib.util
from pathlib import Path
from distutils import log

from setuptools import setup, find_packages, Extension, Command
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


class MakeClean(Command):
    """`python setup.py clean` -> run `make clean` plus wipe Python build dirs.
    """

    description = "run `make clean` and remove build/, dist/, *.egg-info/, etc."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        project_root = Path(__file__).parent.resolve()

        # 1. Delegate to the project Makefile so native sub-modules clean up
        #    their own .o/.d/.so files.
        makefile = project_root / "Makefile"
        if makefile.exists():
            cmd = ["make", "clean"]
            log.info("running %s in %s", " ".join(cmd), project_root)
            subprocess.run(cmd, cwd=str(project_root), check=False)

        # 2. Wipe top-level Python build artefacts.
        targets = [
            project_root / "build",
            project_root / "dist",
            project_root / "wheelhouse",
            project_root / "ep" / "build",
        ]
        targets += list(project_root.glob("*.egg-info"))
        targets += list((project_root / "ep").glob("*.egg-info"))

        # 3. Stale shared libraries that may have been copied in-tree by an
        #    earlier `make` (e.g. uccl/lib/*.so, uccl/p2p*.so,
        #    ep/python/uccl_ep/ep_cpp*.so).
        targets += list((project_root / "uccl" / "lib").glob("*.so"))
        targets += list((project_root / "uccl").glob("p2p*.so"))
        targets += list(
            (project_root / "ep" / "python" / "uccl_ep").glob("ep_cpp*.so")
        )

        for path in targets:
            if not path.exists() and not path.is_symlink():
                continue
            log.info("removing %s", path)
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


class MakeBuildExtension(_build_ext):
    """Custom build_ext that invokes ``make`` before compiling C extensions."""

    def is_rocm(self) -> bool:
        """Find the ROCm
            reference from pytorch/torch/utils/cpp_extension.py::_find_rocm_home
        """
        # Guess #1
        rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
        if rocm_home is None:
            # Guess #2: Support for ROCm distribution from TheRock
            # rocm-sdk-core installs everything under <site-packages>/_rocm_sdk_core
            # (include/, lib/, bin/, ...), so the module's own location is the
            # ROCM_HOME we want. Use find_spec to locate it without importing.
            spec = importlib.util.find_spec('_rocm_sdk_core')
            if spec is not None and spec.origin is not None:
                rocm_home = str(Path(spec.origin).parent.resolve())
        if rocm_home is None:
            # Guess #3
            hipcc_path = shutil.which('hipcc')
            if hipcc_path is not None:
                rocm_home = os.path.dirname(os.path.dirname(
                    os.path.realpath(hipcc_path)))
                # can be either <ROCM_HOME>/hip/bin/hipcc or <ROCM_HOME>/bin/hipcc
                if os.path.basename(rocm_home) == 'hip':
                    rocm_home = os.path.dirname(rocm_home)
            else:
                # Guess #4
                fallback_path = '/opt/rocm'
                if os.path.exists(fallback_path):
                    rocm_home = fallback_path
        if rocm_home is None or not os.path.exists(rocm_home):
            log.warn("No ROCm runtime is found, using ROCM_HOME='%s'", rocm_home)
            return False
        log.info("ROCm runtime is found at %s", rocm_home)
        return True

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

        if self.build_lib and not self.inplace:
            cmd.append(f"BUILD_LIB={Path(self.build_lib).resolve()}")

        cmd.append(ext.make_target)

        env = os.environ.copy()
        env.setdefault("PYTHON", sys.executable)
        env.update(ext.env)

        # Honour an explicit USE_ROCM override (e.g. ``USE_ROCM=0`` to force
        # the CUDA path on a hybrid host); otherwise auto-detect.
        if "USE_ROCM" not in env and self.is_rocm():
            env["USE_ROCM"] = "1"

        # When pip runs the build under PEP 517 isolation it injects the
        # build-env site-packages into PYTHONPATH so its own setuptools/wheel
        # take precedence. That PYTHONPATH leaks into the child python that
        # ep/setup.py drives via make and shadows the host venv, which makes
        # ``import torch`` fail. Drop it so the submake's python falls back
        # to its default site-packages (where torch is installed).
        env.pop("PYTHONPATH", None)

        log.info("running `%s` in %s", " ".join(cmd), ext.sourcedir)
        subprocess.check_call(cmd, cwd=str(ext.sourcedir), env=env)


_build_jobs = int(os.getenv("MAX_JOBS", "32"))

make_ext = MakeExtension(
    name="uccl.make",
    sourcedir=Path(__file__).parent,
    make_target="all",
    make_args=[
        f"PYTHON={sys.executable}",
        f"BUILD_JOBS={_build_jobs}"],
)

setup(
    name="uccl",
    version=VERSION,
    author="UCCL Team",
    description="UCCL: Ultra and Unified CCL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/uccl-project/uccl",
    packages=find_packages(include=["uccl", "uccl.*"]) + ["uccl.ep", "deep_ep"],
    package_dir={
        "uccl": "uccl",
        "uccl.ep": "ep/python/uccl_ep",
        "deep_ep": "ep/deep_ep_wrapper/deep_ep",
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
        "build_ext": MakeBuildExtension,
        "clean": MakeClean,
    },
)
