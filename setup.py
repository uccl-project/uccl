import os
import re
import shutil
import subprocess
import sys
import sysconfig
import importlib.util
from pathlib import Path
from distutils import log
from typing import Optional
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


BUILD_SCRIPT = "build_native.sh"


class ShellExtension(Extension):
    """Extension wrapper that drives external shell-script builds."""

    def __init__(self, name, sourcedir=".", script=BUILD_SCRIPT,
                 targets=("all",), env=None):
        super().__init__(name, sources=[])
        self.sourcedir = Path(sourcedir).resolve()
        self.script = script
        self.targets = list(targets)
        self.env = env or {}


class ShellClean(Command):
    """`python setup.py clean` -> run ``build_native.sh clean`` plus wipe
    Python build dirs.
    """

    description = (
        "run `build_native.sh clean` and remove build/, dist/, *.egg-info/, etc."
    )
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        project_root = Path(__file__).parent.resolve()

        # 1. Delegate to build_native.sh so native sub-modules clean up
        #    their own .o/.d/.so files.
        script = project_root / BUILD_SCRIPT
        if script.exists():
            cmd = ["bash", str(script), "clean"]
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
        #    earlier build (e.g. uccl/lib/*.so, uccl/p2p*.so,
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


class ShellBuildExtension(_build_ext):
    """Custom build_ext that invokes ``build_native.sh`` before compiling
    C extensions.
    """

    @staticmethod
    def _find_rocm_home()->Optional[Path]:
        """Return ROCm install dir as ``Path``, or ``None``.

        Adapted from pytorch ``_find_rocm_home``.
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
            return None
        log.info("ROCm runtime is found at %s", rocm_home)
        return Path(rocm_home)

    @staticmethod
    def _detect_rocm_major(rocm_home: Path):
        """Return ROCm major version, or ``None`` if undetectable."""
        # ``X.Y.Z`` text file shipped by most ROCm installs.
        version_file = rocm_home / ".info" / "version"
        if version_file.exists():
            try:
                head = version_file.read_text().strip().split(".")[0]
                return int(head)
            except (ValueError, OSError):
                pass

        # Fallback: ``HIP_VERSION_MAJOR`` from the hip header.
        hip_version_h = rocm_home / "include" / "hip" / "hip_version.h"
        if hip_version_h.exists():
            try:
                for line in hip_version_h.read_text().splitlines():
                    m = re.match(r"\s*#define\s+HIP_VERSION_MAJOR\s+(\d+)", line)
                    if m:
                        return int(m.group(1))
            except OSError:
                pass

        return None

    @staticmethod
    def _detect_target(env: dict) -> str:
        """Resolve build_native.sh ``TARGET``.

        Precedence: env ``TARGET`` > detected ROCm (``roc6`` / ``roc7``)
        > ``cu12``.
        """
        if env.get("TARGET"):
            return env["TARGET"]
        rocm_home = ShellBuildExtension._find_rocm_home()
        if rocm_home is None:
            return "cu12"
        major = ShellBuildExtension._detect_rocm_major(rocm_home)
        return "roc6" if major == 6 else "roc7"

    def _get_build_output_dir(self):
        """``uccl`` package output dir for build_native.sh.

        Editable/inplace -> source ``uccl/``; install/wheel -> ``build_lib/uccl``.
        build_native.sh derives ``uccl.ep`` from this internally.
        """
        if self.inplace or not self.build_lib:
            return Path(__file__).parent.resolve() / "uccl"
        return Path(self.build_lib).resolve() / "uccl"

    def run(self):
        shell_exts = [ext for ext in self.extensions if isinstance(ext, ShellExtension)]
        for ext in shell_exts:
            self.build_shell_extension(ext)
        # Drop shell-only extensions so the base class doesn't expect artifacts.
        self.extensions = [ext for ext in self.extensions if not isinstance(ext, ShellExtension)]
        super().run()

    def build_shell_extension(self, ext: ShellExtension):
        env = os.environ.copy()
        env.setdefault("PYTHON", sys.executable)
        env.update(ext.env)

        # Drop PEP 517 build-env PYTHONPATH so the child build script's python
        # sees the host venv (otherwise ``import torch`` in ep/setup.py fails).
        env.pop("PYTHONPATH", None)

        # build_native.sh dispatches on TARGET; auto-pick when caller didn't.
        env["TARGET"] = self._detect_target(env)

        # Tell build_native.sh where to drop the ``uccl`` package's
        # artefacts; the script derives the ``uccl.ep`` target dir from
        # this value internally.
        env["UCCL_PY_DIR"] = str(self._get_build_output_dir())

        cmd = ["bash", str(ext.sourcedir / ext.script), *ext.targets]

        log.info("running `%s` in %s", " ".join(cmd), ext.sourcedir)
        subprocess.check_call(cmd, cwd=str(ext.sourcedir), env=env)


shell_ext = ShellExtension(
    name="uccl.shell",
    sourcedir=Path(__file__).parent,
    script=BUILD_SCRIPT,
    targets=["all"],
)

setup(
    name="uccl",
    version=VERSION,
    author="UCCL Team",
    description="UCCL: Ultra and Unified CCL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/uccl-project/uccl",
    packages=find_packages(include=["uccl", "uccl.*", "uccl.ep"]) + ["uccl.ep"],
    package_dir={
        "uccl": "uccl",
        "uccl.ep": "ep/python/uccl_ep",
    },
    ext_modules=[shell_ext, abi3_ext],
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
        "build_ext": ShellBuildExtension,
        "clean": ShellClean,
    },
)
