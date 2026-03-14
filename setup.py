import re
import os
import sys
import sysconfig
from setuptools import setup, find_packages, Extension


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
PACKAGE_NAME = os.environ.get("UCCL_PACKAGE_NAME", "uccl")
_is_backend = PACKAGE_NAME != "uccl"

# When UCCL_PACKAGE_NAME is set (e.g. "uccl-cu12", "uccl-cu13"), build the backend
# package with compiled .so files. Users still "import uccl" — the backend provides
# the uccl namespace.
if _is_backend:
    abi3_ext = Extension(
        "uccl._platform_tag_stub",
        sources=["uccl/_platform_tag_stub.c"],
        py_limited_api=_use_limited_api,
        define_macros=[("Py_LIMITED_API", "0x030C0000")] if _use_limited_api else [],
    )
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        author="UCCL Team",
        description="UCCL: Ultra and Unified CCL",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/uccl-project/uccl",
        packages=find_packages(),
        ext_modules=[abi3_ext],
        package_data={
            "uccl": [
                "lib/*.so",
                "p2p*.so",
                "ep*.so",
                "lib/*.a",
                "collective.py",
                "utils.py",
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
    )
else:
    # Meta-package: no code, extras pull in the right backend (cu12/cu13 with CUDA).
    # Whichever extra you install, you always use "import uccl" in Python.
    #   pip install uccl             → CUDA 12 (default)
    #   pip install uccl[cu13]       → CUDA 13
    #   pip install uccl[cu12-efa]   → CUDA 12 + EFA
    #   pip install uccl[cu13-efa]   → CUDA 13 + EFA
    #   pip install uccl[rocm]       → ROCm (use --extra-index-url for index)
    setup(
        name="uccl",
        version=VERSION,
        author="UCCL Team",
        description="UCCL: Ultra and Unified CCL",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/uccl-project/uccl",
        packages=[],
        license="Apache-2.0",
        classifiers=[
            "Programming Language :: Python :: 3",
        ],
        python_requires=">=3.12",
        install_requires=[f"uccl-cu12=={VERSION}"],
        extras_require={
            "cu12": [f"uccl-cu12=={VERSION}"],
            "cu13": [f"uccl-cu13=={VERSION}"],
            "cu12-efa": [f"uccl-cu12-efa=={VERSION}"],
            "cu13-efa": [f"uccl-cu13-efa=={VERSION}"],
            "rocm": [f"uccl-rocm=={VERSION}"],
        },
    )
