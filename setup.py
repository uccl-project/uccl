import re
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
setup(
    name="uccl",
    version=VERSION,
    author="UCCL Team",
    description="UCCL: Ultra and Unified CCL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/uccl-project/uccl",
    packages=find_packages() + ["uccl.ep"],
    package_dir={
        "uccl.ep": "ep/python/uccl_ep",
    },
    ext_modules=[abi3_ext],
    package_data={
        "uccl": [
            "lib/*.so",
            "p2p*.so",
            "lib/*.a",
            "collective.py",
            "utils.py",
        ],
        "uccl.ep": [
            "_ep_native*.so",
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
