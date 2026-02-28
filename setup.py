import re
import os
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


# Stable-ABI stub so setuptools emits a cp38-abi3 platform tag (same trick as vLLM).
_use_limited_api = not _is_freethreaded()

abi3_ext = Extension(
    "uccl._abi3_stub",
    sources=["uccl/_abi3_stub.c"],
    py_limited_api=_use_limited_api,
    define_macros=[("Py_LIMITED_API", "0x03080000")] if _use_limited_api else [],
)

setup(
    name="uccl",
    version=get_version(),
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
    python_requires=">=3.8",
    options={"bdist_wheel": {"py_limited_api": "cp38"}} if _use_limited_api else {},
    extras_require={
        "cuda": [],
        "rocm": [],
    },
)
