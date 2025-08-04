import re
import os
from setuptools import setup, find_packages, Extension


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


setup(
    name="uccl",
    version=get_version(),
    author="UCCL Team",
    description="UCCL: Ultra and Unified CCL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/uccl-project/uccl",
    packages=find_packages(),
    package_data={
        "uccl": ["lib/*.so", "p2p*.so", "gpu_driven*.so", "collective.py"],
    },
    license="Apache-2.0",
    install_requires=["pybind11>=2.6.0", "paramiko"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
    extras_require={
        "cuda": [],
        "rocm": [],
    },
)
