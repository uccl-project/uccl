from setuptools import setup, find_packages, Extension

ext_modules = [
    Extension(
        name="uccl.p2p",
        sources=[],
    )
]

setup(
    name="uccl",
    version="0.0.1.post2",
    author="UCCL Team",
    description="UCCL: Ultra and Unified CCL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/uccl-project/uccl",
    packages=find_packages(),
    package_data={
        "uccl": ["lib/*.so", "p2p*.so"],
    },
    ext_modules=ext_modules,
    license="Apache-2.0",
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.8',
    extras_require={
        "cuda": [],
        "rocm": [],
    },
)