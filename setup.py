from setuptools import setup, find_packages

setup(
    name="uccl",
    version="0.0.1",
    author="Yang Zhou",
    author_email="yangzhou.rpc@gmail.com",
    description="UCCL: Ultra and Unified CCL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/uccl-project/uccl",
    packages=find_packages(),
    package_data={"uccl": ["lib/libnccl-net-uccl.so"]},
    license="Apache-2.0",
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.8',
)