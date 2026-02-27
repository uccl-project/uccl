from setuptools import setup, find_packages

setup(
    name="deep_ep",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "uccl_ep",
    ],
    author="uccl-project",
    description="A wrapper package for uccl_ep with additional functionality",
    python_requires=">=3.6",
)
