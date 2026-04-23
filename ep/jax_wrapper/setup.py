from setuptools import setup, find_packages

setup(
    name="uccl_ep_jax",
    version="0.1.0",
    description="JAX bindings for UCCL-EP (MoE all-to-all dispatch/combine).",
    author="uccl",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # ``uccl.ep`` must be installed separately by the user (see
        # ``ep/setup.py`` / ``ep/README.md``). We intentionally do not
        # list the PyPI ``uccl`` package as a hard dependency because it
        # ships a CUDA-only wheel that would shadow a local ROCm build.
        "numpy",
        # JAX itself is intentionally left as a soft dependency: users
        # install the flavour (CUDA / ROCm) that matches their build.
    ],
)
