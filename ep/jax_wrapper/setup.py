from setuptools import setup, find_packages

setup(
    name="uccl_ep_jax",
    version="0.1.0",
    description="JAX bindings for UCCL-EP (MoE all-to-all dispatch/combine).",
    author="uccl",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "uccl",
        "numpy",
        # JAX itself is intentionally left as a soft dependency: users
        # install the flavour (CUDA / ROCm) that matches their build.
    ],
)
