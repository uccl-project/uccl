"""
Setup script to build the MoE pack/unpack CUDA extension
Usage: python setup_pack_unpack.py
"""

from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup
import torch
import os

# Get CUDA architectures from torch
cuda_arch = torch.cuda.get_device_capability()
cuda_arch_str = f"{cuda_arch[0]}{cuda_arch[1]}"

setup(
    name='moe_pack_unpack',
    ext_modules=[
        CUDAExtension(
            name='moe_pack_unpack',
            sources=[
                'pack_unpack_ops.cpp',
                'pack_unpack_kernels.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    f'-arch=sm_{cuda_arch_str}',
                    '--threads=32',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    }
)
