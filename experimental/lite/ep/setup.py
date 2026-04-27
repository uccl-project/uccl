import ast
import re
import os
import subprocess
import setuptools
import importlib
import sys
import glob

from pathlib import Path
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths

current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_env_names = ('EP_JIT_CACHE_DIR', 'EP_JIT_PRINT_COMPILER_COMMAND', 'EP_NUM_TOPK_IDX_BITS', 'EP_NCCL_ROOT_DIR')

# Load discover module without triggering `deep_ep.__init__`
find_pkgs_spec = importlib.util.spec_from_file_location('find_pkgs', os.path.join(current_dir, 'deep_ep', 'utils', 'find_pkgs.py'))
find_pkgs = importlib.util.module_from_spec(find_pkgs_spec)
find_pkgs_spec.loader.exec_module(find_pkgs)


# Wheel specific: the wheels only include the SO name of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')


def get_package_version():
    with open(Path(current_dir) / 'deep_ep' / '__init__.py', 'r') as f:
        version_match = re.search(r'^__version__\s*=\s*(.*)$', f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))

    # noinspection PyBroadException
    try:
        status_cmd = ['git', 'status', '--porcelain']
        status_output = subprocess.check_output(status_cmd).decode('ascii').strip()
        if status_output:
            print(f'Warning: Git working directory is not clean. Uncommitted changes:\n{status_output}')
            assert False, 'Git working directory is not clean'

        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except:
        revision = '+local'
    return f'{public_version}{revision}'


def detect_torch_cuda_arch_list() -> str:
    if os.getenv('TORCH_CUDA_ARCH_LIST'):
        return os.environ['TORCH_CUDA_ARCH_LIST']
    if os.getenv('DEEP_EP_CUDA_ARCH'):
        return os.environ['DEEP_EP_CUDA_ARCH']
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()
        if output:
            return output[0].strip()
    except Exception:
        pass
    return '9.0'


def first_arch_major(arch_list: str) -> int:
    first_arch = arch_list.replace(' ', ';').split(';')[0].replace('+PTX', '')
    return int(first_arch.split('.')[0]) if '.' in first_arch else int(first_arch[0])


def existing_include_dirs(paths):
    return [path for path in paths if path and os.path.isdir(path)]


def find_library_name(root_dir: str, pattern: str) -> str:
    libs = sorted(glob.glob(os.path.join(root_dir, 'lib', pattern)))
    assert libs, f'Cannot find {pattern} under {root_dir}/lib'
    for lib in libs:
        if os.path.basename(lib) == pattern.rstrip('*'):
            return os.path.basename(lib)
    return os.path.basename(libs[0])


class CustomBuildPy(build_py):
    def run(self):
        # Make clusters' cache setting default into `envs.py`
        self.generate_default_envs()

        # Finally, run the regular build
        build_py.run(self)

    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        # noinspection PyShadowingNames
        for name in persistent_env_names:
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''

        # Create temporary build directory
        build_include_dir = os.path.join(self.build_lib, 'deep_ep')
        os.makedirs(build_include_dir, exist_ok=True)
        with open(os.path.join(self.build_lib, 'deep_ep', 'envs.py'), 'w') as f:
            f.write(code)


if __name__ == '__main__':
    build_legacy = bool(int(os.getenv('DEEP_EP_BUILD_LEGACY', '0')))
    nvshmem_root_dir = find_pkgs.find_nvshmem_root(optional=not build_legacy)
    nccl_root_dir = find_pkgs.find_nccl_root()
    arch_list = detect_torch_cuda_arch_list()
    os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list
    disable_sm90_features = bool(int(os.getenv('DISABLE_SM90_FEATURES', '1' if first_arch_major(arch_list) < 9 else '0')))

    # `128,2417` is used to suppress warnings of `fmt`
    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3', '--extended-lambda', '--diag-suppress=128,2417']
    sources = ['csrc/python_api.cpp']
    include_dirs = existing_include_dirs([
        f'{current_dir}/deep_ep/include',
        f'{current_dir}/third-party/fmt/include',
        f'{sys.prefix}/include',
        '/usr/local/cuda/include/cccl',
    ]) + include_paths()
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = ['-lcuda']

    # NVSHMEM flags
    if build_legacy:
        sources.extend(['csrc/kernels/legacy/layout.cu', 'csrc/kernels/legacy/intranode.cu',
                        'csrc/kernels/legacy/internode.cu', 'csrc/kernels/legacy/internode_ll.cu',
                        'csrc/kernels/backend/nvshmem.cu'])
        include_dirs.extend([f'{nvshmem_root_dir}/include'])
        library_dirs.extend([f'{nvshmem_root_dir}/lib'])
        nvcc_dlink.extend(['-dlink', f'-L{nvshmem_root_dir}/lib', '-lnvshmem_device'])
        extra_link_args.extend([f'-l:libnvshmem_host.so', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_root_dir}/lib'])
    else:
        cxx_flags.append('-DDEEP_EP_DISABLE_LEGACY')
        nvcc_flags.append('-DDEEP_EP_DISABLE_LEGACY')

    # NCCL flags
    nccl_lib_name = find_library_name(nccl_root_dir, 'libnccl.so*')
    sources.extend(['csrc/kernels/backend/nccl.cu'])
    include_dirs.extend([f'{nccl_root_dir}/include'])
    library_dirs.extend([f'{nccl_root_dir}/lib'])
    extra_link_args.extend([f'-l:{nccl_lib_name}', f'-Wl,-rpath,{nccl_root_dir}/lib'])

    # CUDA driver sources
    sources.extend(['csrc/kernels/backend/cuda_driver.cu'])

    if disable_sm90_features:
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')
    else:
        # CUDA 12 flags
        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
        assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
        os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # Legacy environment name
    if 'TOPK_IDX_BITS' in os.environ:
        assert 'EP_NUM_TOPK_IDX_BITS' not in os.environ
        os.environ['EP_NUM_TOPK_IDX_BITS'] = os.environ['TOPK_IDX_BITS']

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if 'EP_NUM_TOPK_IDX_BITS' in os.environ:
        num_topk_idx_bits = int(os.environ['EP_NUM_TOPK_IDX_BITS'])
        cxx_flags.append(f'-DEP_NUM_TOPK_IDX_BITS={num_topk_idx_bits}')
        nvcc_flags.append(f'-DEP_NUM_TOPK_IDX_BITS={num_topk_idx_bits}')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print(f' > Build legacy/NVSHMEM path: {build_legacy}')
    print(f' > NVSHMEM path: {nvshmem_root_dir}')
    print(f' > NCCL path: {nccl_root_dir}')
    # Print persistent env variables
    persistent_envs = []
    for name in persistent_env_names:
        if name in os.environ:
            persistent_envs.append((name, os.environ[name]))
    if len(persistent_envs) > 0:
        print(f' > Persistent envs:')
        for k, v in persistent_envs:
            print(f'   > {k}: {v}')
    print()

    setuptools.setup(
        name='deep_ep',
        version=get_package_version(),
        packages=setuptools.find_packages(include=['deep_ep', 'deep_ep.*']),
        package_data={
            'deep_ep': [
                'include/deep_ep/**/*',
            ]
        },
        ext_modules=[
            CUDAExtension(name='deep_ep._C',
                          include_dirs=include_dirs,
                          library_dirs=library_dirs,
                          sources=sources,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args)
        ],
        cmdclass={
            'build_ext': BuildExtension,
            'build_py': CustomBuildPy
        }
    )
