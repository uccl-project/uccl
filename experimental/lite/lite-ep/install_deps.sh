#!/bin/bash
set -e

# Check CUDA availability and get version
check_cuda() {
    command -v nvcc &> /dev/null || command -v nvidia-smi &> /dev/null
}

# Check HIP availability and get version
check_rocm() {
    command -v hipcc &> /dev/null
}

get_cuda_version() {
    # Prefer nvcc if available
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}' | head -n1
        return
    fi
    # Fallback: parse "CUDA Version: 12.8" from nvidia-smi (driver reports supported CUDA)
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}' | head -n1
    fi
}

# Use `uv pip install` in uv-created virtualenvs (pyvenv.cfg marks them); otherwise plain pip.
pip_install() {
    if command -v uv &> /dev/null && [[ -n "${VIRTUAL_ENV:-}" ]] &&
        [[ -f "${VIRTUAL_ENV}/pyvenv.cfg" ]] &&
        grep -qE '^uv[[:space:]]*=' "${VIRTUAL_ENV}/pyvenv.cfg" 2> /dev/null; then
        uv pip install "$@"
    else
        pip install "$@"
    fi
}

# LLVM 14 formatter (PyPI distribution: clang-format)
pip_install "clang-format==14.0.6"
pip_install nanobind --upgrade
pip_install black

# Check if we're in a conda environment
if [[ ! -z "${CONDA_PREFIX}" ]]; then
    conda install -c conda-forge libstdcxx-ng -y
fi

# Install PyTorch with automatic CUDA version handling
echo "Checking CUDA environment..."
if check_cuda; then
    # Install CUDA dependencies
    CUDA_VERSION=$(get_cuda_version)
    echo "Detected CUDA version: $CUDA_VERSION"
    
    # Create PyTorch-compatible suffix (cuXXY where XXY is major*10 + minor)
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    PYTORCH_SUFFIX="cu$((10#$CUDA_MAJOR * 10 + 10#$CUDA_MINOR))"

    # PyTorch pip indices above cu130 are often absent (e.g. CUDA 13.2 -> cu132); cap at cu130.
    pt_num="${PYTORCH_SUFFIX#cu}"
    if [[ "$pt_num" =~ ^[0-9]+$ ]] && [[ "$pt_num" -gt 130 ]]; then
        echo "Detected index $PYTORCH_SUFFIX; using cu130 (highest reliably published stable index)"
        PYTORCH_SUFFIX="cu130"
    fi

    # Verify PyTorch wheel exists for this version, fallback to latest if not
    if curl --fail --output /dev/null --silent --head "https://download.pytorch.org/whl/$PYTORCH_SUFFIX/torch/" &> /dev/null; then
        echo "Using PyTorch suffix: $PYTORCH_SUFFIX"
    else
        echo "No exact match for $PYTORCH_SUFFIX, using latest compatible version"
        PYTORCH_SUFFIX="cu${CUDA_MAJOR}1"  # Fallback to major version + .1
    fi
    
    pip_install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$PYTORCH_SUFFIX"
elif check_rocm; then
    echo "Detected ROCM"
    # Install Pytorch using nightly
    pip_install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0
else
    echo "No CUDA or ROCM detected"
    exit 1
fi

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
if python -c "import torch" &> /dev/null; then
    echo "PyTorch installed successfully"
else
    echo "PyTorch installation failed. Please check your network connection or install manually."
    exit 1
fi

# Get PyTorch include paths
echo "Retrieving PyTorch path information..."
TORCH_INCLUDE=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'include')")
TORCH_API_INCLUDE=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'include/torch/csrc/api/include')")
TORCH_LIB=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")

# Configure environment variables
echo "Configuring environment variables..."
export CXXFLAGS="-I$TORCH_INCLUDE -I$TORCH_API_INCLUDE $CXXFLAGS"
export LDFLAGS="-L$TORCH_LIB $LDFLAGS"
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"

# Compilation instructions
echo "All dependencies installed and environment configured"
