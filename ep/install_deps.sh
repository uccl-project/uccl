#!/bin/bash
set -e

# Check CUDA availability and get version
check_cuda() {
    [[ -n "$(get_cuda_version)" ]]
}

# Check HIP availability and get version
check_rocm() {
    command -v hipcc &> /dev/null
}

get_cuda_version() {
    # Prefer CUDA toolkit version when nvcc is available.
    if command -v nvcc &> /dev/null; then
        # Extracts version like "12.8" from nvcc output.
        nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}' | head -n1
        return
    fi

    # Fallback to NVIDIA driver-reported runtime CUDA version.
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}' | head -n1
    fi
}

# Install common dependencies (system packages require root privileges)
if [[ $EUID -eq 0 ]]; then
    apt install -y nvtop clang-format-14 python3-pip
else
    echo "Not running as root; skipping apt install of system packages."
    echo "Please install these manually if needed: nvtop clang-format-14 python3-pip"
fi
pip install pybind11 --upgrade
pip install black

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
    
    # Verify PyTorch wheel exists for this version, fallback to latest if not.
    if command -v curl &> /dev/null; then
        if curl --output /dev/null --silent --head "https://download.pytorch.org/whl/$PYTORCH_SUFFIX/torch/" &> /dev/null; then
            echo "Using PyTorch suffix: $PYTORCH_SUFFIX"
        else
            echo "No exact match for $PYTORCH_SUFFIX, using latest compatible version"
            PYTORCH_SUFFIX="cu${CUDA_MAJOR}1"  # Fallback to major version + .1
        fi
    else
        echo "curl not found; skipping wheel existence check and using: $PYTORCH_SUFFIX"
    fi
    
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$PYTORCH_SUFFIX"
elif check_rocm; then
    echo "Detected ROCM"
    # Install Pytorch using nightly
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0
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
