#!/bin/bash
set -e

# Check CUDA availability and get version
check_cuda() {
    command -v nvcc &> /dev/null
}

# Check HIP availability and get version
check_rocm() {
    command -v hipcc &> /dev/null
}

get_cuda_version() {
    # Extracts version like "12.8" from nvcc output
    nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}' | head -n1
}

# Run system-level commands with sudo when available, otherwise directly.
if [[ "$(id -u)" -eq 0 ]]; then
    SUDO_CMD=()
    CAN_INSTALL_APT=1
elif command -v sudo &> /dev/null && sudo -n true &> /dev/null; then
    SUDO_CMD=(sudo)
    CAN_INSTALL_APT=1
else
    SUDO_CMD=()
    CAN_INSTALL_APT=0
fi

# Install common dependencies
if [[ "$CAN_INSTALL_APT" -eq 1 ]]; then
    "${SUDO_CMD[@]}" apt install -y clang-format-14
else
    echo "No root/passwordless sudo. Skipping apt dependencies: clang-format-14"
fi
pip install pybind11 nanobind --upgrade
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
    
    # Verify PyTorch wheel exists for this version, fallback to latest if not
    if curl --fail --output /dev/null --silent --head "https://download.pytorch.org/whl/$PYTORCH_SUFFIX/torch/" &> /dev/null; then
        echo "Using PyTorch suffix: $PYTORCH_SUFFIX"
    # temporary fallback since cu131 is not available now
    elif [[ "$PYTORCH_SUFFIX" == "cu131" ]]; then
        echo "Detected PyTorch suffix cu131, which is currently not available, temporarily falling back to cu130 for now"
        PYTORCH_SUFFIX="cu130"
    else
        echo "No exact match for $PYTORCH_SUFFIX, using latest compatible version"
        PYTORCH_SUFFIX="cu${CUDA_MAJOR}1"  # Fallback to major version + .1
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
