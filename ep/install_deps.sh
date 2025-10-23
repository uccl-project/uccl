#!/bin/bash
set -e

pip3 install black

echo "Installing pybind11..."
pip3 install pybind11 --upgrade

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
    if curl --output /dev/null --silent --head "https://download.pytorch.org/whl/$PYTORCH_SUFFIX/torch/" &> /dev/null; then
        echo "Using PyTorch suffix: $PYTORCH_SUFFIX"
    else
        echo "No exact match for $PYTORCH_SUFFIX, using latest compatible version"
        PYTORCH_SUFFIX="cu${CUDA_MAJOR}1"  # Fallback to major version + .1
    fi
    
    pip3 install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$PYTORCH_SUFFIX"
elif check_rocm; then
    # Install Pytorch using nightly
    pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0
else
    echo "No CUDA or ROCM detected"
    exit 1
fi

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
if python3 -c "import torch" &> /dev/null; then
    echo "PyTorch installed successfully"
else
    echo "PyTorch installation failed. Please check your network connection or install manually."
    exit 1
fi

# Get PyTorch include paths
echo "Retrieving PyTorch path information..."
TORCH_INCLUDE=$(python3 -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'include')")
TORCH_API_INCLUDE=$(python3 -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'include/torch/csrc/api/include')")
TORCH_LIB=$(python3 -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")

# Configure environment variables
echo "Configuring environment variables..."
export CXXFLAGS="-I$TORCH_INCLUDE -I$TORCH_API_INCLUDE $CXXFLAGS"
export LDFLAGS="-L$TORCH_LIB $LDFLAGS"
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"

# Compilation instructions
echo "All dependencies installed and environment configured"