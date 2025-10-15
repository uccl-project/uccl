#!/bin/bash
set -e

sudo apt install -y nvtop
sudo apt install -y libgoogle-glog-dev
sudo apt install -y clang-format-14

CONDA_ENV_NAME="${1}"
if [ -z "$CONDA_ENV_NAME" ]; then
    echo "Please provide the conda environment name as the first argument, e.g., bash install_deps.sh myenv"
    exit 1
fi

# Ensure conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV_NAME"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

conda install -c conda-forge libstdcxx-ng
pip install black

# Check if pip is installed in the environment
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip3..."
    sudo apt update
    sudo apt install -y python3-pip
fi

# Install pybind11
echo "Installing pybind11..."
pip3 install pybind11 --upgrade

# Check CUDA availability and get version
check_cuda() {
    command -v nvcc &> /dev/null
}

get_cuda_version() {
    # Extracts version like "12.8" from nvcc output
    nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}' | head -n1
}

# Install PyTorch with automatic CUDA version handling
echo "Checking CUDA environment..."
if check_cuda; then
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
else
    echo "No CUDA detected"
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