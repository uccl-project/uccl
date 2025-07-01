FROM ubuntu:22.04

# Use noninteractive to skip prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    g++ \
    make \
    patchelf \
    rdma-core \
    libibverbs-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libgtest-dev \
    libelf-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python-is-python3 \
    python3-venv

# Set CUDA path (bind-mount from host)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

WORKDIR /io

# Optional: preinstall build tools
RUN pip install build auditwheel
