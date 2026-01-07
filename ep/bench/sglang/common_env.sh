#!/bin/bash
# Common environment setup for SGLang server launches
# Source this file in your launch scripts: source "$(dirname "$0")/common_env.sh"

# ============================
# Library paths
# ============================
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:${LD_LIBRARY_PATH}

# ============================
# NVSHMEM configuration
# ============================
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa

# ============================
# NCCL configuration
# ============================
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="^lo,docker"

# ============================
# Libfabric/EFA configuration
# ============================
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1

# ============================
# CUDA and SGLang configuration
# ============================
export CUDA_HOME=/usr/local/cuda
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SG_DEEPGEMM_JIT=1
