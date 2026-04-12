#!/bin/bash
# Common environment setup for SGLang server launches with Intel irdma NICs
# Source this file in your launch scripts: source "$(dirname "$0")/common_env_intel_nic.sh"

# ============================
# Library paths
# ============================
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# ============================
# Intel RDMA (irdma) configuration
# ============================
export NCCL_IB_HCA="irdma-mkp0:1"
export UCCL_IB_HCA="irdma-mkp0:1"
export NCCL_IB_GID_INDEX=1
export UCCL_IB_GID_INDEX=1

# ============================
# NCCL configuration
# ============================
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="eno0"
export UCCL_SOCKET_IFNAME="eno0"

# ============================
# CUDA and SGLang configuration
# ============================
export CUDA_HOME=/usr/local/cuda
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SG_DEEPGEMM_JIT=1
