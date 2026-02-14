#!/usr/bin/env bash
set -e

LAM_DIR="/home/ubuntu/lam"
ENV_PATH="${LAM_DIR}/uccl_lam_local"

echo "Creating ${LAM_DIR} if it doesn't exist..."
mkdir -p "$LAM_DIR"

echo "Creating conda env at ${ENV_PATH} with Python 3.10..."
conda create -p "$ENV_PATH" python=3.10 -y

echo "Activating environment and starting shell..."
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"
exec "$SHELL"
