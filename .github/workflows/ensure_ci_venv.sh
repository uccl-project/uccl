#!/usr/bin/env bash
# Used by AMD self-hosted CI: ensure the uccl-ci-sandbox uv venv exists and has
# the expected ROCm torch/torchvision + build-time deps installed. Idempotent.
#
# After this script succeeds, callers should:
#   source "$UCCL_CI_SANDBOX/.venv/bin/activate"
#
# Env:
#   UCCL_CI_SANDBOX   (default: $HOME/uccl-ci-sandbox)
#   UCCL_CI_PYTHON    (default: 3.12)
#   UCCL_CI_TORCH_INDEX (default: https://download.pytorch.org/whl/nightly/rocm7.1)

set -euo pipefail

SANDBOX="${UCCL_CI_SANDBOX:-$HOME/uccl-ci-sandbox}"
PY_VER="${UCCL_CI_PYTHON:-3.12}"
TORCH_INDEX="${UCCL_CI_TORCH_INDEX:-https://download.pytorch.org/whl/nightly/rocm7.1}"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$SANDBOX"

if [ ! -d "$SANDBOX/.venv" ]; then
  # --seed installs pip/setuptools/wheel into the venv so that tools which
  # shell out to `python -m pip ...` (e.g. build.sh --install) work.
  uv venv "$SANDBOX/.venv" --python "$PY_VER" --seed
  # shellcheck disable=SC1091
  source "$SANDBOX/.venv/bin/activate"
  uv pip install --pre torch torchvision --index-url "$TORCH_INDEX"
  uv pip install nanobind pybind11
else
  # shellcheck disable=SC1091
  source "$SANDBOX/.venv/bin/activate"
  python -m pip --version >/dev/null 2>&1 || uv pip install pip setuptools wheel
  if ! python -c "import torch" >/dev/null 2>&1; then
    uv pip install --pre torch torchvision --index-url "$TORCH_INDEX" --reinstall
  fi
  python -c "import nanobind, pybind11" >/dev/null 2>&1 || uv pip install nanobind pybind11
fi
