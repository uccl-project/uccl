#!/usr/bin/env bash
# One-shot dependency setup for a fresh machine with CUDA (or ROCm via ep/install_deps.sh).
# Usage:
#   source scripts/bootstrap.sh

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
UCCL_ROOT="$(cd "$THIS_DIR/.." && pwd)"
cd "$UCCL_ROOT"

UV_INSTALL_URL="${UV_INSTALL_URL:-https://astral.sh/uv/install.sh}"
DEFAULT_PYTHON="${BOOTSTRAP_PYTHON:-3.12}"

ensure_uv() {
    if command -v uv &>/dev/null; then
        return 0
    fi
    echo "[bootstrap] uv not found; installing from $UV_INSTALL_URL"
    curl -LsSf "$UV_INSTALL_URL" | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    if ! command -v uv &>/dev/null; then
        echo "[bootstrap] uv is still not on PATH. Add ~/.local/bin (or ~/.cargo/bin) to PATH and re-run." >&2
        return 1
    fi
}

ensure_venv() {
    local vdir="$UCCL_ROOT/.venv"
    if [[ ! -d "$vdir" ]]; then
        echo "[bootstrap] creating uv virtualenv at $vdir (python $DEFAULT_PYTHON)"
        uv venv "$vdir" --python "$DEFAULT_PYTHON"
    fi
}

activate_venv() {
    source "$UCCL_ROOT/.venv/bin/activate"
}

ensure_uv
ensure_venv
activate_venv

echo "[bootstrap] running ep/install_deps.sh"
bash "$UCCL_ROOT/ep/install_deps.sh"

echo "[bootstrap] done."
