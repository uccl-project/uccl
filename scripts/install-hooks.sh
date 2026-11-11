#!/usr/bin/env bash
# Install formatting hooks for this clone, without GPU/build dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${FORMAT_PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'

# The pinned PyPI wheel cannot run natively on Apple Silicon. Homebrew's
# versioned package supplies the same formatter without replacing system LLVM.
if [[ -z "${CLANG_FORMAT:-}" && "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
    if ! command -v brew >/dev/null 2>&1; then
        echo "Install Homebrew, or set CLANG_FORMAT to native clang-format 14.0.6." >&2
        exit 1
    fi
    if [[ ! -x "$(brew --prefix llvm@14)/bin/clang-format" ]]; then
        brew install llvm@14
    fi
fi

if [[ ! -x .venv-format/bin/python ]]; then
    "$PYTHON" -m venv .venv-format
fi
.venv-format/bin/python -m pip install --disable-pip-version-check -r scripts/requirements-format.txt
.venv-format/bin/pre-commit install --install-hooks --allow-missing-config
echo "Formatting hooks installed. If a commit is reformatted, review, re-stage, and commit again."
