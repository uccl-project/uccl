#!/usr/bin/env bash
# Format tracked files using the same configuration as commits and CI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -x .venv-format/bin/pre-commit ]]; then
    PRE_COMMIT="$ROOT/.venv-format/bin/pre-commit"
elif command -v pre-commit >/dev/null 2>&1; then
    PRE_COMMIT="pre-commit"
else
    echo "Run 'bash scripts/install-hooks.sh' to install the pinned formatters." >&2
    exit 1
fi

# pre-commit exits nonzero if it makes changes or encounters an error.
exec "$PRE_COMMIT" run --all-files --show-diff-on-failure "$@"
