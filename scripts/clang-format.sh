#!/usr/bin/env bash
# clang-format 14's PyPI macOS wheel is Intel-only despite its universal2 tag.
set -euo pipefail

FORMATTER="${CLANG_FORMAT:-clang-format}"
if [[ -z "${CLANG_FORMAT:-}" && "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
    if ! command -v brew >/dev/null 2>&1; then
        echo "Apple Silicon needs Homebrew llvm@14, or CLANG_FORMAT pointing to native clang-format 14.0.6." >&2
        exit 1
    fi
    FORMATTER="$(brew --prefix llvm@14)/bin/clang-format"
fi

if ! VERSION="$("$FORMATTER" --version)" || [[ ! "$VERSION" =~ version[[:space:]]14\.0\.6([[:space:]]|$) ]]; then
    echo "clang-format 14.0.6 is required. Run 'bash scripts/install-hooks.sh'." >&2
    exit 1
fi
exec "$FORMATTER" -i "$@"
