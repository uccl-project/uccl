#!/bin/bash
# format.sh - Format all C++ files in project

set -e

# Directories to format (excluding thirdparty/, scripts/, docs/, etc.)

DIRECTORIES=("collective" "ep" "p2p" "include" "experimental")

EXTENSIONS=("cpp" "cxx" "cc" "h" "hpp" "cu" "cuh")
EXCLUDE=("collective/afxdp/lib")

REQUIRED_VERSION="14"

# Use CLANG_FORMAT if set; otherwise prefer clang-format-14, then fall back to clang-format
if [ -n "$CLANG_FORMAT" ]; then
    :
elif command -v clang-format-14 &> /dev/null; then
    CLANG_FORMAT="clang-format-14"
else
    CLANG_FORMAT="clang-format"
fi

# Check if clang-format is installed
if ! command -v "$CLANG_FORMAT" &> /dev/null; then
    echo "$CLANG_FORMAT could not be found. Please install clang-format-14 (e.g. apt install clang-format-14)."
    exit 1
fi

# Ensure clang-format version is exactly 14
INSTALLED_VERSION=$("$CLANG_FORMAT" --version | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -1 | cut -d. -f1)

if [ "$INSTALLED_VERSION" != "$REQUIRED_VERSION" ]; then
    echo "clang-format version $REQUIRED_VERSION is required. Found version: $INSTALLED_VERSION ($CLANG_FORMAT)."
    echo "Install version 14 (e.g. apt install clang-format-14) or set CLANG_FORMAT=/path/to/clang-format-14."
    exit 1
fi

echo "Formatting C++ files..."

EXCLUDE_ARGS=()
for EXC in "${EXCLUDE[@]}"; do
    EXCLUDE_ARGS+=( -path "$EXC" -prune -o )
done

FILES=()

for DIR in "${DIRECTORIES[@]}"; do
    if [ -d "$DIR" ]; then
        for EXT in "${EXTENSIONS[@]}"; do
            while IFS= read -r -d '' FILE; do
                FILES+=("$FILE")
            done < <(find "$DIR" "${EXCLUDE_ARGS[@]}" -type f -name "*.${EXT}" -print0)
        done
    fi
done

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No files to format."
    exit 0
fi

for FILE in "${FILES[@]}"; do
    echo "Formatting $FILE"
    "$CLANG_FORMAT" -i "$FILE"
done

echo "Formatting Python files with black..."

PYTHON_DIRS=("p2p" "ep")  # Adjust as needed
BLACK_EXCLUDES=("thirdparty" "docs" "build")

# Convert to exclude args
BLACK_EXCLUDE_ARGS=$(IFS="|"; echo "${BLACK_EXCLUDES[*]}")

for DIR in "${PYTHON_DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "  → $DIR"
        black "$DIR" --exclude "$BLACK_EXCLUDE_ARGS"
    fi
done

echo "Formatting complete."