#!/bin/bash
# format.sh - Format all C++ files in project

set -e

# Directories to format (excluding thirdparty/, scripts/, doc/, etc.)
DIRECTORIES=("afxdp" "efa" "gpu_driven" "rdma" "misc" "p2p" "include")
EXTENSIONS=("cpp" "cxx" "cc" "h" "hpp" "cu" "cuh")
EXCLUDE=("afxdp/lib")

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
    echo "clang-format could not be found. Please install it first."
    exit 1
fi

# Ensure clang-format version is 14
REQUIRED_VERSION="14"

# Get major version
INSTALLED_VERSION=$(clang-format --version | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -1 | cut -d. -f1)

if [ "$INSTALLED_VERSION" != "$REQUIRED_VERSION" ]; then
    echo "clang-format version $REQUIRED_VERSION is required. Found version: $INSTALLED_VERSION."
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
    clang-format -i "$FILE"
done

echo "Formatting Python files with black..."

PYTHON_DIRS=("p2p" "gpu_driven")  # Adjust as needed
BLACK_EXCLUDES=("thirdparty" "doc" "build")

# Convert to exclude args
BLACK_EXCLUDE_ARGS=$(IFS="|"; echo "${BLACK_EXCLUDES[*]}")

for DIR in "${PYTHON_DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "  → $DIR"
        black "$DIR" --exclude "$BLACK_EXCLUDE_ARGS"
    fi
done

echo "Formatting complete."