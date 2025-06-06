#!/bin/bash
# format.sh - Format all C++ files in project

set -e

# Directories to format (excluding thirdparty/, scripts/, doc/, etc.)
DIRECTORIES=("afxdp" "efa" "gpu_driven" "rdma_cuda" "rdma_hip" "misc")

# Extensions to format
EXTENSIONS=("cpp" "cxx" "cc" "h" "hpp")

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
    echo "clang-format could not be found. Please install it first."
    exit 1
fi

echo "Formatting C++ files..."

for DIR in "${DIRECTORIES[@]}"; do
    if [ -d "$DIR" ]; then
        for EXT in "${EXTENSIONS[@]}"; do
            FILES=$(find "$DIR" -type f -name "*.${EXT}")
            if [ -n "$FILES" ]; then
                echo "$FILES" | xargs clang-format -i
            fi
        done
    fi
done

echo "Formatting complete."