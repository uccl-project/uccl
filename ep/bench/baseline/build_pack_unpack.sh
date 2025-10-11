#!/bin/bash
# Build script for MoE pack/unpack CUDA extension

set -e

cd "$(dirname "$0")"

echo "Building MoE pack/unpack CUDA extension..."
python setup_pack_unpack.py build_ext --inplace

echo "Done! Extension built successfully."
echo ""
echo "To test, run:"
echo "  python -c 'import moe_pack_unpack; print(\"CUDA extension loaded successfully!\")'"
