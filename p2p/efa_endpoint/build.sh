#!/bin/bash
# Build script for EFA refactored code

set -e  # Exit on error

echo "Building efa_refactor..."

# Correct compilation command
g++ -std=c++17 -Wall -I. main.cpp \
    -libverbs -lefa -lcuda \
    -o efa_refactor

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo "  Executable: ./efa_refactor"
    ls -lh efa_refactor
else
    echo "✗ Build failed!"
    exit 1
fi
