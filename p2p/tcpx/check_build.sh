#!/bin/bash
#************************************************************************
# TCPX Build Diagnostic Script
# Copyright (c) 2024, UCCL Project. All rights reserved.
#************************************************************************

echo "=== TCPX Build Diagnostic ==="
echo "Current directory: $(pwd)"
echo

# Check core source files
echo "=== Core Source Files ==="
echo "RX Components:"
for file in rx/rx_cmsg_parser.cc rx/rx_cmsg_parser.h rx/rx_descriptor.cc rx/rx_descriptor.h; do
    if [ -f "$file" ]; then
        echo "✓ $file ($(wc -l < "$file") lines)"
    else
        echo "✗ $file - MISSING"
    fi
done

echo
echo "Device Components:"
for file in device/unpack_kernels.cu device/unpack_launch.cu device/unpack_launch.h; do
    if [ -f "$file" ]; then
        echo "✓ $file ($(wc -l < "$file") lines)"
    else
        echo "✗ $file - MISSING"
    fi
done

echo
echo "Implementation Files:"
for file in tcpx_impl.cc tcpx_interface.h; do
    if [ -f "$file" ]; then
        echo "✓ $file ($(wc -l < "$file") lines)"
    else
        echo "✗ $file - MISSING"
    fi
done

# Check test files
echo
echo "=== Test Files ==="
for file in tests/test_tcpx_transfer.cc tests/test_build_simple.cc tests/test_device_discovery.cc tests/test_connection.cc tests/test_tcpx.cc tests/test_performance.cc; do
    if [ -f "$file" ]; then
        echo "✓ $file ($(wc -l < "$file") lines)"
    else
        echo "✗ $file - MISSING"
    fi
done

# Check built objects
echo
echo "=== Built Objects ==="
for file in rx/rx_cmsg_parser.o rx/rx_descriptor.o device/unpack_kernels.o device/unpack_launch.o; do
    if [ -f "$file" ]; then
        echo "✓ $file ($(stat -c%s "$file") bytes)"
    else
        echo "✗ $file - NOT BUILT"
    fi
done

# Check executables
echo
echo "=== Executables ==="
for file in tests/test_build_simple tests/test_tcpx_transfer tests/test_device_discovery tests/test_connection tests/test_tcpx tests/test_performance; do
    if [ -f "$file" ]; then
        echo "✓ $file ($(stat -c%s "$file") bytes)"
    else
        echo "✗ $file - NOT BUILT"
    fi
done

# Check tools
echo
echo "=== Build Tools ==="
if command -v g++ >/dev/null 2>&1; then
    echo "✓ g++ - $(g++ --version | head -1)"
else
    echo "✗ g++ - NOT FOUND"
fi

if command -v nvcc >/dev/null 2>&1; then
    echo "✓ nvcc - $(nvcc --version | grep release)"
else
    echo "✗ nvcc - NOT FOUND"
fi

if command -v make >/dev/null 2>&1; then
    echo "✓ make - $(make --version | head -1)"
else
    echo "✗ make - NOT FOUND"
fi

# Check CUDA
echo
echo "=== CUDA Environment ==="
if [ -d "/usr/local/cuda" ]; then
    echo "✓ CUDA installation found at /usr/local/cuda"
    if [ -f "/usr/local/cuda/include/cuda_runtime.h" ]; then
        echo "✓ CUDA headers available"
    else
        echo "✗ CUDA headers missing"
    fi
    if [ -f "/usr/local/cuda/lib64/libcudart.so" ]; then
        echo "✓ CUDA runtime library available"
    else
        echo "✗ CUDA runtime library missing"
    fi
else
    echo "✗ CUDA installation not found"
fi

# Check GPU
echo
echo "=== GPU Information ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✓ nvidia-smi available"
    nvidia-smi -L | head -3
else
    echo "✗ nvidia-smi not found"
fi

echo
echo "=== Build Recommendations ==="

# Count missing files
missing_core=0
missing_tests=0

for file in rx/rx_cmsg_parser.cc rx/rx_cmsg_parser.h rx/rx_descriptor.cc rx/rx_descriptor.h device/unpack_kernels.cu device/unpack_launch.cu device/unpack_launch.h tcpx_impl.cc tcpx_interface.h; do
    if [ ! -f "$file" ]; then
        ((missing_core++))
    fi
done

for file in tests/test_tcpx_transfer.cc tests/test_build_simple.cc; do
    if [ ! -f "$file" ]; then
        ((missing_tests++))
    fi
done

if [ $missing_core -eq 0 ]; then
    echo "✓ All core files present - try: make core"
    if [ $missing_tests -eq 0 ]; then
        echo "✓ All test files present - try: make all"
    else
        echo "⚠ Some test files missing - try: make core test_build_simple"
    fi
else
    echo "✗ Missing $missing_core core files - check file locations"
fi

echo
echo "=== Quick Commands ==="
echo "make check     - Run Makefile file check"
echo "make core      - Build only core components"
echo "make quick     - Build core + simple test"
echo "make clean     - Clean all build artifacts"
echo "make all       - Build everything (if files present)"
