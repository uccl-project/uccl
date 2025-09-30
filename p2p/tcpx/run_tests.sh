#!/bin/bash
#************************************************************************
# TCPX Test Runner Script
# Copyright (c) 2024, UCCL Project. All rights reserved.
#************************************************************************

echo "TCPX Unpack Architecture Test Suite"
echo "===================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
total_tests=0
passed_tests=0

run_test() {
    local test_name="$1"
    local test_executable="$2"
    
    echo -e "${YELLOW}Running $test_name...${NC}"
    total_tests=$((total_tests + 1))
    
    if [ ! -f "$test_executable" ]; then
        echo -e "${RED}✗ $test_name: Executable not found${NC}"
        echo
        return 1
    fi
    
    if ./"$test_executable"; then
        echo -e "${GREEN}✓ $test_name: PASSED${NC}"
        passed_tests=$((passed_tests + 1))
    else
        echo -e "${RED}✗ $test_name: FAILED${NC}"
    fi
    echo
}

# Check if tests are built
echo "Checking test executables..."
if [ ! -f "tests/test_build_simple" ]; then
    echo "Tests not built. Building now..."
    make all_tests
    echo
fi

# Run tests in order
echo "=== Step 1: Basic Build Test ==="
run_test "Basic Build Test" "tests/test_build_simple"

echo "=== Step 2: RX Component Tests ==="
run_test "RX CMSG Parser Test" "tests/test_rx_cmsg_parser"
run_test "RX Descriptor Builder Test" "tests/test_rx_descriptor"

echo "=== Step 3: Device Component Tests ==="
# Check if CUDA is available
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "CUDA detected, running device tests..."
    run_test "Device Unpack Test" "tests/test_device_unpack"
else
    echo "No CUDA detected, skipping device tests"
fi

echo "=== Step 4: Integration Tests ==="
if [ -f "tests/test_tcpx_transfer" ]; then
    echo "End-to-end transfer test available"
    echo "To run: ./tests/test_tcpx_transfer server (in one terminal)"
    echo "        ./tests/test_tcpx_transfer client <server_ip> (in another terminal)"
else
    echo "Integration test not available"
fi

# Summary
echo "=== Test Summary ==="
echo "Total tests run: $total_tests"
echo "Tests passed: $passed_tests"
echo "Tests failed: $((total_tests - passed_tests))"

if [ $passed_tests -eq $total_tests ]; then
    echo -e "${GREEN}✓ All tests PASSED!${NC}"
    echo
    echo "TCPX Unpack Architecture is working correctly!"
    echo
    echo "Next steps:"
    echo "1. Run integration test: ./tests/test_tcpx_transfer"
    echo "2. Test with real NCCL workloads"
    echo "3. Performance tuning and optimization"
    exit 0
else
    echo -e "${RED}✗ Some tests FAILED!${NC}"
    echo
    echo "Please check the failed tests and fix any issues."
    exit 1
fi
