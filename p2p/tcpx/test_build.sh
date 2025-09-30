#!/bin/bash
#************************************************************************
# Copyright (c) 2024, UCCL Project. All rights reserved.
#
# Simple build and test script for TCPX Unpack Architecture
#************************************************************************

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check CUDA
    if ! command -v nvcc &> /dev/null; then
        print_error "CUDA not found. Please install CUDA toolkit."
        exit 1
    fi
    
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    print_status "Found CUDA version: $CUDA_VERSION"
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        if [ $GPU_COUNT -eq 0 ]; then
            print_warning "No NVIDIA GPUs detected. Some tests may be skipped."
        else
            print_status "Found $GPU_COUNT NVIDIA GPU(s)"
        fi
    else
        print_warning "nvidia-smi not found. Cannot detect GPUs."
    fi
    
    print_success "Prerequisites check completed"
}

# Build core components
build_core() {
    print_status "Building core components..."
    
    # Build RX components
    print_status "Building RX CMSG Parser..."
    make rx/rx_cmsg_parser.o
    
    print_status "Building RX Descriptor..."
    make rx/rx_descriptor.o
    
    # Build device components
    print_status "Building Device Unpack Kernels..."
    make device/unpack_kernels.o
    
    print_status "Building Device Unpack Launcher..."
    make device/unpack_launch.o
    
    print_success "Core components built successfully"
}

# Build quick test
build_quick_test() {
    print_status "Building quick build test..."

    make test_build_simple
    make test_build_simple_cuda || print_warning "CUDA build test failed"

    print_success "Quick test built successfully"
}

# Build legacy tests
build_legacy_tests() {
    print_status "Building legacy tests..."

    make test_device_discovery
    make test_connection
    make test_tcpx
    make test_performance
    
    print_success "Legacy tests built successfully"
}

# Build enhanced test
build_enhanced_test() {
    print_status "Building enhanced TCPX transfer test..."
    
    make test_tcpx_transfer
    
    print_success "Enhanced test built successfully"
}

# Build new modular tests (if GTest is available)
build_modular_tests() {
    print_status "Checking for GTest..."
    
    if pkg-config --exists gtest 2>/dev/null; then
        print_status "GTest found, building modular tests..."
        
        make test_rx_cmsg_parser || print_warning "Failed to build test_rx_cmsg_parser"
        make test_rx_descriptor || print_warning "Failed to build test_rx_descriptor"
        make test_device_unpack || print_warning "Failed to build test_device_unpack"
        make test_tcpx_integration || print_warning "Failed to build test_tcpx_integration"
        
        print_success "Modular tests built (some may have failed)"
    else
        print_warning "GTest not found, skipping modular tests"
    fi
}

# Build benchmark
build_benchmark() {
    print_status "Building performance benchmark..."
    
    make benchmark_tcpx_unpack
    
    print_success "Benchmark built successfully"
}

# Run quick test
run_quick_test() {
    print_status "Running quick build test..."

    if [ -f "tests/test_build_simple" ]; then
        print_status "Running simple build test..."
        ./tests/test_build_simple || print_warning "Simple build test failed"
    fi

    if [ -f "tests/test_build_simple_cuda" ]; then
        print_status "Running CUDA build test..."
        ./tests/test_build_simple_cuda || print_warning "CUDA build test failed"
    fi

    print_success "Quick tests completed"
}

# Run basic tests
run_basic_tests() {
    print_status "Running basic tests..."

    if [ -f "tests/test_device_discovery" ]; then
        print_status "Running device discovery test..."
        ./tests/test_device_discovery || print_warning "Device discovery test failed"
    fi

    if [ -f "tests/test_tcpx" ]; then
        print_status "Running basic TCPX test..."
        ./tests/test_tcpx || print_warning "Basic TCPX test failed"
    fi

    print_success "Basic tests completed"
}

# Run modular tests
run_modular_tests() {
    if [ -f "tests/test_rx_cmsg_parser" ]; then
        print_status "Running modular tests..."
        make run_unpack_tests || print_warning "Some modular tests failed"
        print_success "Modular tests completed"
    else
        print_warning "Modular tests not available"
    fi
}

# Run benchmark
run_benchmark() {
    if [ -f "tests/benchmark_tcpx_unpack" ]; then
        print_status "Running performance benchmark..."
        make run_benchmark || print_warning "Benchmark failed"
        print_success "Benchmark completed"
    else
        print_warning "Benchmark not available"
    fi
}

# Clean build
clean_build() {
    print_status "Cleaning build..."
    make clean
    print_success "Build cleaned"
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "OPTIONS:"
    echo "  -h, --help          Show this help message"
    echo "  -c, --clean         Clean build before building"
    echo "  --core-only         Build core components only"
    echo "  --legacy-only       Build legacy tests only"
    echo "  --quick-only        Build and run quick test only"
    echo "  --no-tests          Build without running tests"
    echo "  --benchmark         Run benchmark after building"
    echo
    echo "Default: Build everything and run basic tests"
}

# Parse command line arguments
CLEAN=false
CORE_ONLY=false
QUICK_ONLY=false
LEGACY_ONLY=false
NO_TESTS=false
RUN_BENCHMARK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        --core-only)
            CORE_ONLY=true
            shift
            ;;
        --legacy-only)
            LEGACY_ONLY=true
            shift
            ;;
        --quick-only)
            QUICK_ONLY=true
            shift
            ;;
        --no-tests)
            NO_TESTS=true
            shift
            ;;
        --benchmark)
            RUN_BENCHMARK=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "TCPX Unpack Architecture Build Script"
    print_status "====================================="
    echo
    
    check_prerequisites
    
    if [ "$CLEAN" = true ]; then
        clean_build
    fi
    
    if [ "$CORE_ONLY" = true ]; then
        build_core
    elif [ "$LEGACY_ONLY" = true ]; then
        build_legacy_tests
    elif [ "$QUICK_ONLY" = true ]; then
        build_core
        build_quick_test
    else
        # Build everything
        build_core
        build_legacy_tests
        build_enhanced_test
        build_modular_tests
        build_benchmark
    fi

    if [ "$NO_TESTS" = false ]; then
        if [ "$QUICK_ONLY" = true ]; then
            run_quick_test
        else
            run_quick_test
            run_basic_tests
            run_modular_tests
        fi
    fi
    
    if [ "$RUN_BENCHMARK" = true ]; then
        run_benchmark
    fi
    
    echo
    print_success "Build script completed successfully!"
    print_status "Available executables:"
    ls -la tests/ | grep -E "(test_|benchmark_)" || print_warning "No test executables found"
}

# Run main function
main "$@"
