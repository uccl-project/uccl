#!/bin/bash
#************************************************************************
# Copyright (c) 2024, UCCL Project. All rights reserved.
#
# Build and test script for TCPX Unpack Architecture
#************************************************************************

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_TYPE=${BUILD_TYPE:-Release}
BUILD_DIR=${BUILD_DIR:-build}
INSTALL_PREFIX=${INSTALL_PREFIX:-/usr/local}
CUDA_ARCH=${CUDA_ARCH:-sm_70}
NUM_JOBS=${NUM_JOBS:-$(nproc)}

# Function to print colored output
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

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check CMake
    if ! command -v cmake &> /dev/null; then
        print_error "CMake not found. Please install CMake 3.18 or later."
        exit 1
    fi
    
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    print_status "Found CMake version: $CMAKE_VERSION"
    
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
            nvidia-smi -L | head -n1
        fi
    else
        print_warning "nvidia-smi not found. Cannot detect GPUs."
    fi
    
    # Check GTest
    if ! pkg-config --exists gtest; then
        print_warning "GTest not found via pkg-config. Will try to find via CMake."
    else
        GTEST_VERSION=$(pkg-config --modversion gtest)
        print_status "Found GTest version: $GTEST_VERSION"
    fi
    
    print_success "Prerequisites check completed"
}

# Function to clean build directory
clean_build() {
    if [ -d "$BUILD_DIR" ]; then
        print_status "Cleaning existing build directory..."
        rm -rf "$BUILD_DIR"
    fi
}

# Function to configure build
configure_build() {
    print_status "Configuring build..."
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    cmake .. \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_VERBOSE_MAKEFILE=OFF
    
    cd ..
    print_success "Build configured successfully"
}

# Function to build project
build_project() {
    print_status "Building project with $NUM_JOBS parallel jobs..."
    
    cd "$BUILD_DIR"
    make -j"$NUM_JOBS"
    cd ..
    
    print_success "Build completed successfully"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    cd "$BUILD_DIR"
    
    # Run individual test suites
    echo
    print_status "Running RX CMSG Parser tests..."
    if ./test_rx_cmsg_parser; then
        print_success "RX CMSG Parser tests passed"
    else
        print_error "RX CMSG Parser tests failed"
        return 1
    fi
    
    echo
    print_status "Running RX Descriptor tests..."
    if ./test_rx_descriptor; then
        print_success "RX Descriptor tests passed"
    else
        print_error "RX Descriptor tests failed"
        return 1
    fi
    
    echo
    print_status "Running Device Unpack tests..."
    if ./test_device_unpack; then
        print_success "Device Unpack tests passed"
    else
        print_error "Device Unpack tests failed"
        return 1
    fi
    
    echo
    print_status "Running Integration tests..."
    if ./test_tcpx_integration; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed"
        return 1
    fi
    
    cd ..
    print_success "All tests passed!"
}

# Function to run benchmarks
run_benchmarks() {
    print_status "Running performance benchmarks..."
    
    cd "$BUILD_DIR"
    
    if [ -f "./benchmark_tcpx_unpack" ]; then
        echo
        print_status "Running TCPX Unpack benchmark..."
        ./benchmark_tcpx_unpack
        print_success "Benchmark completed"
    else
        print_warning "Benchmark executable not found"
    fi
    
    cd ..
}

# Function to install project
install_project() {
    print_status "Installing project..."
    
    cd "$BUILD_DIR"
    sudo make install
    cd ..
    
    print_success "Installation completed"
}

# Function to generate documentation
generate_docs() {
    if command -v doxygen &> /dev/null; then
        print_status "Generating documentation..."
        cd "$BUILD_DIR"
        make doc
        cd ..
        print_success "Documentation generated in $BUILD_DIR/html/"
    else
        print_warning "Doxygen not found. Skipping documentation generation."
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [TARGETS]"
    echo
    echo "OPTIONS:"
    echo "  -h, --help          Show this help message"
    echo "  -c, --clean         Clean build directory before building"
    echo "  -t, --build-type    Build type (Debug|Release|RelWithDebInfo) [default: Release]"
    echo "  -j, --jobs          Number of parallel jobs [default: $(nproc)]"
    echo "  -a, --arch          CUDA architecture [default: sm_70]"
    echo "  --install-prefix    Installation prefix [default: /usr/local]"
    echo
    echo "TARGETS:"
    echo "  configure           Configure build only"
    echo "  build               Build project only"
    echo "  test                Run tests only"
    echo "  benchmark           Run benchmarks only"
    echo "  install             Install project"
    echo "  docs                Generate documentation"
    echo "  all                 Configure, build, and test (default)"
    echo
    echo "EXAMPLES:"
    echo "  $0                  # Configure, build, and test"
    echo "  $0 -c -t Debug      # Clean build with debug configuration"
    echo "  $0 build test       # Build and test only"
    echo "  $0 benchmark        # Run benchmarks only"
}

# Parse command line arguments
CLEAN=false
TARGETS=()

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
        -t|--build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -j|--jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        -a|--arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --install-prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        configure|build|test|benchmark|install|docs|all)
            TARGETS+=("$1")
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Default target if none specified
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=("all")
fi

# Main execution
main() {
    print_status "TCPX Unpack Architecture Build Script"
    print_status "======================================"
    print_status "Build type: $BUILD_TYPE"
    print_status "CUDA arch: $CUDA_ARCH"
    print_status "Parallel jobs: $NUM_JOBS"
    print_status "Install prefix: $INSTALL_PREFIX"
    echo
    
    check_prerequisites
    
    if [ "$CLEAN" = true ]; then
        clean_build
    fi
    
    for target in "${TARGETS[@]}"; do
        case $target in
            configure)
                configure_build
                ;;
            build)
                if [ ! -f "$BUILD_DIR/Makefile" ]; then
                    configure_build
                fi
                build_project
                ;;
            test)
                if [ ! -f "$BUILD_DIR/test_rx_cmsg_parser" ]; then
                    if [ ! -f "$BUILD_DIR/Makefile" ]; then
                        configure_build
                    fi
                    build_project
                fi
                run_tests
                ;;
            benchmark)
                if [ ! -f "$BUILD_DIR/benchmark_tcpx_unpack" ]; then
                    if [ ! -f "$BUILD_DIR/Makefile" ]; then
                        configure_build
                    fi
                    build_project
                fi
                run_benchmarks
                ;;
            install)
                if [ ! -f "$BUILD_DIR/Makefile" ]; then
                    configure_build
                    build_project
                fi
                install_project
                ;;
            docs)
                if [ ! -f "$BUILD_DIR/Makefile" ]; then
                    configure_build
                fi
                generate_docs
                ;;
            all)
                configure_build
                build_project
                run_tests
                ;;
            *)
                print_error "Unknown target: $target"
                exit 1
                ;;
        esac
    done
    
    echo
    print_success "All operations completed successfully!"
}

# Run main function
main "$@"
