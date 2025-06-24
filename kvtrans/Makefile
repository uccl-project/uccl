# Makefile for KVTrans Engine pybind11 project

# Compiler and flags
CUDA_HOME ?= /usr/local/cuda
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64
CXX := g++
CXXFLAGS := -O3 -shared -std=c++17 -fPIC -I../include -I../rdma -I$(CUDA_INC) \
	-Wno-pointer-arith -Wno-sign-compare -Wno-unused-variable \
	-Wl,-rpath=/usr/lib/x86_64-linux-gnu -lglog -lgflags -lgtest -lz -lelf -libverbs -lpthread \
	-DLAZY_CREATE_ENGINE

# Python and pybind11 configuration
PYTHON := python3
PYTHON_CONFIG := python3-config
PYEXT           := $(shell $(PYTHON_CONFIG) --extension-suffix)
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags)

# Target and source files
TARGET   := kvtrans_engine$(PYEXT)
SOURCES := engine.cc pybind_engine.cc
OBJECTS := $(SOURCES:.cc=.o)

# Default target
all: $(TARGET)

# Build the shared library
$(TARGET): $(OBJECTS) ../rdma/librdma.a
	$(CXX) $(OBJECTS) ../rdma/librdma.a \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB)


../rdma/librdma.a: ../rdma/*.cc ../rdma/*.h
	make CXXFLAGS=-DLAZY_CREATE_ENGINE -C ../rdma

# Compile source files
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)
	make -C ../rdma clean

# Test the module
test: $(TARGET)
	$(PYTHON) test_engine.py

# Install pybind11 if not available
install-deps:
	pip3 install pybind11

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the pybind11 module"
	@echo "  clean        - Remove build artifacts"
	@echo "  test         - Run the test script"
	@echo "  install-deps - Install pybind11 dependency"
	@echo "  help         - Show this help message"

.PHONY: all clean test install-deps help 