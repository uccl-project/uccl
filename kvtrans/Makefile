# Makefile for KVTrans Engine pybind11 project

# Compiler and flags
CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC -I../include \
	-Wno-pointer-arith -Wno-sign-compare -Wno-unused-variable -Wno-comment

# Python and pybind11 configuration
PYTHON := python3
PYTHON_CONFIG := python3-config
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags)

# Target and source files
TARGET := kvtrans_engine$(shell $(PYTHON_CONFIG) --extension-suffix)
SOURCES := engine.cc pybind_engine.cc
OBJECTS := $(SOURCES:.cc=.o)

# Default target
all: $(TARGET)

# Build the shared library
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(PYTHON_LDFLAGS)

# Compile source files
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

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