# Makefile for UCCL P2P Engine pybind11 project

# Compiler and flags
CUDA_HOME ?= /usr/local/cuda
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64
CXX := g++
CXXFLAGS := -O3 -shared -std=c++17 -fPIC -I../include -I../rdma -I$(CUDA_INC) \
	-Wno-pointer-arith -Wno-sign-compare -Wno-unused-variable \
	-Wl,-rpath=/usr/lib/x86_64-linux-gnu -lglog -lgflags -lgtest -lz -lelf -libverbs -lpthread

# Python and pybind11 configuration
PYTHON          ?= python3
PYTHON_CONFIG    = $(PYTHON)-config
PYEXT           := $(shell $(PYTHON_CONFIG) --extension-suffix)
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags)

# Installation path
PYTHON_SITE_PACKAGES := $(shell $(PYTHON) -c "import site; print(site.getsitepackages()[0])")
INSTALL_DIR          := $(PYTHON_SITE_PACKAGES)/uccl
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCDIR ?= $(PREFIX)/include

# Redis
REDIS_PKGCONFIG := pkg-config
REDIS_CFLAGS    := $(shell $(REDIS_PKGCONFIG) --cflags hiredis redis++)
REDIS_LIBS      := $(shell $(REDIS_PKGCONFIG) --libs hiredis redis++)
CXXFLAGS       += $(REDIS_CFLAGS)
LDFLAGS         = -L$(CUDA_LIB) -lcudart -lcuda $(REDIS_LIBS) \
                  -Wl,-rpath,$(CUDA_LIB) -lglog -lgflags -lgtest \
                  -lz -lelf -libverbs -lpthread

# Target and source files
TARGET   := p2p$(PYEXT)
PLUGIN_SO := libuccl_engine.so
CAPI_SOURCE := uccl_engine.cc
CAPI_HEADER := uccl_engine.h
CAPI_OBJECT := $(CAPI_SOURCE:.cc=.o)
SOURCES := engine.cc pybind_engine.cc
CORE_SOURCE := engine.cc
CORE_OBJECT := engine_plugin.o
OBJECTS := $(SOURCES:.cc=.o)

# Default target
all: $(TARGET) $(PLUGIN_SO)

# Build the shared libraries
$(TARGET): $(OBJECTS) ../rdma/librdma.a
	$(CXX) $(OBJECTS) ../rdma/librdma.a \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB)

$(PLUGIN_SO): $(CAPI_OBJECT) $(CORE_OBJECT) ../rdma/librdma.a
	$(CXX) $(CAPI_OBJECT) $(CORE_OBJECT) ../rdma/librdma.a \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB)


../rdma/librdma.a: $(filter-out ../rdma/nccl_plugin.cc, $(wildcard ../rdma/*.cc)) ../rdma/*.h
	make CXXFLAGS="-DLAZY_CREATE_ENGINE -DDISABLE_CALL_ONCE_STATIC" -C ../rdma librdma.a -j$(nproc)

# Compile source files
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -c $< -o $@

# Compile engine.cc for plugin (without Python flags)
engine_plugin.o: engine.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CAPI source file (without Python flags)
$(CAPI_OBJECT): $(CAPI_SOURCE) $(CAPI_HEADER)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Install the module
install: $(TARGET) $(PLUGIN_SO)
	@mkdir -p $(INSTALL_DIR)
	@cp $(TARGET) $(INSTALL_DIR)/
	@echo "Installation complete. Module installed as: $(INSTALL_DIR)/$(TARGET)"
	install -m 755 $(PLUGIN_SO) $(LIBDIR)/
	install -m 644 $(CAPI_HEADER) $(INCDIR)/
# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(CORE_OBJECT) $(CAPI_OBJECT) $(TARGET) $(PLUGIN_SO)
	make -C ../rdma clean -j$(nproc)

# Test the module
test: $(TARGET)
	$(PYTHON) tests/test_engine_write.py

# Install pybind11 if not available
install-deps:
	pip3 install pybind11

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the pybind11 module"
	@echo "  install      - Install the module to Python site-packages"
	@echo "  clean        - Remove build artifacts"
	@echo "  test         - Run the test script"
	@echo "  install-deps - Install pybind11 dependency"
	@echo "  help         - Show this help message"

.PHONY: all clean test install-deps help install 
