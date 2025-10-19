# Makefile for UCCL P2P Engine pybind11 project

# Compiler and flags
CUDA_HOME ?= /usr/local/cuda
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64
RDMA_HOME := ../collective/rdma
CXX := g++
CXXFLAGS := -O3 -shared -std=c++17 -fPIC -I../include -I$(RDMA_HOME) -I$(CUDA_INC) \
	-Wno-pointer-arith -Wno-sign-compare -Wno-unused-variable \
	-Wl,-rpath=/usr/lib/x86_64-linux-gnu -lglog -lgflags -lgtest -lz -lelf -libverbs -lpthread

# TCPX optional integration
USE_TCPX ?= 0
TCPX_DIR := tcpx
ifeq ($(USE_TCPX),1)
  CXXFLAGS += -DUSE_TCPX -I$(TCPX_DIR)/include
endif
# link core
TCPX_CORE_OBJS := $(TCPX_DIR)/device/unpack_kernels.o $(TCPX_DIR)/device/unpack_launch.o \
                  $(TCPX_DIR)/src/bootstrap.o $(TCPX_DIR)/src/channel_manager.o \
                  $(TCPX_DIR)/src/transfer_flow.o $(TCPX_DIR)/src/session_manager.o \
                  $(TCPX_DIR)/src/transfer_manager.o

# Python and pybind11 configuration
PYTHON          ?= python3
PYTHON_CONFIG    = $(PYTHON)-config
PYEXT           := $(shell $(PYTHON_CONFIG) --extension-suffix)
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags)

PYTHON_LIB := $(shell $(PYTHON_CONFIG) --ldflags | grep -o 'python[0-9.]*' | head -1 | sed 's/^/-l/')

# Installation path
PYTHON_SITE_PACKAGES := $(shell $(PYTHON) -c "import site; print(site.getsitepackages()[0])")
INSTALL_DIR          := $(PYTHON_SITE_PACKAGES)/uccl
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCDIR ?= $(PREFIX)/include

LDFLAGS         = -L$(CUDA_LIB) -lcudart -lcuda \
                  -Wl,-rpath,$(CUDA_LIB) -lglog -lgflags -lgtest \
                  -lz -lelf -libverbs -lpthread

# Target and source files
TARGET   := p2p$(PYEXT)
PLUGIN_SO := libuccl_engine.so
CAPI_SOURCE := uccl_engine.cc
CAPI_HEADER := uccl_engine.h
CAPI_OBJECT := $(CAPI_SOURCE:.cc=.o)
SOURCES := engine.cc pybind_engine.cc
CORE_OBJECT := engine.o
OBJECTS := $(SOURCES:.cc=.o)

ifeq ($(USE_TCPX),1)
OBJECTS += $(TCPX_CORE_OBJS)
endif

# Default target
ifeq ($(USE_TCPX),1)
all: tcpx_core $(TARGET) $(PLUGIN_SO)
else
all: $(TARGET) $(PLUGIN_SO)
endif

# Build the shared libraries
$(TARGET): $(OBJECTS) $(RDMA_HOME)/librdma.a
	$(CXX) $(OBJECTS) $(RDMA_HOME)/librdma.a \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB)

$(PLUGIN_SO): $(CAPI_OBJECT) $(CORE_OBJECT) $(RDMA_HOME)/librdma_plugin.a $(TCPX_CORE_OBJS)
	$(CXX) $(CAPI_OBJECT) $(CORE_OBJECT) $(RDMA_HOME)/librdma_plugin.a $(TCPX_CORE_OBJS) \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB) $(PYTHON_LIB) -libverbs -lz -lelf


$(RDMA_HOME)/librdma.a: $(filter-out $(RDMA_HOME)/nccl_plugin.cc, $(wildcard $(RDMA_HOME)/*.cc)) $(RDMA_HOME)/*.h
	make -C $(RDMA_HOME) librdma.a -j$(nproc)

$(RDMA_HOME)/librdma_plugin.a: $(filter-out $(RDMA_HOME)/nccl_plugin.cc, $(wildcard $(RDMA_HOME)/*.cc)) $(RDMA_HOME)/*.h
	make CXXFLAGS="-DDISABLE_CALL_ONCE_STATIC" -C $(RDMA_HOME) librdma_plugin.a

# Compile source files
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -c $< -o $@

# TCPX build
tcpx_core:
	$(MAKE) -C $(TCPX_DIR) core -j$(nproc)
	
# Install the module
install: $(TARGET) $(PLUGIN_SO)
	@mkdir -p $(INSTALL_DIR)
	@cp $(TARGET) $(INSTALL_DIR)/
	@echo "Installation complete. Module installed as: $(INSTALL_DIR)/$(TARGET)"
	install -m 755 $(PLUGIN_SO) $(LIBDIR)/
	install -m 755 $(RDMA_HOME)/librdma_plugin.a $(LIBDIR)/
	install -m 644 $(CAPI_HEADER) $(INCDIR)/

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(CORE_OBJECT) $(CAPI_OBJECT) $(TARGET) $(PLUGIN_SO)
	make -C $(RDMA_HOME) clean -j$(nproc)
ifeq ($(USE_TCPX),1)
	make -C $(TCPX_DIR) clean -j$(nproc)
endif

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

.PHONY: all clean test install-deps help install tcpx_core
