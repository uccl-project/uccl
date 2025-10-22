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
USE_TCPX ?= $(shell echo $${USE_TCPX:-0})
TCPX_DIR := tcpx

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

LDFLAGS = -L$(CUDA_LIB) -lcudart -lcuda \
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

# TCPX Integration (auto-detect .cc + .cu)
ifeq ($(USE_TCPX),1)
  CXXFLAGS += -DUSE_TCPX -I$(TCPX_DIR)/include -I$(TCPX_DIR)
  $(info 🔧 Building with TCPX support)

  TCPX_CC_SRCS := $(wildcard $(TCPX_DIR)/src/*.cc)
  TCPX_CU_SRCS := $(wildcard $(TCPX_DIR)/device/*.cu)
  TCPX_OBJS := $(patsubst %.cc,%.o,$(TCPX_CC_SRCS)) $(patsubst %.cu,%.o,$(TCPX_CU_SRCS))
  OBJECTS  += $(TCPX_OBJS)

  $(TCPX_DIR)/src/%.o: $(TCPX_DIR)/src/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

  $(TCPX_DIR)/device/%.o: $(TCPX_DIR)/device/%.cu
	nvcc -c $< -o $@ -Xcompiler -fPIC -O3
endif

# Default target
ifeq ($(USE_TCPX),1)
all: $(TARGET) $(PLUGIN_SO)
else
all: $(TARGET) $(PLUGIN_SO)
endif

# Build the shared libraries
$(TARGET): $(OBJECTS) $(RDMA_HOME)/librdma.a
	$(CXX) $(OBJECTS) $(RDMA_HOME)/librdma.a \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB)

$(PLUGIN_SO): $(CAPI_OBJECT) $(CORE_OBJECT) $(RDMA_HOME)/librdma_plugin.a $(TCPX_OBJS)
	$(CXX) $(CAPI_OBJECT) $(CORE_OBJECT) $(RDMA_HOME)/librdma_plugin.a $(TCPX_OBJS) \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB) $(PYTHON_LIB) -libverbs -lz -lelf

$(RDMA_HOME)/librdma.a:
	make -C $(RDMA_HOME) librdma.a -j$(nproc)

$(RDMA_HOME)/librdma_plugin.a:
	make CXXFLAGS="-DDISABLE_CALL_ONCE_STATIC" -C $(RDMA_HOME) librdma_plugin.a

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -c $< -o $@

# Install
install: $(TARGET) $(PLUGIN_SO)
	@mkdir -p $(INSTALL_DIR)
	@cp $(TARGET) $(INSTALL_DIR)/
	@echo "Installed: $(INSTALL_DIR)/$(TARGET)"
	install -m 755 $(PLUGIN_SO) $(LIBDIR)/
	install -m 755 $(RDMA_HOME)/librdma_plugin.a $(LIBDIR)/
	install -m 644 $(CAPI_HEADER) $(INCDIR)/

clean:
	rm -f $(OBJECTS) $(CORE_OBJECT) $(CAPI_OBJECT) $(TARGET) $(PLUGIN_SO)
	make -C $(RDMA_HOME) clean -j$(nproc)
ifeq ($(USE_TCPX),1)
	make -C $(TCPX_DIR) clean -j$(nproc)
endif
