# Makefile for UCCL P2P Engine pybind11 project

# TCPX optional integration
USE_TCPX ?= $(shell echo $${USE_TCPX:-0})

# Compiler and flags
CUDA_HOME ?= /usr/local/cuda
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64
RDMA_HOME := ../collective/rdma
CXX := g++
CXXFLAGS := -O3 -shared -std=c++17 -fPIC -I../include -I$(RDMA_HOME) -I$(CUDA_INC) \
	-Wno-pointer-arith -Wno-sign-compare -Wno-unused-variable \
	-Wl,-rpath=/usr/lib/x86_64-linux-gnu

ifeq ($(USE_TCPX),1)
	CXXFLAGS += -DUSE_TCPX
	LINK_LIBS := -lglog -lgflags -lpthread -ldl
else
	CXXFLAGS += -libverbs
	LINK_LIBS := -lglog -lgflags -lgtest -lz -lelf -lpthread -libverbs
endif

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

LDFLAGS         = -L$(CUDA_LIB) -lcudart -lcuda -Wl,-rpath,$(CUDA_LIB) $(LINK_LIBS)

# Target and source files
P2P_PYTHON_EXT := p2p$(PYEXT)
P2P_SHARED_LIB := libuccl_p2p.so
RDMA_PLUGIN_LIB := librdma_plugin.a
CAPI_SOURCE := uccl_engine.cc
CAPI_HEADER := uccl_engine.h
CAPI_OBJECT := $(CAPI_SOURCE:.cc=.o)
ifeq ($(USE_TCPX),1)
NCCL_HOME ?= ../thirdparty/nccl/build
NCCL_INC  ?= $(NCCL_HOME)/include
NCCL_LIB  ?= $(NCCL_HOME)/lib
ifeq (,$(wildcard $(NCCL_INC)/nccl.h))
$(error NCCL header not found at $(NCCL_INC)/nccl.h; set NCCL_HOME)
endif
ifeq (,$(wildcard $(NCCL_LIB)/libnccl.so*))
$(error NCCL library not found at $(NCCL_LIB)/libnccl.so*; set NCCL_HOME)
endif
SOURCES := nccl_tcpx_endpoint.cc
CORE_OBJECTS := $(SOURCES:.cc=.o)
CXXFLAGS += -I$(NCCL_INC)
LDFLAGS += -L$(NCCL_LIB) -lnccl -Wl,-rpath,$(NCCL_LIB)
else
SOURCES := engine.cc pybind_engine.cc
CORE_OBJECT := engine.o
endif

ifeq ($(USE_TCPX),1)
OBJECTS := $(CORE_OBJECTS)
else
OBJECTS := $(SOURCES:.cc=.o)
endif

# Build the shared libraries
ifeq ($(USE_TCPX),1)
# Default target
all: $(P2P_SHARED_LIB)

# only build plugin for tcpx now, TODO: add tcpx to p2p	  
$(P2P_SHARED_LIB): $(CAPI_OBJECT) $(CORE_OBJECTS)
	$(CXX) $(CAPI_OBJECT) $(CORE_OBJECTS) \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB) $(PYTHON_LIB)
else
# Default target
all: $(P2P_PYTHON_EXT) $(P2P_SHARED_LIB)

$(P2P_PYTHON_EXT): $(OBJECTS) $(RDMA_HOME)/librdma.a
	$(CXX) $(OBJECTS) $(RDMA_HOME)/librdma.a \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB)

$(P2P_SHARED_LIB): $(CAPI_OBJECT) $(CORE_OBJECT) $(RDMA_PLUGIN_LIB)
	$(CXX) $(CAPI_OBJECT) $(CORE_OBJECT) $(RDMA_PLUGIN_LIB) \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB) $(PYTHON_LIB) -libverbs -lz -lelf

$(RDMA_HOME)/librdma.a: $(filter-out $(RDMA_HOME)/nccl_plugin.cc, $(wildcard $(RDMA_HOME)/*.cc)) $(RDMA_HOME)/*.h
	make -C $(RDMA_HOME) librdma.a -j$(nproc)

RDMA_OBJECTS := $(patsubst $(RDMA_HOME)/%.cc,%.o,$(filter-out $(RDMA_HOME)/nccl_plugin.cc,$(wildcard $(RDMA_HOME)/*.cc)))

$(RDMA_PLUGIN_LIB): $(RDMA_OBJECTS)
	ar rcs $@ $(RDMA_OBJECTS)

$(RDMA_OBJECTS): %.o: $(RDMA_HOME)/%.cc
	$(CXX) -c $< -o $@ $(CXXFLAGS) -DDISABLE_CALL_ONCE_STATIC -DUSE_CUDA -I$(RDMA_HOME)
endif

# Compile source files
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -c $< -o $@

# Install the module

ifeq ($(USE_TCPX),1)
install: $(P2P_SHARED_LIB)
	@mkdir -p $(INSTALL_DIR)
	install -m 755 $(P2P_SHARED_LIB) $(LIBDIR)/
	install -m 644 $(CAPI_HEADER) $(INCDIR)/
else
install: $(P2P_PYTHON_EXT) $(P2P_SHARED_LIB) $(RDMA_PLUGIN_LIB)
	@mkdir -p $(INSTALL_DIR)
	@cp $(P2P_PYTHON_EXT) $(INSTALL_DIR)/
	@echo "Installation complete. Module installed as: $(INSTALL_DIR)/$(P2P_PYTHON_EXT)"
	install -m 755 $(P2P_SHARED_LIB) $(LIBDIR)/
	install -m 755 $(RDMA_PLUGIN_LIB) $(LIBDIR)/
	install -m 644 $(CAPI_HEADER) $(INCDIR)/
endif
# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(CAPI_OBJECT) $(P2P_SHARED_LIB)
ifneq ($(USE_TCPX),1)
	rm -f $(P2P_PYTHON_EXT) $(RDMA_PLUGIN_LIB) $(RDMA_OBJECTS)
endif

# Test the module
test:
ifneq ($(USE_TCPX),1)
	$(MAKE) $(P2P_PYTHON_EXT)
	$(PYTHON) tests/test_engine_write.py
endif

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
