# Makefile for UCCL P2P Engine pybind11 project

# TCPX optional integration
USE_TCPX ?= $(shell echo $${USE_TCPX:-0})
# EFA optional integration
USE_EFA ?= $(shell echo $${USE_EFA:-0})

# TCP endpoint integration
USE_TCP ?= $(shell echo $${USE_TCP:-0})

# Compiler and flags
CUDA_HOME ?= /usr/local/cuda
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64
EFA_HOME ?= /opt/amazon/efa

# Building with the following settings:
INC = -I./ -I$(CUDA_HOME)/include
LIBS = -L ${CUDA_HOME}/lib64 -libverbs -lcudart -lcuda -lpthread -lglog -lgflags -lgtest -lz -lelf
LIBS2 =

CXX := g++
CXXFLAGS := -O3 -shared -std=c++17 -fPIC -I. -I./include -I../include -I$(CUDA_INC) \
	-Wno-pointer-arith -Wno-sign-compare -Wno-unused-variable \
	-Wl,-rpath=/usr/lib/x86_64-linux-gnu

ifeq ($(USE_TCPX),1)
	CXXFLAGS += -DUCCL_P2P_USE_TCPX
	LIBS2 += -lglog -lgflags -lpthread -ldl
else ifeq ($(USE_EFA),1)
	CXXFLAGS += -DUCCL_P2P_USE_EFA
	INC += -I${EFA_HOME}/include
	LIBS += -L ${EFA_HOME}/lib -lefa
else
	LIBS2 += -lglog -lgflags -lgtest -lz -lelf -lpthread -libverbs
endif

# Add NCCL-over-TCP endpoint flag
ifeq ($(USE_TCP),1)
	CXXFLAGS += -DUCCL_P2P_USE_NCCL
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

LDFLAGS         = -L$(CUDA_LIB) -lcudart -lcuda -Wl,-rpath,$(CUDA_LIB) $(LIBS2)

ifeq ($(USE_TCP),1)
NCCL_HOME ?= ../thirdparty/nccl/build
NCCL_INC  ?= $(NCCL_HOME)/include
NCCL_LIB  ?= $(NCCL_HOME)/lib
CXXFLAGS  += -I$(NCCL_INC)
LDFLAGS   += -L$(NCCL_LIB) -lnccl -Wl,-rpath,$(NCCL_LIB)
endif

# Target and source files
P2P_PYTHON_EXT := p2p$(PYEXT)
P2P_SHARED_LIB := libuccl_p2p.so
CAPI_SOURCE := uccl_engine.cc
CAPI_HEADER := uccl_engine.h include/common.h
CAPI_OBJECT := $(CAPI_SOURCE:.cc=.o)
ifeq ($(USE_TCPX),1)
NCCL_HOME ?= ../thirdparty/nccl/build
NCCL_INC  ?= $(NCCL_HOME)/include
NCCL_LIB  ?= $(NCCL_HOME)/lib
P2P_SHARED_LIB := libuccl_p2p.so
SOURCES := nccl_tcpx_endpoint.cc
CORE_OBJECTS := $(SOURCES:.cc=.o)
CXXFLAGS += -DUCCL_P2P_USE_TCPX -I$(NCCL_INC)
LDFLAGS  += -L$(NCCL_LIB) -lnccl -Wl,-rpath,$(NCCL_LIB)
OBJECTS := $(CORE_OBJECTS)
else ifeq ($(USE_TCP),1)
SOURCES := engine.cc engine_pybind.cc nccl/nccl_endpoint.cc
CORE_OBJECT := engine.o nccl/nccl_endpoint.o
else
SOURCES := engine.cc engine_pybind.cc
CORE_OBJECT := engine.o
endif

ifeq ($(USE_TCPX),1)
OBJECTS := $(CORE_OBJECTS)
else
OBJECTS := $(SOURCES:.cc=.o)
endif

# Transport library configuration (comma-separated)
TRANSPORTS ?= rdma
TRANSPORT_LIBS :=
ifneq (,$(findstring rdma,$(TRANSPORTS)))
    TRANSPORT_LIBS += libuccl_p2p_rdma.so
endif
ifneq (,$(findstring tcp,$(TRANSPORTS)))
    TRANSPORT_LIBS += libuccl_p2p_tcp.so
endif
ifneq (,$(findstring tcpx,$(TRANSPORTS)))
    TRANSPORT_LIBS += libuccl_p2p_tcpx.so
endif
ifneq (,$(findstring efa,$(TRANSPORTS)))
    TRANSPORT_LIBS += libuccl_p2p_efa.so
endif

# Build the shared libraries
ifeq ($(USE_TCPX),1)
# Default target
all: $(P2P_SHARED_LIB)

# only build plugin for tcpx now, TODO: add tcpx to p2p	  
$(P2P_SHARED_LIB): $(CAPI_OBJECT) $(CORE_OBJECTS) $(CU_OBJECTS)
	$(CXX) $(CAPI_OBJECT) $(CORE_OBJECTS) $(CU_OBJECTS) \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB) $(PYTHON_LIB)
else
# Default target
all: $(P2P_PYTHON_EXT) $(P2P_SHARED_LIB) $(TRANSPORT_LIBS)

$(P2P_PYTHON_EXT): $(OBJECTS)
	$(CXX) $(OBJECTS) \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB) $(INC) $(LIBS)

$(P2P_SHARED_LIB): $(CAPI_OBJECT) $(CORE_OBJECT)
	$(CXX) $(CAPI_OBJECT) $(CORE_OBJECT) \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB) $(PYTHON_LIB) $(INC) $(LIBS)
endif

# Compile source files
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -c $< -o $@

# Compile NCCL-over-TCP source files
nccl/%.o: nccl/%.cc
	@mkdir -p nccl
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -I. -c $< -o $@

# Transport library build rules

# RDMA transport (default, no flags needed)
libuccl_p2p_rdma.so: engine_rdma.o rdma_factory.o
	$(CXX) -shared $^ -o $@ -L$(CUDA_LIB) -lcudart -lcuda -libverbs -lglog -lgflags -lpthread -Wl,-rpath,$(CUDA_LIB)

engine_rdma.o: engine.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -I. -c $< -o $@

rdma_factory.o: rdma_factory.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -I. -c $< -o $@

# TCP transport (with NCCL flag)
libuccl_p2p_tcp.so: engine_tcp.o nccl/nccl_endpoint_tcp.o tcp_factory.o
	$(CXX) -shared $^ -o $@ -L$(CUDA_LIB) -lcudart -lcuda -L$(NCCL_LIB) -lnccl -lglog -lgflags -lpthread -Wl,-rpath,$(CUDA_LIB):$(NCCL_LIB)

engine_tcp.o: engine.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -DUCCL_P2P_USE_NCCL -I$(NCCL_INC) -I. -c $< -o $@

nccl/nccl_endpoint_tcp.o: nccl/nccl_endpoint.cc
	@mkdir -p nccl
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -DUCCL_P2P_USE_NCCL -I$(NCCL_INC) -I. -c $< -o $@

tcp_factory.o: tcp_factory.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -DUCCL_P2P_USE_NCCL -I$(NCCL_INC) -I. -c $< -o $@

# TCPX transport
libuccl_p2p_tcpx.so: nccl_tcpx_endpoint_lib.o tcpx_factory.o
	$(CXX) -shared $^ -o $@ -L$(CUDA_LIB) -lcudart -lcuda -L$(NCCL_LIB) -lnccl -lglog -lgflags -lpthread -Wl,-rpath,$(CUDA_LIB):$(NCCL_LIB)

nccl_tcpx_endpoint_lib.o: nccl_tcpx_endpoint.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -DUCCL_P2P_USE_TCPX -I$(NCCL_INC) -I. -c $< -o $@

tcpx_factory.o: tcpx_factory.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -DUCCL_P2P_USE_TCPX -I$(NCCL_INC) -I. -c $< -o $@

# EFA transport (AWS Elastic Fabric Adapter)
libuccl_p2p_efa.so: engine_efa.o efa_factory.o
	$(CXX) -shared $^ -o $@ -L$(CUDA_LIB) -lcudart -lcuda -L$(EFA_HOME)/lib -lefa -libverbs -lglog -lgflags -lpthread -Wl,-rpath,$(CUDA_LIB):$(EFA_HOME)/lib

engine_efa.o: engine.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -DUCCL_P2P_USE_EFA -I$(EFA_HOME)/include -I. -c $< -o $@

efa_factory.o: efa_factory.cc
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -DUCCL_P2P_USE_EFA -I$(EFA_HOME)/include -I. -c $< -o $@

ifeq ($(USE_TCPX),1)
install: $(P2P_SHARED_LIB)
	install -m 755 $(P2P_SHARED_LIB) $(LIBDIR)/
	install -m 644 $(CAPI_HEADER) $(INCDIR)/
else
install: $(P2P_PYTHON_EXT) $(P2P_SHARED_LIB) $(TRANSPORT_LIBS)
	$(MAKE) $(P2P_PYTHON_EXT)
	@mkdir -p $(INSTALL_DIR)
	@cp $(P2P_PYTHON_EXT) $(INSTALL_DIR)/
	@echo "Installation complete. Module installed as: $(INSTALL_DIR)/$(P2P_PYTHON_EXT)"
	install -m 755 $(P2P_SHARED_LIB) $(LIBDIR)/
	install -m 755 $(TRANSPORT_LIBS) $(LIBDIR)/
	install -m 644 $(CAPI_HEADER) $(INCDIR)/
	@echo "Installed transport libraries: $(TRANSPORT_LIBS)"
endif
# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(CAPI_OBJECT) $(P2P_SHARED_LIB)
	rm -f nccl/*.o
	rm -f engine_rdma.o engine_tcp.o engine_efa.o
	rm -f rdma_factory.o tcp_factory.o tcpx_factory.o efa_factory.o
	rm -f nccl/nccl_endpoint_tcp.o nccl_tcpx_endpoint_lib.o
	rm -f $(TRANSPORT_LIBS)
ifneq ($(USE_TCPX),1)
	rm -f $(P2P_PYTHON_EXT)
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
	@echo "  all          - Build the pybind11 module and transport libraries"
	@echo "  install      - Install the module and libraries to system"
	@echo "  clean        - Remove build artifacts"
	@echo "  test         - Run the test script"
	@echo "  install-deps - Install pybind11 dependency"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Build options:"
	@echo "  TRANSPORTS   - Comma-separated transports to build (default: rdma,tcp,tcpx,efa)"
	@echo "                 Examples:"
	@echo "                   make TRANSPORTS=rdma"
	@echo "                   make TRANSPORTS=rdma,tcp"
	@echo "                   make TRANSPORTS=efa"
	@echo "                   make TRANSPORTS=rdma,efa"
	@echo ""
	@echo "Runtime configuration:"
	@echo "  UCCL_P2P_TRANSPORT - Select transport at runtime: rdma | tcp | tcpx | efa"
	@echo "                       Default: rdma"
	@echo "                       Example: export UCCL_P2P_TRANSPORT=tcp"
	@echo "                       Example: export UCCL_P2P_TRANSPORT=efa"

.PHONY: all clean test install-deps help install
