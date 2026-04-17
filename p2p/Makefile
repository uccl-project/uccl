# Makefile for UCCL P2P Engine nanobind project (stable ABI)

# EFA optional integration
USE_EFA ?= $(shell echo $${USE_EFA:-0})

# TCPX optional integration
USE_TCPX ?= $(shell echo $${USE_TCPX:-0})

# TCP optional integration
USE_TCP ?= $(shell echo $${USE_TCP:-0})

# NCCL optional integration
USE_NCCL ?= $(shell echo $${USE_NCCL:-0})

ENABLE_NCCL := 0
ifneq ($(filter 1,$(USE_TCPX) $(USE_TCP) $(USE_NCCL)),)
ENABLE_NCCL := 1
endif

# DietGPU optional integration
USE_DIETGPU ?= $(shell echo $${USE_DIETGPU:-0})

# Compiler and flags
CUDA_HOME ?= /usr/local/cuda
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64
EFA_HOME ?= /opt/amazon/efa

# Building with the following settings:
INC = -I./ -I$(CUDA_HOME)/include
LIBS = -L ${CUDA_HOME}/lib64 -libverbs -lcudart -lcuda -lpthread -lz -lelf
LIBS2 =

CXX := g++
CXXFLAGS := -O3 -shared -std=c++17 -fPIC -I. -I./include -I../include -I$(CUDA_INC) \
	-MMD -MP \
	-Wno-pointer-arith -Wno-sign-compare -Wno-unused-variable \
	-Wl,-rpath=/usr/lib/x86_64-linux-gnu

ifeq ($(USE_EFA),1)
	CXXFLAGS += -DUCCL_P2P_USE_EFA
	INC += -I${EFA_HOME}/include
	LIBS += -L ${EFA_HOME}/lib -lefa
else
	LIBS2 +=  -lz -lelf -lpthread -libverbs -ldl
endif

# Add NCCL-over-TCP/NCCL-over-TCPX endpoint flag
ifeq ($(ENABLE_NCCL),1)
	CXXFLAGS += -DUCCL_P2P_USE_NCCL
endif

# ---------------- DietGPU ----------------
DIETGPU_ROOT      := ../thirdparty/dietgpu
DIETGPU_FLOAT_DIR := $(DIETGPU_ROOT)/dietgpu/float
DIETGPU_LIB       := $(DIETGPU_FLOAT_DIR)/libdietgpu_float.so

ifeq ($(USE_DIETGPU),1)
	DIETGPU_INC  := -I$(DIETGPU_ROOT)
	DIETGPU_LIBS := -L$(DIETGPU_FLOAT_DIR) -ldietgpu_float \
	                -Wl,-rpath,$(abspath $(DIETGPU_FLOAT_DIR)) \
	                -Wl,-rpath,'$$ORIGIN/lib'
	CXXFLAGS += -DUSE_DIETGPU
else
	DIETGPU_INC  :=
	DIETGPU_LIBS :=
endif

CXXFLAGS += $(DIETGPU_INC)

# Python and nanobind configuration
PYTHON          ?= python3
PYTHON_CONFIG    = $(PYTHON)-config
NB_DIR          := $(shell $(PYTHON) -c "import nanobind as _nb, os; print(os.path.dirname(_nb.__file__))")
NB_INCLUDES     := -I$(NB_DIR)/include -I$(NB_DIR)/ext/robin_map/include -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
# Stable ABI requires Python >= 3.12
PY_STABLE_ABI_OK := $(shell $(PYTHON) -c "import sys; print(1 if sys.version_info >= (3, 12) else 0)")
ifeq ($(PY_STABLE_ABI_OK),1)
NB_ABI_FLAGS    := -DPy_LIMITED_API=0x030C0000 -DNB_STABLE_ABI=1
PYEXT           := .abi3.so
else
NB_ABI_FLAGS    :=
PYEXT           := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
endif
NB_SRC_DIR      := $(NB_DIR)/src
NB_SRCS         := nb_internals.cpp nb_func.cpp nb_type.cpp nb_enum.cpp \
                   nb_ndarray.cpp nb_static_property.cpp common.cpp error.cpp \
                   trampoline.cpp implicit.cpp
NB_OBJECTS      := $(addprefix nb_,$(NB_SRCS:.cpp=.o))
PYTHON_LDFLAGS  := $(shell $(PYTHON_CONFIG) --ldflags)
PYTHON_LIB      := $(shell $(PYTHON_CONFIG) --ldflags | grep -o 'python[0-9.]*' | head -1 | sed 's/^/-l/')

# Installation path
PYTHON_SITE_PACKAGES := $(shell $(PYTHON) -c "import site; print(site.getsitepackages()[0])")
INSTALL_DIR          := $(PYTHON_SITE_PACKAGES)/uccl
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCDIR ?= $(PREFIX)/include

LDFLAGS         = -L$(CUDA_LIB) -lcudart -lcuda -Wl,-rpath,$(CUDA_LIB) $(LIBS2) $(DIETGPU_LIBS)

ifeq ($(ENABLE_NCCL),1)
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
SOURCES := engine.cc engine_api.cc
CORE_OBJECT := engine.o
ifeq ($(ENABLE_NCCL),1)
SOURCES += nccl/nccl_endpoint.cc
CORE_OBJECT += nccl/nccl_endpoint.o
endif
OBJECTS := $(SOURCES:.cc=.o)

# Build the shared libraries
# Default target
all: $(P2P_PYTHON_EXT) $(P2P_SHARED_LIB)

$(P2P_PYTHON_EXT): $(OBJECTS) $(NB_OBJECTS)
	$(CXX) $(OBJECTS) $(NB_OBJECTS) \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB) $(INC) $(LIBS)

$(P2P_SHARED_LIB): $(CAPI_OBJECT) $(CORE_OBJECT)
	$(CXX) $(CAPI_OBJECT) $(CORE_OBJECT) \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB) $(PYTHON_LIB) $(INC) $(LIBS)

# Compile nanobind source files for stable ABI
nb_%.o: $(NB_SRC_DIR)/%.cpp
	$(CXX) -O3 -std=c++17 -fPIC -fvisibility=hidden $(NB_ABI_FLAGS) $(NB_INCLUDES) -c $< -o $@

# Compile source files
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(NB_INCLUDES) $(NB_ABI_FLAGS) -c $< -o $@

# Compile NCCL-over-TCP source files
nccl/%.o: nccl/%.cc
	@mkdir -p nccl
	$(CXX) $(CXXFLAGS) $(NB_INCLUDES) $(NB_ABI_FLAGS) -I. -c $< -o $@

# Auto-generated header dependencies
DEPS := $(OBJECTS:.o=.d) $(CAPI_OBJECT:.o=.d)
-include $(DEPS)

define safe_install
	@echo "Installing $(1) to $(2)..."
	@install -m $(3) $(1) $(2)/ || \
		{ echo "Warning: Unable to install $(1) to $(2)/"; \
		  echo "  Make sure $(2) exists and you have write permissions (try sudo make install)"; \
		  exit 0; }
endef

install: $(P2P_PYTHON_EXT)
	$(MAKE) $(P2P_PYTHON_EXT)
	@mkdir -p $(INSTALL_DIR)
	@rm -f $(INSTALL_DIR)/p2p.cpython-*.so $(INSTALL_DIR)/p2p.abi3.so
	@cp $(P2P_PYTHON_EXT) $(INSTALL_DIR)/
	@echo "Installation complete. Module installed as: $(INSTALL_DIR)/$(P2P_PYTHON_EXT)"
	@if [ -f libuccl_p2p_rdma.so ]; then \
		echo "Transport backends detected, installing dispatcher..."; \
		$(MAKE) install-all-transport; \
	else \
		echo "No transport backends, installing monolithic libuccl_p2p.so..."; \
		$(MAKE) $(P2P_SHARED_LIB); \
		install -m 755 $(P2P_SHARED_LIB) $(LIBDIR)/; \
	fi
	$(call safe_install,$(CAPI_HEADER),$(INCDIR),644)
	@echo "Installation complete!"
# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(CAPI_OBJECT) $(NB_OBJECTS)
	rm -f $(DEPS)
	rm -f nccl/*.o nccl/*.d libuccl_p2p*.so
	rm -f $(P2P_PYTHON_EXT)

# Test the module
test:
	$(MAKE) $(P2P_PYTHON_EXT)
	$(PYTHON) tests/test_engine_write.py

# Install nanobind if not available
install-deps:
	pip3 install nanobind

# ---------------------------------------------------------------------------
# Runtime transport selection: build all backends + thin dispatcher
# ---------------------------------------------------------------------------
DISPATCH_SRC := uccl_p2p_dispatch.cc
DISPATCH_OBJ := uccl_p2p_dispatch.o

# Auto-detect which optional backends can be built
HAS_EFA  := $(shell test -f $(EFA_HOME)/include/infiniband/efadv.h && echo 1 || echo 0)
HAS_NCCL := $(shell echo '\#include <nccl.h>' | $(CXX) -x c++ -E - >/dev/null 2>&1 && echo 1 || echo 0)

TRANSPORT_TARGETS := dispatch-lib
ifeq ($(HAS_EFA),1)
TRANSPORT_TARGETS += transport-efa
endif
ifeq ($(HAS_NCCL),1)
TRANSPORT_TARGETS += transport-nccl
endif
# RDMA last so its .o files remain for the default make install
TRANSPORT_TARGETS += transport-rdma

all-transport: $(TRANSPORT_TARGETS)
	@echo "Built transports: $(TRANSPORT_TARGETS)"

# Thin dispatcher (dlopen-based, no transport code).
$(DISPATCH_OBJ): $(DISPATCH_SRC) uccl_p2p_ops.h uccl_engine.h include/common.h
	$(CXX) -O3 -std=c++17 -fPIC -I. -I./include -I../include -I$(CUDA_INC) -c $< -o $@

dispatch-lib: $(DISPATCH_OBJ)
	$(CXX) -shared -fPIC $< -o libuccl_p2p.so -ldl

# Each backend: re-invoke make with the appropriate flags and output name.
clean-backend:
	@rm -f $(CAPI_OBJECT) engine.o engine_api.o nccl/*.o nccl/*.d

transport-rdma:
	@$(MAKE) clean-backend
	@$(MAKE) P2P_SHARED_LIB=libuccl_p2p_rdma.so libuccl_p2p_rdma.so

transport-efa:
	@$(MAKE) clean-backend
	@$(MAKE) P2P_SHARED_LIB=libuccl_p2p_efa.so USE_EFA=1 libuccl_p2p_efa.so

transport-nccl:
	@$(MAKE) clean-backend
	@$(MAKE) P2P_SHARED_LIB=libuccl_p2p_nccl.so USE_NCCL=1 libuccl_p2p_nccl.so

install-all-transport:
	$(call safe_install,libuccl_p2p.so,$(LIBDIR),755)
	$(call safe_install,libuccl_p2p_rdma.so,$(LIBDIR),755)
	$(call safe_install,libuccl_p2p_efa.so,$(LIBDIR),755)
	$(call safe_install,libuccl_p2p_nccl.so,$(LIBDIR),755)
	$(call safe_install,$(CAPI_HEADER),$(INCDIR),644)
	@echo "Transport backends installed!"

# Help target
help:
	@echo "Available targets:"
	@echo "  all               - Build the nanobind module (stable ABI)"
	@echo "  all-transport     - Build all transport backends + dispatcher"
	@echo "  install           - Install the module to Python site-packages"
	@echo "  install-all-transport - Install dispatcher + all backends"
	@echo "  clean             - Remove build artifacts"
	@echo "  test              - Run the test script"
	@echo "  install-deps      - Install nanobind dependency"
	@echo "  help              - Show this help message"

.PHONY: all all-transport dispatch-lib transport-rdma transport-efa \
        transport-nccl clean clean-backend test install-deps \
        help install install-all-transport
