# Makefile for UCCL P2P Engine nanobind project (stable ABI)
#
# Single binary, runtime transport selection via UCCL_P2P_TRANSPORT env var.
# libibverbs, libnccl, and libefa are loaded dynamically at runtime via
# dlopen/dlsym (see include/*_wrapper.h). No transport libraries are linked.
#
# Runtime: UCCL_P2P_TRANSPORT=rdma|nccl|efa (default: rdma)

# DietGPU optional integration
USE_DIETGPU ?= $(shell echo $${USE_DIETGPU:-0})

# Compiler and flags
CUDA_HOME ?= /usr/local/cuda
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64
EFA_HOME ?= /opt/amazon/efa
NCCL_HOME ?= ../thirdparty/nccl/build
NCCL_INC  ?= $(NCCL_HOME)/include

# Building with the following settings:
# NOTE: -libverbs, -lnccl, -lefa are NOT linked at compile time.
INC = -I./ -I$(CUDA_HOME)/include
LIBS = -L ${CUDA_HOME}/lib64 -lcudart -lcuda -lpthread -lz -lelf -ldl

CXX := g++
CXXFLAGS := -O3 -shared -std=c++17 -fPIC -I. -I./include -I../include -I$(CUDA_INC) \
	-I$(EFA_HOME)/include -I$(NCCL_INC) \
	-MMD -MP \
	-Wno-pointer-arith -Wno-sign-compare -Wno-unused-variable \
	-Wl,-rpath=/usr/lib/x86_64-linux-gnu

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

# libibverbs is linked but loaded lazily at runtime - the dlsym wrappers in
# ibverbs_dl.cc handle most symbols; the compat-layer symbols (ibv_query_port,
# ibv_reg_mr etc.) are resolved from the linked libibverbs when the .so loads.
# libnccl and libefa are NOT linked - fully dlsym'd at runtime.
LDFLAGS = -L$(CUDA_LIB) -lcudart -lcuda -Wl,-rpath,$(CUDA_LIB) \
          -lz -lelf -lpthread -ldl -libverbs $(DIETGPU_LIBS)

# Target and source files - always compile ALL sources
P2P_PYTHON_EXT := p2p$(PYEXT)
P2P_SHARED_LIB := libuccl_p2p.so
CAPI_SOURCE := uccl_engine.cc
CAPI_HEADER := uccl_engine.h include/common.h
CAPI_OBJECT := $(CAPI_SOURCE:.cc=.o)
# dlsym wrapper objects provide ibv_*/nccl*/efadv_* symbols at link time
DL_SOURCES := rdma/ibverbs_dl.cc nccl/nccl_dl.cc rdma/providers/efa/efadv_dl.cc
DL_OBJECTS := rdma/ibverbs_dl.o nccl/nccl_dl.o rdma/providers/efa/efadv_dl.o
SOURCES := engine.cc engine_api.cc nccl/nccl_endpoint.cc
CORE_OBJECT := engine.o nccl/nccl_endpoint.o $(DL_OBJECTS)
OBJECTS := $(SOURCES:.cc=.o)

# Default target
all: $(P2P_PYTHON_EXT) $(P2P_SHARED_LIB)

$(P2P_PYTHON_EXT): $(OBJECTS) $(NB_OBJECTS) $(DL_OBJECTS)
	$(CXX) $(OBJECTS) $(DL_OBJECTS) $(NB_OBJECTS) \
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

# Compile NCCL source files
nccl/%.o: nccl/%.cc
	@mkdir -p nccl
	$(CXX) $(CXXFLAGS) $(NB_INCLUDES) $(NB_ABI_FLAGS) -I. -c $< -o $@

# Compile RDMA provider source files
rdma/providers/efa/%.o: rdma/providers/efa/%.cc
	@mkdir -p rdma/providers/efa
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

install: $(P2P_PYTHON_EXT) $(P2P_SHARED_LIB)
	$(MAKE) $(P2P_PYTHON_EXT)
	@mkdir -p $(INSTALL_DIR)
	@rm -f $(INSTALL_DIR)/p2p.cpython-*.so $(INSTALL_DIR)/p2p.abi3.so
	@cp $(P2P_PYTHON_EXT) $(INSTALL_DIR)/
	@echo "Installation complete. Module installed as: $(INSTALL_DIR)/$(P2P_PYTHON_EXT)"
	$(call safe_install,$(P2P_SHARED_LIB),$(LIBDIR),755)
	$(call safe_install,$(CAPI_HEADER),$(INCDIR),644)
	@echo "Installation complete!"

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(CAPI_OBJECT) $(DL_OBJECTS) $(NB_OBJECTS) $(P2P_SHARED_LIB)
	rm -f $(DEPS)
	rm -f nccl/*.o nccl/*.d rdma/*.o rdma/*.d rdma/providers/efa/*.o rdma/providers/efa/*.d
	rm -f $(P2P_PYTHON_EXT)

# Test the module
test:
	$(MAKE) $(P2P_PYTHON_EXT)
	$(PYTHON) tests/test_engine_write.py

# Install nanobind if not available
install-deps:
	pip3 install nanobind

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the nanobind module and shared library"
	@echo "  install      - Install the module to Python site-packages"
	@echo "  clean        - Remove build artifacts"
	@echo "  test         - Run the test script"
	@echo "  install-deps - Install nanobind dependency"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Runtime options (no rebuild needed):"
	@echo "  UCCL_P2P_TRANSPORT=rdma|nccl|efa  (default: rdma)"
	@echo "  UCCL_P2P_DISABLE_IPC=1            (force network transport)"
	@echo "  UCCL_IBV_SO=/path/to/libibverbs.so (custom ibverbs path)"
	@echo "  UCCL_NCCL_SO=/path/to/libnccl.so  (custom nccl path)"

.PHONY: all clean test install-deps help install
