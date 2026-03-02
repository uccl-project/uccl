# Makefile for UCCL P2P Engine nanobind project (stable ABI)

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
	-MMD -MP \
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

# Python and nanobind configuration (stable ABI)
PYTHON          ?= python3
PYTHON_CONFIG    = $(PYTHON)-config
PYEXT           := .abi3.so
NB_DIR          := $(shell $(PYTHON) -c "import nanobind as _nb, os; print(os.path.dirname(_nb.__file__))")
NB_INCLUDES     := -I$(NB_DIR)/include -I$(NB_DIR)/ext/robin_map/include -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
NB_ABI_FLAGS    := -DPy_LIMITED_API=0x030C0000 -DNB_STABLE_ABI=1
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
SOURCES := engine.cc engine_api.cc nccl/nccl_endpoint.cc
CORE_OBJECT := engine.o nccl/nccl_endpoint.o
else
SOURCES := engine.cc engine_api.cc
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
$(P2P_SHARED_LIB): $(CAPI_OBJECT) $(CORE_OBJECTS) $(CU_OBJECTS)
	$(CXX) $(CAPI_OBJECT) $(CORE_OBJECTS) $(CU_OBJECTS) \
	      -L$(CUDA_LIB) -lcudart -lcuda \
	      -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS) $(CXXFLAGS) \
	      -Wl,-rpath,$(CUDA_LIB) $(PYTHON_LIB)
else
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
endif

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

ifeq ($(USE_TCPX),1)
install: $(P2P_SHARED_LIB)
	$(call safe_install,$(P2P_SHARED_LIB),$(LIBDIR),755)
	$(call safe_install,$(CAPI_HEADER),$(INCDIR),644)
	@echo "Installation complete!"
else
install: $(P2P_PYTHON_EXT) $(P2P_SHARED_LIB)
	$(MAKE) $(P2P_PYTHON_EXT)
	@mkdir -p $(INSTALL_DIR)
	@cp $(P2P_PYTHON_EXT) $(INSTALL_DIR)/
	@echo "Installation complete. Module installed as: $(INSTALL_DIR)/$(P2P_PYTHON_EXT)"
	$(call safe_install,$(P2P_SHARED_LIB),$(LIBDIR),755)
	$(call safe_install,$(CAPI_HEADER),$(INCDIR),644)
	@echo "Installation complete!"
endif
# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(CAPI_OBJECT) $(NB_OBJECTS) $(P2P_SHARED_LIB)
	rm -f $(DEPS)
	rm -f nccl/*.o nccl/*.d
ifneq ($(USE_TCPX),1)
	rm -f $(P2P_PYTHON_EXT)
endif

# Test the module
test:
ifneq ($(USE_TCPX),1)
	$(MAKE) $(P2P_PYTHON_EXT)
	$(PYTHON) tests/test_engine_write.py
endif

# Install nanobind if not available
install-deps:
	pip3 install nanobind

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the nanobind module (stable ABI)"
	@echo "  install      - Install the module to Python site-packages"
	@echo "  clean        - Remove build artifacts"
	@echo "  test         - Run the test script"
	@echo "  install-deps - Install nanobind dependency"
	@echo "  help         - Show this help message"

.PHONY: all clean test install-deps help install
