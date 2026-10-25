# Shared P2P build. Keep device/toolchain settings in a small profile and
# select optional transports here; all entry points build the same core.
.DEFAULT_GOAL := all
PROFILE ?= cuda
include make/profiles/$(PROFILE).mk

ENABLE_NCCL ?= 1
ENABLE_CXI ?= 1
ENABLE_EFA ?= 1
USE_DIETGPU ?= 0
PYTHON ?= python3
PYTHON_CONFIG ?= $(PYTHON)-config
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCDIR ?= $(PREFIX)/include
EFA_HOME ?= /opt/amazon/efa
LIBFABRIC_HOME ?= /opt/libfabric

NB_DIR ?= $(shell $(PYTHON) -c "import nanobind, os; print(os.path.dirname(nanobind.__file__))")
NB_SRC_DIR := $(NB_DIR)/src
NB_INCLUDES := -I$(NB_DIR)/include -I$(NB_DIR)/ext/robin_map/include \
  -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_STABLE_ABI_OK := $(shell $(PYTHON) -c "import sys, sysconfig; print(int(sys.version_info >= (3, 12) and not sysconfig.get_config_var('Py_GIL_DISABLED')))")
ifeq ($(PY_STABLE_ABI_OK),1)
NB_ABI_FLAGS := -DPy_LIMITED_API=0x030C0000 -DNB_STABLE_ABI=1
PYEXT := .abi3.so
else
NB_ABI_FLAGS :=
PYEXT := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
endif
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags)
INSTALL_DIR ?= $(shell $(PYTHON) -c "import site; print(site.getsitepackages()[0])")/uccl

# Isolate device, optional features and Python ABI. A config stamp below also
# invalidates objects when compilers, SDK paths or user flags change in place.
BUILD_DIR ?= .build/$(PROFILE)/nccl$(ENABLE_NCCL)-cxi$(ENABLE_CXI)-efa$(ENABLE_EFA)-diet$(USE_DIETGPU)/$(PYEXT)
override CPPFLAGS += -I. -I./util -I./rdma -I./rdma/providers -I../include \
  $(DEVICE_CPPFLAGS) -DUCCL_DEVICE_BACKEND_HEADER=\"util/device/$(DEVICE_BACKEND)/config.h\" \
  -DUCCL_P2P_ENABLE_NCCL=$(ENABLE_NCCL) -DUCCL_P2P_ENABLE_CXI=$(ENABLE_CXI) \
  -DUCCL_P2P_ENABLE_EFA=$(ENABLE_EFA) -I$(EFA_HOME)/include
override CXXFLAGS += -O3 -std=c++17 -fPIC -MMD -MP \
  -Wno-pointer-arith -Wno-sign-compare -Wno-unused-variable
IBV_WRAP_FLAGS := -Wl,--wrap=ibv_get_device_list \
  -Wl,--wrap=ibv_query_port -Wl,--wrap=ibv_reg_mr \
  -Wl,--wrap=ibv_reg_dmabuf_mr -Wl,--wrap=ibv_create_cq \
  -Wl,--wrap=ibv_create_qp -Wl,--wrap=ibv_qp_to_qp_ex
override LDLIBS += $(DEVICE_LDLIBS) -lz -lelf -lpthread -ldl
override LDFLAGS += -shared -Wl,-Bsymbolic-functions $(IBV_WRAP_FLAGS)

ifeq ($(USE_DIETGPU),1)
ifneq ($(DIETGPU_SUPPORTED),1)
$(error DietGPU is not supported by the $(PROFILE) profile)
endif
DIETGPU_ROOT := ../thirdparty/dietgpu
DIETGPU_FLOAT_DIR := $(DIETGPU_ROOT)/dietgpu/float
override CPPFLAGS += -DUSE_DIETGPU -I$(DIETGPU_ROOT)
override LDLIBS += -L$(DIETGPU_FLOAT_DIR) -ldietgpu_float \
  -Wl,-rpath,$(abspath $(DIETGPU_FLOAT_DIR)) -Wl,-rpath,'$$ORIGIN/lib'
endif

CORE_SOURCES := engine.cc \
  util/adaptive_sleeper.cc util/common.cc util/epoll_client.cc \
  util/epoll_server.cc util/memory_allocator.cc util/seq_num.cc \
  rdma/compression.cc rdma/rdma_context.cc rdma/rdma_connection.cc \
  rdma/rdma_ctrl_channel.cc rdma/rdma_data_channel.cc \
  rdma/rdma_device.cc rdma/rdma_endpoint.cc \
  rdma/providers/efa_data_channel_impl.cc rdma/providers/ib_data_channel_impl.cc \
  rdma/ibverbs_dl.cc rdma/efadv_dl.cc $(DEVICE_SOURCES)
ifeq ($(ENABLE_NCCL),1)
override CPPFLAGS += -I./nccl
CORE_SOURCES += nccl/nccl_endpoint.cc nccl/nccl_dl.cc
endif
ifeq ($(ENABLE_CXI),1)
override CPPFLAGS += -I$(LIBFABRIC_HOME)/include
CORE_SOURCES += cxi/cxi_endpoint.cc cxi/fabric_dl.cc
endif
CORE_OBJECTS := $(addprefix $(BUILD_DIR)/,$(CORE_SOURCES:.cc=.o))
API_OBJECTS := $(BUILD_DIR)/engine_api.o $(BUILD_DIR)/uccl_engine.o
NB_SRCS := nb_internals.cpp nb_func.cpp nb_type.cpp nb_enum.cpp nb_ndarray.cpp \
  nb_static_property.cpp common.cpp error.cpp trampoline.cpp implicit.cpp
NB_OBJECTS := $(addprefix $(BUILD_DIR)/nanobind/,$(NB_SRCS:.cpp=.o))
P2P_PYTHON_EXT := p2p$(PYEXT)
P2P_SHARED_LIB := libuccl_p2p.so

# Shared artifact names may be newer than this profile's objects. Relink on
# every invocation so selecting another profile cannot reuse a foreign binary.
.PHONY: all clean install FORCE $(P2P_PYTHON_EXT) $(P2P_SHARED_LIB)
all: $(P2P_PYTHON_EXT) $(P2P_SHARED_LIB)

# Quote make values as shell literals, including user-supplied flags.
shell_quote = '$(subst ','"'"',$(1))'
$(BUILD_DIR)/config: FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' $(call shell_quote,$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(NB_INCLUDES) $(NB_ABI_FLAGS) $(LDFLAGS) $(LDLIBS)) > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

$(P2P_PYTHON_EXT): $(CORE_OBJECTS) $(BUILD_DIR)/engine_api.o $(NB_OBJECTS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS) $(PYTHON_LDFLAGS)

$(P2P_SHARED_LIB): $(CORE_OBJECTS) $(BUILD_DIR)/uccl_engine.o
	$(CXX) $(LDFLAGS) -Wl,--no-undefined $^ -o $@ $(LDLIBS)

$(BUILD_DIR)/%.o: %.cc $(BUILD_DIR)/config
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(NB_INCLUDES) $(NB_ABI_FLAGS) -c $< -o $@

$(BUILD_DIR)/nanobind/%.o: $(NB_SRC_DIR)/%.cpp $(BUILD_DIR)/config
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(NB_INCLUDES) $(NB_ABI_FLAGS) -fvisibility=hidden -c $< -o $@

-include $(CORE_OBJECTS:.o=.d) $(API_OBJECTS:.o=.d) $(NB_OBJECTS:.o=.d)

install: all
	install -d $(INSTALL_DIR) $(LIBDIR) $(INCDIR)
	rm -f $(INSTALL_DIR)/p2p.so $(INSTALL_DIR)/p2p.abi3.so $(INSTALL_DIR)/p2p.cpython-*.so
	install -m 755 $(P2P_PYTHON_EXT) $(INSTALL_DIR)/
	install -m 755 $(P2P_SHARED_LIB) $(LIBDIR)/
	install -m 644 uccl_engine.h util/common.h $(INCDIR)/

clean:
	rm -f $(CORE_OBJECTS) $(API_OBJECTS) $(NB_OBJECTS)
	rm -f $(CORE_OBJECTS:.o=.d) $(API_OBJECTS:.o=.d) $(NB_OBJECTS:.o=.d) $(BUILD_DIR)/config
	rm -f $(P2P_PYTHON_EXT) $(P2P_SHARED_LIB) test_notify_serde test_epoll_server notif_e2e

test_notify_serde: tests/test_notify_serde.cc util/common.h
	$(CXX) -std=c++17 -O2 -I. -I./util -o $@ tests/test_notify_serde.cc

test_epoll_server: tests/test_epoll_server.cc util/epoll_server.cc util/epoll_server.h
	$(CXX) -std=c++17 -O2 -I. -I./util -I../include -o $@ tests/test_epoll_server.cc util/epoll_server.cc -lpthread

notif_e2e: tests/test_notify_e2e.cc $(P2P_SHARED_LIB)
	$(CXX) -std=c++17 -O2 $(CPPFLAGS) -o $@ tests/test_notify_e2e.cc \
	  -L. -luccl_p2p $(DEVICE_LDLIBS) -Wl,-rpath,'$$ORIGIN'

test: test_notify_serde test_epoll_server all
	./test_notify_serde
	./test_epoll_server
	$(PYTHON) tests/test_engine_write.py

install-deps:
	$(PYTHON) -m pip install nanobind

help:
	@echo 'Targets: all install clean test test_notify_serde test_epoll_server notif_e2e install-deps'
	@echo 'Profiles: Makefile (CUDA), Makefile.rocm, Makefile.dtk, Makefile.therock, Makefile.cnrt, Makefile.musa'
	@echo 'Options: ENABLE_NCCL=0|1 ENABLE_CXI=0|1 ENABLE_EFA=0|1 USE_DIETGPU=0|1'
.PHONY: test install-deps help
