CXX       ?= g++
NVCC      ?= nvcc
AR        ?= ar

CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 80 90 100

CXX_STD   := -std=c++17
CXXFLAGS  := $(CXX_STD) -Wall -Wextra -fPIC -O2
CUFLAGS   := $(CXX_STD) -Xcompiler -Wall,-Wextra,-fPIC -O2
NVCC_GENCODE := $(foreach arch,$(CUDA_ARCH),-gencode arch=compute_$(arch),code=sm_$(arch))

INCLUDES  := -Icore -Icollective -Inccl -Ithirdparty -I$(CUDA_HOME)/include -I$(CUDA_HOME)/include/cccl
DEFINES   := -DUSE_IBVERBS -DMSCCLPP_USE_CUDA
LDFLAGS   := -L$(CUDA_HOME)/lib64
LDLIBS    := -lcudart -lcuda -libverbs -lnuma -lpthread -ldl -lstdc++fs

BUILD_DIR := build

# --- Core infra (core/) ---
CORE_CC_SRCS  := $(wildcard core/*.cc)
CORE_CPP_SRCS := $(wildcard core/*.cpp)
CORE_CU_SRCS  := $(wildcard core/*.cu)

CORE_CC_OBJS  := $(patsubst core/%.cc,$(BUILD_DIR)/core/%.o,$(CORE_CC_SRCS))
CORE_CPP_OBJS := $(patsubst core/%.cpp,$(BUILD_DIR)/core/%.o,$(CORE_CPP_SRCS))
CORE_CU_OBJS  := $(patsubst core/%.cu,$(BUILD_DIR)/core/%.cu.o,$(CORE_CU_SRCS))
CORE_OBJS     := $(CORE_CC_OBJS) $(CORE_CPP_OBJS) $(CORE_CU_OBJS)

# --- Collective library (collective/) ---
COLL_CC_SRCS  := $(wildcard collective/*.cc)
COLL_CU_SRCS  := $(wildcard collective/*.cu)

COLL_CC_OBJS  := $(patsubst collective/%.cc,$(BUILD_DIR)/collective/%.o,$(COLL_CC_SRCS))
COLL_CU_OBJS  := $(patsubst collective/%.cu,$(BUILD_DIR)/collective/%.cu.o,$(COLL_CU_SRCS))
COLL_OBJS     := $(COLL_CC_OBJS) $(COLL_CU_OBJS)

# --- Targets ---
LIB_CORE_SO := $(BUILD_DIR)/libmint.so
LIB_CORE_A  := $(BUILD_DIR)/libmint.a
LIB_COLL_SO := $(BUILD_DIR)/libmint_collective.so
LIB_COLL_A  := $(BUILD_DIR)/libmint_collective.a

.PHONY: all clean core collective

all: core collective

core: $(LIB_CORE_SO) $(LIB_CORE_A)
collective: $(LIB_COLL_SO) $(LIB_COLL_A)

$(BUILD_DIR)/core $(BUILD_DIR)/collective:
	mkdir -p $@

# Core compilation rules
$(BUILD_DIR)/core/%.o: core/%.cc | $(BUILD_DIR)/core
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -c $< -o $@

$(BUILD_DIR)/core/%.o: core/%.cpp | $(BUILD_DIR)/core
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -c $< -o $@

$(BUILD_DIR)/core/%.cu.o: core/%.cu | $(BUILD_DIR)/core
	$(NVCC) $(CUFLAGS) $(NVCC_GENCODE) $(INCLUDES) $(DEFINES) -c $< -o $@

# Collective library compilation rules
$(BUILD_DIR)/collective/%.o: collective/%.cc | $(BUILD_DIR)/collective
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -c $< -o $@

$(BUILD_DIR)/collective/%.cu.o: collective/%.cu | $(BUILD_DIR)/collective
	$(NVCC) $(CUFLAGS) $(NVCC_GENCODE) $(INCLUDES) $(DEFINES) -c $< -o $@

# Core library
$(LIB_CORE_SO): $(CORE_OBJS)
	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(LIB_CORE_A): $(CORE_OBJS)
	$(AR) rcs $@ $^

# Collective library (links against core)
$(LIB_COLL_SO): $(COLL_OBJS) $(LIB_CORE_SO)
	$(CXX) -shared -o $@ $(COLL_OBJS) -L$(BUILD_DIR) -lmint $(LDFLAGS) $(LDLIBS)

$(LIB_COLL_A): $(COLL_OBJS)
	$(AR) rcs $@ $^

clean:
	rm -rf $(BUILD_DIR)
