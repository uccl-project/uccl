CXX       ?= g++
NVCC      ?= nvcc
AR        ?= ar

CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 80 90 100

CXX_STD   := -std=c++17
CXXFLAGS  := $(CXX_STD) -Wall -Wextra -fPIC -O2
CUFLAGS   := $(CXX_STD) -Xcompiler -Wall,-Wextra,-fPIC -O2
NVCC_GENCODE := $(foreach arch,$(CUDA_ARCH),-gencode arch=compute_$(arch),code=sm_$(arch))

INCLUDES  := -Icore -Ikernel -Ithirdparty -I$(CUDA_HOME)/include -I$(CUDA_HOME)/include/cccl
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

# --- Collective kernels (kernel/) ---
KERN_CC_SRCS  := $(wildcard kernel/*.cc)
KERN_CU_SRCS  := $(wildcard kernel/*.cu)

KERN_CC_OBJS  := $(patsubst kernel/%.cc,$(BUILD_DIR)/kernel/%.o,$(KERN_CC_SRCS))
KERN_CU_OBJS  := $(patsubst kernel/%.cu,$(BUILD_DIR)/kernel/%.cu.o,$(KERN_CU_SRCS))
KERN_OBJS     := $(KERN_CC_OBJS) $(KERN_CU_OBJS)

# --- Targets ---
LIB_CORE_SO   := $(BUILD_DIR)/libmint.so
LIB_CORE_A    := $(BUILD_DIR)/libmint.a
LIB_KERNEL_SO := $(BUILD_DIR)/libmint_kernel.so
LIB_KERNEL_A  := $(BUILD_DIR)/libmint_kernel.a

.PHONY: all clean core kernel

all: core kernel

core: $(LIB_CORE_SO) $(LIB_CORE_A)
kernel: $(LIB_KERNEL_SO) $(LIB_KERNEL_A)

$(BUILD_DIR)/core $(BUILD_DIR)/kernel:
	mkdir -p $@

# Core compilation rules
$(BUILD_DIR)/core/%.o: core/%.cc | $(BUILD_DIR)/core
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -c $< -o $@

$(BUILD_DIR)/core/%.o: core/%.cpp | $(BUILD_DIR)/core
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -c $< -o $@

$(BUILD_DIR)/core/%.cu.o: core/%.cu | $(BUILD_DIR)/core
	$(NVCC) $(CUFLAGS) $(NVCC_GENCODE) $(INCLUDES) $(DEFINES) -c $< -o $@

# Kernel compilation rules
$(BUILD_DIR)/kernel/%.o: kernel/%.cc | $(BUILD_DIR)/kernel
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -c $< -o $@

$(BUILD_DIR)/kernel/%.cu.o: kernel/%.cu | $(BUILD_DIR)/kernel
	$(NVCC) $(CUFLAGS) $(NVCC_GENCODE) $(INCLUDES) $(DEFINES) -c $< -o $@

# Core library
$(LIB_CORE_SO): $(CORE_OBJS)
	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(LIB_CORE_A): $(CORE_OBJS)
	$(AR) rcs $@ $^

# Kernel library (links against core)
$(LIB_KERNEL_SO): $(KERN_OBJS) $(LIB_CORE_SO)
	$(CXX) -shared -o $@ $(KERN_OBJS) -L$(BUILD_DIR) -lmint $(LDFLAGS) $(LDLIBS)

$(LIB_KERNEL_A): $(KERN_OBJS)
	$(AR) rcs $@ $^

clean:
	rm -rf $(BUILD_DIR)
