CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc
ARCH ?= sm_90a
STAGES ?= 0
OUT ?= build
ROOT := $(abspath ../../..)
EP := $(ROOT)/thirdparty/nccl-ep

FLAGS := -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr -arch=$(ARCH) -shared -Xcompiler=-fPIC,-pthread
FLAGS += -DUCCL_EXPERT_STAGES=$(STAGES)
INCLUDES := -I$(ROOT) -I$(EP) -I$(EP)/include -I$(NCCL_DIR)/include
RPATH := -Xlinker=-rpath -Xlinker=$(CUDA_HOME)/lib64
ifneq ($(filter sm_100a sm_103a,$(ARCH)),)
MMA_FLAGS := -DUCCL_EXPERT_BLACKWELL=1
endif

.PHONY: all
all: $(OUT)/libexpert_mma.so $(OUT)/libht_model_bridge.so

$(OUT):
	mkdir -p $@

$(OUT)/libexpert_mma.so: expert_mma.cu $(EP)/device/ht_scheduler.hpp | $(OUT)
	$(NVCC) $(FLAGS) $(MMA_FLAGS) $(INCLUDES) -I$(CUTLASS_DIR)/include \
		-I$(CUTLASS_DIR)/tools/util/include $< -L$(CUDA_HOME)/lib64/stubs -lcuda $(RPATH) -o $@

$(OUT)/libht_model_bridge.so: ht_bridge.cpp $(EP)/device/ht_scheduler.hpp | $(OUT)
	$(NVCC) $(FLAGS) $(INCLUDES) $< -L$(UCCL_EP_DIR) -lnccl_ep \
		-Xlinker=$(NCCL_DIR)/lib/libnccl.so.2 $(RPATH) \
		-Xlinker=-rpath -Xlinker=$(UCCL_EP_DIR) \
		-Xlinker=-rpath -Xlinker=$(NCCL_DIR)/lib -o $@
