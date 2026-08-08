# Mirrors experimental/mscclpp/apps/nccl/CMakeLists.txt:
#   mscclpp_nccl_obj  <- GLOB src/*.cu  -> here, all *.cu in this directory
#   mscclpp_nccl      <- SHARED mscclpp_obj + mscclpp_nccl_obj
#   mscclpp_nccl_static
#   mscclpp_audit_nccl <- audit-shim/*.cc

NVCC      ?= nvcc
CXX       ?= g++
AR        ?= ar

MINT_DIR  := ..
BUILD_DIR := $(MINT_DIR)/build
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 80 90 100

CXX_STD   := -std=c++17
CXXFLAGS  := $(CXX_STD) -Wall -Wextra -fPIC -O2
CUFLAGS   := $(CXX_STD) -Xcompiler -Wall,-Wextra,-fPIC -O2
NVCC_GENCODE := $(foreach arch,$(CUDA_ARCH),-gencode arch=compute_$(arch),code=sm_$(arch))

INCLUDES  := -I. -I$(MINT_DIR)/core -I$(MINT_DIR)/collective -I$(MINT_DIR)/thirdparty \
             -I$(CUDA_HOME)/include -I$(CUDA_HOME)/include/cccl
DEFINES   := -DUSE_IBVERBS -DMSCCLPP_USE_CUDA

OUT_DIR   := build
CU_SRCS   := $(wildcard *.cu)
CU_OBJS   := $(patsubst %.cu,$(OUT_DIR)/%.cu.o,$(CU_SRCS))

LIB_MINT      := $(BUILD_DIR)/libmint.so
LIB_MINT_COLL := $(BUILD_DIR)/libmint_collective.so

LIB_NCCL_SO  := $(OUT_DIR)/libmscclpp_nccl.so
LIB_NCCL_A   := $(OUT_DIR)/libmscclpp_nccl.a
LIB_AUDIT_SO := $(OUT_DIR)/libmscclpp_audit_nccl.so
AUDIT_OBJ    := $(OUT_DIR)/audit_nccl.o

LDFLAGS := -L$(BUILD_DIR) -L$(CUDA_HOME)/lib64
LDLIBS  := -Wl,--no-as-needed -lmint_collective -lmint -lcudart -lcuda -libverbs -lnuma \
           -lpthread -ldl -lstdc++fs

.PHONY: all clean mint-deps

all: mint-deps $(LIB_NCCL_SO) $(LIB_NCCL_A) $(LIB_AUDIT_SO)

mint-deps:
	$(MAKE) -C $(MINT_DIR) core collective

$(OUT_DIR):
	mkdir -p $@

$(OUT_DIR)/%.cu.o: %.cu | $(OUT_DIR)
	$(NVCC) $(CUFLAGS) $(NVCC_GENCODE) $(INCLUDES) $(DEFINES) -c $< -o $@

$(LIB_NCCL_SO): $(CU_OBJS) $(LIB_MINT_COLL) $(LIB_MINT)
	$(CXX) -shared -o $@ $(CU_OBJS) $(LDFLAGS) -Wl,-rpath,'$$ORIGIN:$(abspath $(BUILD_DIR))' $(LDLIBS)

$(LIB_NCCL_A): $(CU_OBJS)
	$(AR) rcs $@ $^

$(AUDIT_OBJ): audit_nccl.cc | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -fPIC -c $< -o $@

$(LIB_AUDIT_SO): $(AUDIT_OBJ)
	$(CXX) -shared -o $@ $^

clean:
	rm -rf $(OUT_DIR)
