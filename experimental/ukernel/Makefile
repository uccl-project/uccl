# Compiler and environment
CUDA_PATH      ?= /usr/local/cuda
override CXX   := /usr/bin/g++
NVCC           := $(CUDA_PATH)/bin/nvcc
CONDA_LIB_HOME ?= /usr/lib

# GPU architecture detection
DETECTED_SM := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '.')
SM ?= $(if $(DETECTED_SM),$(DETECTED_SM),70)

# Base compilation flags
CXXFLAGS   := -O0 -std=c++17 -Wall -Wno-unused-function -pthread -fPIC \
              -I$(CUDA_PATH)/include -I$(CONDA_LIB_HOME)/../include

NVCCFLAGS  := -g -O0 -std=c++17 \
              --compiler-options "-fPIC -Wall -pthread -I$(CONDA_LIB_HOME)/../include" \
              -arch=sm_$(SM) \
              -ccbin $(CXX)

LDFLAGS    := -L$(CONDA_LIB_HOME) \
              -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 \
              -lcudart -lcuda \
              -lgflags -lgtest -lz -lelf -libverbs -lnl-3 -lnl-route-3 -lpthread -lglog

INCLUDES := -Iinclude -I$(CUDA_PATH)/include -I/usr/include -I../../include

# Redis support: when USE_REDIS_OOB=1
ifeq ($(USE_REDIS_OOB),1)
PKG_CONFIG := pkg-config
REDIS_CFLAGS := $(shell $(PKG_CONFIG) --cflags hiredis)
REDIS_LIBS   := $(shell $(PKG_CONFIG) --libs hiredis)

CXXFLAGS   += $(REDIS_CFLAGS) -DUSE_REDIS_OOB
NVCCFLAGS  += -DUSE_REDIS_OOB --compiler-options "$(REDIS_CFLAGS)"
LDFLAGS    += $(REDIS_LIBS)
endif

# Source files
SRC_CPP := $(shell find src -name '*.cc' | grep -v 'src/device') # skip device now
SRC_CU  := $(shell find src -name '*.cu' | grep -v 'src/device')
TEST_DIR := tests
SRC_TEST := $(shell find $(TEST_DIR) -name 'test_*.cc')

OBJ_CC   := $(SRC_CPP:.cc=.o)
OBJ_CU   := $(SRC_CU:.cu=.o)
OBJ_TEST := $(SRC_TEST:.cc=.o)

# Targets
STATIC_LIB  := libukernel.a
SHARED_LIB  := libukernel.so
TARGET_TEST := test_main

# Build both static, shared, and test by default
all: $(STATIC_LIB) $(SHARED_LIB) $(TARGET_TEST)

# Compilation rules
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# Static library
$(STATIC_LIB): $(OBJ_CC)
	ar rcs $@ $^

# Shared library
$(SHARED_LIB): $(OBJ_CC) $(OBJ_CU)
	$(CXX) -shared $(CXXFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

# Test executable
$(TARGET_TEST): $(OBJ_TEST) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJ_TEST) $(STATIC_LIB) $(LDFLAGS) -o $@

# Run tests
run_test: $(TARGET_TEST)
	./$(TARGET_TEST)

# Clean
clean:
	rm -f $(OBJ_CC) $(OBJ_CU) $(OBJ_TEST) $(TARGET_TEST) $(STATIC_LIB) $(SHARED_LIB) *.d src/*.d tests/*.d

# Include dependencies
-include $(OBJ_CC:.o=.d) $(OBJ_CU:.o=.d) $(OBJ_TEST:.o=.d)

.PHONY: all clean run_test