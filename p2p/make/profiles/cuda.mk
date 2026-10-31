DEVICE_BACKEND := cuda
CUDA_HOME ?= /usr/local/cuda
# CUDA-compatible SDKs may keep headers and runtime libraries apart.
CUDA_INC ?= $(CUDA_HOME)/include
CUDA_LIB ?= $(CUDA_HOME)/lib64
DEVICE_CPPFLAGS := -I$(CUDA_INC)
DEVICE_LDLIBS := -L$(CUDA_LIB) -lcudart -lcuda -Wl,-rpath,$(CUDA_LIB)
DIETGPU_SUPPORTED := 1
