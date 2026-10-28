DEVICE_BACKEND := cuda
CUDA_HOME ?= /usr/local/cuda
DEVICE_CPPFLAGS := -I$(CUDA_HOME)/include
DEVICE_LDLIBS := -L$(CUDA_HOME)/lib64 -lcudart -lcuda -Wl,-rpath,$(CUDA_HOME)/lib64
DIETGPU_SUPPORTED := 1
