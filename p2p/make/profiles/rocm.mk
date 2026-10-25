DEVICE_BACKEND := hip
HIP_HOME ?= /opt/rocm
CONDA_LIB_HOME ?= /usr/lib
DEVICE_CPPFLAGS := -D__HIP_PLATFORM_AMD__ -I$(HIP_HOME)/include -I$(CONDA_LIB_HOME)/../include
DEVICE_LDLIBS := -L$(HIP_HOME)/lib -lamdhip64 -Wl,-rpath,$(HIP_HOME)/lib \
  -L$(CONDA_LIB_HOME) -Wl,-rpath,$(CONDA_LIB_HOME)
DIETGPU_SUPPORTED := 1
