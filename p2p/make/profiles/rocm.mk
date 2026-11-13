DEVICE_BACKEND := hip
HIP_HOME ?= /opt/rocm
CONDA_LIB_HOME ?= /usr/lib
DEVICE_CPPFLAGS := -D__HIP_PLATFORM_AMD__ -I$(HIP_HOME)/include -I$(CONDA_LIB_HOME)/../include
DEVICE_LDLIBS := -L$(HIP_HOME)/lib -lamdhip64 -Wl,-rpath,$(HIP_HOME)/lib \
  -L$(CONDA_LIB_HOME) -Wl,-rpath,$(CONDA_LIB_HOME)
DIETGPU_SUPPORTED := 1

# CXI is the HPE Slingshot backend and needs libfabric user-space headers;
# a stock ROCm host does not have them. Opt in with ENABLE_CXI=1.
ENABLE_CXI ?= 0
