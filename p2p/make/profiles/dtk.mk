HIP_HOME ?= /opt/dtk
include make/profiles/rocm.mk
DEVICE_CPPFLAGS += -D__UCCL_DTK__
