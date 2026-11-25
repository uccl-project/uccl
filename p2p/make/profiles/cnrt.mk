DEVICE_BACKEND := cnrt
NEUWARE_HOME ?= /usr/local/neuware
DEVICE_CPPFLAGS := -D__CAMBRICON_PLATFORM_MLU__ -I$(NEUWARE_HOME)/include
DEVICE_LDLIBS := -L$(NEUWARE_HOME)/lib64 -lcnrt -lcndrv -Wl,-rpath,$(NEUWARE_HOME)/lib64
DEVICE_SOURCES := device/cnrt/staging.cc
