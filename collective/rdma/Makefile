ABS_ROOT_PATH := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ABS_REPO_ROOT := $(shell realpath $(ABS_ROOT_PATH)/../..)

NCCL_HOME:=$(ABS_REPO_ROOT)/thirdparty/nccl
CUDA_HOME?=/usr/local/cuda

INC = -I./ -I$(CUDA_HOME)/include -I$(ABS_REPO_ROOT)/include
LIBS = -lglog -lgflags -lgtest -lz -lelf -libverbs -L /usr/local/cuda/lib64 -lcudart -lpthread
LIBS_SHARED = -lglog -lgflags -lgtest -lz -lelf -libverbs
override CXXFLAGS += -O3 -g -std=c++17 -Wno-pointer-arith -Wno-interference-size -fPIC -DUSE_CUDA
DEPS = *.h
PLUGIN_SO = libnccl-net-uccl.so
NCCL_INC:= -I$(NCCL_HOME)/build/include -I$(NCCL_HOME)/src/include -I$(CUDA_HOME)/include

lib_src = $(wildcard *.cc)
lib_src := $(filter-out %_main.cc,$(lib_src))
lib_src := $(filter-out %_test.cc,$(lib_src))
lib_src := $(filter-out %_plugin.cc,$(lib_src))
lib_obj = $(lib_src:.cc=.o)

test_src = $(wildcard *_test.cc)
test_bin = $(test_src:.cc=)

.PHONY: build
build: $(test_bin) $(lib_obj) $(PLUGIN_SO) librdma.a librdma_plugin.a

azure: $(test_bin) $(lib_obj)

%_test: %_test.cc $(DEPS) $(lib_obj)
	g++ $< -o $@ $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

%.o: %.cc $(DEPS)
	g++ -c $< -o $@ $(INC) $(CXXFLAGS)

$(PLUGIN_SO): nccl_plugin.cc $(DEPS) $(lib_obj)
	g++ $(NCCL_INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) nccl_plugin.cc $(lib_obj) $(INC) $(LIBS_SHARED) $(CXXFLAGS)

librdma.a: $(lib_obj)
	ar rcs $@ $(lib_obj)

librdma_plugin.a: $(lib_obj)
	ar rcs $@ $(lib_obj)

.PHONY: clean
clean:
	rm -f *.o $(test_bin) $(PLUGIN_SO) librdma.a librdma_plugin.a
