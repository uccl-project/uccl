ABS_ROOT_PATH := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ABS_REPO_ROOT := $(shell realpath $(ABS_ROOT_PATH)/..)

NCCL_HOME:=../thirdparty/nccl-sg
CUDA_HOME:=/usr/local/cuda
EFA_HOME:=/opt/amazon/efa

INC = -I./ -I$(CUDA_HOME)/include -I${EFA_HOME}/include -I$(ABS_REPO_ROOT)/include
LIBS = -L ${CUDA_HOME}/lib64 -L ${EFA_HOME}/lib -libverbs -lefa -lcudart -lcuda -lpthread -lglog -lgflags -lgtest -lz -lelf
LIBS_SHARED = ${LIBS}
override CXXFLAGS += -O3 -g -std=c++17 -Wno-pointer-arith -Wno-interference-size -fPIC -DUSE_CUDA
DEPS = *.h
PLUGIN_SO = libnccl-net.so

lib_src = $(wildcard *.cc)
lib_src := $(filter-out %_main.cc,$(lib_src))
lib_src := $(filter-out %_test.cc,$(lib_src))
lib_src := $(filter-out %_plugin.cc,$(lib_src))
lib_obj = $(lib_src:.cc=.o)
lib_obj += scattered_memcpy.o

main_src = $(wildcard *_main.cc)
main_src += $(wildcard *_test.cc)
main_bin = $(main_src:.cc=)

.PHONY: build
build: $(main_bin) $(PLUGIN_SO)

transport_test: transport_test.cc $(DEPS) $(lib_obj)
	g++ transport_test.cc -o transport_test $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

util_efa_test: util_efa_test.cc $(DEPS) $(lib_obj)
	g++ util_efa_test.cc -o util_efa_test $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

timely_test: timely_test.cc $(DEPS) $(lib_obj)
	g++ timely_test.cc -o timely_test $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

timing_wheel_test: timing_wheel_test.cc $(DEPS) $(lib_obj)
	g++ timing_wheel_test.cc -o timing_wheel_test $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

util_lrpc_test: util_lrpc_test.cc $(DEPS) $(lib_obj)
	g++ util_lrpc_test.cc -o util_lrpc_test $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

scattered_memcpy_test: scattered_memcpy_test.cc $(DEPS) $(lib_obj)
	g++ scattered_memcpy_test.cc -o scattered_memcpy_test $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

scattered_memcpy.o: scattered_memcpy.cu scattered_memcpy.cuh
	nvcc -c scattered_memcpy.cu -o scattered_memcpy.o -arch=sm_70 -Xcompiler -fPIC

%.o: %.cc $(DEPS)
	g++ -c $< -o $@ $(INC) $(CXXFLAGS)

NCCL_INC:= -I$(NCCL_HOME)/build/include -I$(NCCL_HOME)/src/include -I$(CUDA_HOME)/include

$(PLUGIN_SO): nccl_plugin.cc $(DEPS) $(lib_obj)
	g++ $(NCCL_INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) nccl_plugin.cc $(lib_obj) $(INC) $(LIBS_SHARED) $(CXXFLAGS)

.PHONY: clean
clean:
	rm -f *.o $(main_bin) $(PLUGIN_SO)
