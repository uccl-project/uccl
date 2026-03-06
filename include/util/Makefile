ABS_ROOT_PATH := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ABS_REPO_ROOT := $(shell realpath $(ABS_ROOT_PATH)/../..)

CUDA_HOME?=/usr/local/cuda

INC = -I$(ABS_REPO_ROOT)/include -I$(CUDA_HOME)/include
LIBS = -lglog -lgflags -lz -lelf -libverbs -L $(CUDA_HOME)/lib64 -lcudart -lpthread
override CXXFLAGS += -O3 -g -std=c++17 -Wno-pointer-arith -Wno-interference-size -fPIC -D__STDC_FORMAT_MACROS -DUSE_CUDA

DEPS = *.h

test_src = $(wildcard *_test.cc)
test_bin = $(test_src:.cc=)

.PHONY: build
build: $(test_bin)

%_test: %_test.cc $(DEPS)
	g++ $< -o $@ $(INC) $(LIBS) $(CXXFLAGS)

.PHONY: clean
clean:
	rm -f $(test_bin)
