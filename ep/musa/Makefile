MUSA_HOME ?= /usr/local/musa
MCCL_HOME ?= /usr
MPICXX ?= mpicxx
MCC ?= $(MUSA_HOME)/bin/mcc
CXXFLAGS ?= -std=c++17 -O3 -g -Wall -Wextra -ffp-contract=off
MUSAFLAGS ?= -std=c++17 -O3 -ffp-contract=off
INCLUDES = -I. -I$(MUSA_HOME)/include -I$(MCCL_HOME)/include
LIBS = -L$(MUSA_HOME)/lib -L$(MCCL_HOME)/lib \
	-Wl,-rpath,$(MUSA_HOME)/lib -Wl,-rpath,$(MCCL_HOME)/lib \
	-lmccl -lmusart
OBJECTS = build/route_plan.o build/exchange_plan.o build/session.o build/kernels.o

.PHONY: probe kernels ep host-test
probe: mccl_p2p_probe
kernels: build/device_test
ep: build/ep_test
host-test:
	python3 test_ep.py

mccl_p2p_probe: mccl_p2p_probe.cc
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $< $(LIBS) -o $@

build:
	mkdir -p $@

build/%.o: %.cc route_plan.h exchange_plan.h session.h kernels.h | build
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

build/kernels.o: kernels.mu kernels.h kernel_math.h | build
	$(MCC) $(MUSAFLAGS) $(INCLUDES) -c $< -o $@

build/affine.o: tests/affine.mu | build
	$(MCC) $(MUSAFLAGS) $(INCLUDES) -c $< -o $@

build/device_test: tests/device_test.mu tests/numeric.h build/kernels.o build/route_plan.o
	$(MCC) $(MUSAFLAGS) $(INCLUDES) tests/device_test.mu build/kernels.o build/route_plan.o -L$(MUSA_HOME)/lib \
		-Wl,-rpath,$(MUSA_HOME)/lib -lmusart -o $@

build/ep_test: tests/ep_test.cc tests/numeric.h $(OBJECTS) build/affine.o
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) tests/ep_test.cc $(OBJECTS) build/affine.o $(LIBS) -o $@
