CXX      := g++
SM       ?= 80
SM_NUM   := $(shell printf '%s\n' "$(SM)" | sed 's/[^0-9].*//')
ENABLE_TMA := $(shell if [ -n "$(SM_NUM)" ] && [ "$(SM_NUM)" -ge 90 ]; then echo 1; else echo 0; fi)
CXXFLAGS := -std=c++17 -O3 -g -Wall
CXXFLAGS += -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA)

VPATH := .:test:../transport:../device:../device/fifo

INCLUDES := -I. -I./test -I../transport -I../device -I../device/fifo -I../../include

CXX      := g++
SM       ?= 80
SM_NUM   := $(shell printf '%s\n' "$(SM)" | sed 's/[^0-9].*//')
ENABLE_TMA := $(shell if [ -n "$(SM_NUM)" ] && [ "$(SM_NUM)" -ge 90 ]; then echo 1; else echo 0; fi)
CXXFLAGS := -std=c++17 -O3 -g -Wall
CXXFLAGS += -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA)

VPATH := .:test

INCLUDES := -I. -I./test -I./backend -I../../include

CCL_SRC := topology.cc plan.cc selector.cc executor.cc backend/device_backend.cc

TEST_SRC := test/test_ccl_core.cc

CCL_OBJ := $(CCL_SRC:.cc=.o)
TEST_OBJ := $(TEST_SRC:.cc=.o)

.PHONY: all test clean

all: test

test: test_ccl_core
	./test_ccl_core

test_ccl_core: $(CCL_OBJ) $(TEST_OBJ)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) -lpthread

%.o: %.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

clean:
	rm -f *.o *.d test/*.d test_ccl_core

-include $(CCL_OBJ:.o=.d) $(TEST_OBJ:.o=.d)
