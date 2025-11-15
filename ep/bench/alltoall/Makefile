# ===============================================================================
# Makefile for All-to-All RDMA Benchmark
# ===============================================================================

CUDA_PATH ?= /usr/local/cuda
CXX := g++

TARGET := benchmark

CXX_FLAGS := -std=c++17 -Wall -Wextra -pthread
CXX_FLAGS += -O3 -DNDEBUG

INCLUDES := -I$(CUDA_PATH)/include

LIBS := -lpthread -lcudart -libverbs -lefa -lnuma
LIBS += -L$(CUDA_PATH)/lib64

SRC_CPP := alltoall_efa.cpp

OBJ_CPP := $(SRC_CPP:.cpp=.o)

# ===============================================================================
# Build Rules
# ===============================================================================
.DEFAULT_GOAL := all

all: $(TARGET)

$(TARGET): $(OBJ_CPP)
	@echo "Linking $@..."
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -o $@ $^ $(LIBS)
	@echo "Build complete: $@"

%.o: %.cpp
	@echo "Compiling C++: $<"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	@echo "Cleaning..."
	@rm -f $(TARGET) *.o
	@echo "Done."

.PHONY: all clean
