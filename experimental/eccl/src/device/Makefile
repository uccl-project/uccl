# 设置编译器和标志
CXX = g++
CXXFLAGS = -std=c++17 -O3 -g -Wall

# 设置 CUDA 编译器和标志
NVCC = nvcc
NVCCFLAGS = -O3 -g -G -Xcompiler -Wall -arch=sm_80

# 设置库和包含目录
INCLUDES = -I./ -I/usr/local/cuda/include
LIBS = -L/usr/local/cuda/lib -lcudart -lnuma

# 目标文件
TARGET = test_persistent

# 源文件
SRC = ./test_persistent.cc ./c2d_fifo.cc ./operator.cu
OBJ = test_persistent.o c2d_fifo.o operator.o

# Source files
SRC_CPP := ./test_persistent.cc ./c2d_fifo.cc
SRC_CU  := ./operator.cu

OBJ_CC   := $(SRC_CPP:.cc=.o)
OBJ_CU   := $(SRC_CU:.cu=.o)

# 为 .cc 文件添加编译规则
%.o: %.cc
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

# 为 .cu 文件添加编译规则
%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

# 链接
$(TARGET): $(OBJ_CC) $(OBJ_CU)
	$(NVCC) $(OBJ_CC) $(OBJ_CU) -o $(TARGET) $(LIBS) $(NVCCFLAGS)

# 清理
clean:
	rm -f $(OBJ_CC) $(OBJ_CU) $(TARGET)
