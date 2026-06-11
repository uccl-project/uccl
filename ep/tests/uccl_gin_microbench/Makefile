# UCCL-GIN vs NCCL-GIN microbench.
#
# Prereq: build the UCCL ep objects first (this reuses them rather than
# recompiling proxy/rdma standalone, which need the full EFA/define chain):
#     make -C ../.. install PYTHON=$VIRTUAL_ENV/bin/python CUDA_PATH=/usr/local/cuda-13.0 SM=90 -j
# That leaves ../../src/*.o in place; we link the transport objects from there.
#
# Then:  make            (uses CUDA_HOME / NCCL from the active venv)
# Run :  see README.md (mpirun across the 2 nodes)

CUDA_HOME ?= /usr/local/cuda-13.0
EFA_HOME  ?= /opt/amazon/efa
PYTHON    ?= python3
NUM_PROXY_THS ?= 4

NVCC := $(CUDA_HOME)/bin/nvcc
ifneq ($(wildcard /opt/amazon/openmpi5/bin/mpicxx),)
MPICXX ?= /opt/amazon/openmpi5/bin/mpicxx
else ifneq ($(wildcard /opt/amazon/openmpi/bin/mpicxx),)
MPICXX ?= /opt/amazon/openmpi/bin/mpicxx
else
MPICXX ?= mpicxx
endif

# NCCL headers/libs from the installed wheel (nvidia-nccl-cuXX).
NCCL_INC := $(shell $(PYTHON) -c "import nvidia.nccl,pathlib;print(pathlib.Path(nvidia.nccl.__path__[0])/'include')")
NCCL_LIB := $(shell $(PYTHON) -c "import nvidia.nccl,pathlib;print(pathlib.Path(nvidia.nccl.__path__[0])/'lib')")
NCCL_SO  := $(notdir $(firstword $(wildcard $(NCCL_LIB)/libnccl.so $(NCCL_LIB)/libnccl.so.*)))
MPI_PREFIX := $(patsubst %/bin/mpicxx,%,$(MPICXX))
MPI_INC    := -I$(MPI_PREFIX)/include
MPI_LIB    := -L$(MPI_PREFIX)/lib -lmpi -Xlinker -rpath -Xlinker $(MPI_PREFIX)/lib

INCLUDES := -I$(NCCL_INC) -I../../include -I../../../include -I$(CUDA_HOME)/include \
            -I$(EFA_HOME)/include $(MPI_INC)

# Transport objects produced by `make -C ../..` (the non-binding TUs).
EP_OBJS := ../../src/proxy.o ../../src/rdma.o ../../src/common.o \
           ../../src/uccl_proxy.o ../../src/fifo.o ../../src/adaptive_sleeper.o

DEFS  := -DUCCL_NUM_PROXY_THS=$(NUM_PROXY_THS) -DEFA -DNCCL_CHECK_CUDACC=1
NVCCFLAGS := -O3 -std=c++17 -arch=sm_90 $(DEFS) $(INCLUDES) \
             -Xcompiler "-fPIC -pthread"
LDFLAGS := -L$(CUDA_HOME)/lib64 -L$(EFA_HOME)/lib -L$(NCCL_LIB) $(MPI_LIB) \
           -l:$(NCCL_SO) -lefa -libverbs -lnl-3 -lnl-route-3 -lnuma -lcudart -lcuda -lpthread \
           -Xlinker -rpath -Xlinker $(NCCL_LIB)

microbench: microbench.cu $(EP_OBJS)
	$(NVCC) $(NVCCFLAGS) microbench.cu $(EP_OBJS) -o $@ $(LDFLAGS)

clean:
	rm -f microbench

.PHONY: clean
