#!/bin/bash

# Compilation command for memory_allocator_example.cpp

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

g++ -O3 -g memory_allocator_example.cpp -o memory_allocator_example \
    -I /opt/amazon/efa/include \
    -L /opt/amazon/efa/lib \
    -lfabric -libverbs -lefa \
    -lpthread \
    -I${CUDA_HOME}/include \
    -L${CUDA_HOME}/lib64 \
    -lcudart -lcuda \
    -I ../../include \
    -std=c++17

echo "Compilation complete. Run with: ./memory_allocator_example"

# g++ -O3 -g test_ring.cpp -o test_ring \
#     -I /opt/amazon/efa/include \
#     -L /opt/amazon/efa/lib \
#     -lfabric -libverbs -lefa \
#     -lpthread \
#     -I${CUDA_HOME}/include \
#     -L${CUDA_HOME}/lib64 \
#     -lcudart -lcuda \
#     -I ../../include \
#     -std=c++17
