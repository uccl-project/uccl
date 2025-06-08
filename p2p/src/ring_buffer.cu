#include "ring_buffer.cuh"
#include <atomic>
#include <new>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
