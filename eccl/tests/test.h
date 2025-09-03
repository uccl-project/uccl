#pragma once

#include <iostream>

void test_communicator();
void test_find_best_rdma_for_gpu(int gpu_id);
void test_redis_oob();
void test_generate_host_id();
void test_meta_exchange_multi_threads(int world_size);