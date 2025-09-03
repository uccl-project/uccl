#pragma once

#include "util/util.h"

std::tuple<int, std::string> find_best_rdma_for_gpu(int gpu_id);
std::string generate_host_id(bool include_ip=false);
