#pragma once

#include "util/util.h"

namespace UKernel {
namespace Transport {

std::string generate_host_id(bool include_ip = false);
void cleanup_ipc_shm();

}  // namespace Transport
}  // namespace UKernel
