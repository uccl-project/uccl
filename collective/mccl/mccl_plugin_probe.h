#ifndef UCCL_MCCL_PLUGIN_PROBE_H_
#define UCCL_MCCL_PLUGIN_PROBE_H_

#include "mccl_net_abi.h"
#include <string>

namespace uccl {

struct McclPluginProbeResult {
  bool loaded = false;
  std::string error;
  mcclNet_v11_t const* plugin = nullptr;
};

McclPluginProbeResult probe_mccl_net_plugin(char const* path);

}  // namespace uccl

#endif  // UCCL_MCCL_PLUGIN_PROBE_H_
