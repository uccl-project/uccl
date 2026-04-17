#pragma once

#include "../define.h"
#include <cstring>
#include <string>

enum class NicMode { EFA, IB };

inline bool is_nic_usable(std::string const& nic_name, NicMode mode) {
  if (mode == NicMode::EFA && strncmp(nic_name.c_str(), "rdmap", 5) != 0) {
    return false;
  }

  int dev_count = 0;
  ibv_device** dev_list = ibv_get_device_list(&dev_count);
  if (!dev_list) {
    return false;
  }
  bool usable = false;
  for (int i = 0; i < dev_count; ++i) {
    if (std::strcmp(ibv_get_device_name(dev_list[i]), nic_name.c_str()) != 0) {
      continue;
    }
    ibv_context* ctx = ibv_open_device(dev_list[i]);
    if (!ctx) break;

    ibv_port_attr port_attr{};
    if (ibv_query_port(ctx, kPortNum, &port_attr) == 0) {
      if (mode == NicMode::EFA) {
        if (port_attr.state == IBV_PORT_ACTIVE &&
            (port_attr.link_layer == IBV_LINK_LAYER_UNSPECIFIED ||
             port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ||
             port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND)) {
          usable = true;
        }
      } else {
        if (port_attr.state == IBV_PORT_ACTIVE &&
            (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ||
             port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) &&
            (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET ||
             port_attr.gid_tbl_len > 0)) {
          usable = true;
        }
      }
    }
    ibv_close_device(ctx);
    break;
  }
  ibv_free_device_list(dev_list);
  return usable;
}
