#ifndef TESTS_INCLUDE_INFINIBAND_VERBS_H
#define TESTS_INCLUDE_INFINIBAND_VERBS_H

#include <cstdint>

enum ibv_link_layer {
  IBV_LINK_LAYER_UNSPECIFIED,
  IBV_LINK_LAYER_INFINIBAND,
  IBV_LINK_LAYER_ETHERNET,
};

enum ibv_query_port_flags {
  IBV_QPF_GRH_REQUIRED = 1 << 0,
};

union ibv_gid {
  uint8_t raw[16];
};

struct ibv_global_route {
  union ibv_gid dgid;
  uint32_t flow_label;
  uint8_t sgid_index;
  uint8_t hop_limit;
  uint8_t traffic_class;
};

struct ibv_ah_attr {
  struct ibv_global_route grh;
  uint16_t dlid;
  uint8_t sl;
  uint8_t src_path_bits;
  uint8_t static_rate;
  uint8_t is_global;
  uint8_t port_num;
};

struct ibv_port_attr {
  uint32_t flags;
  enum ibv_link_layer link_layer;
};

#endif  // TESTS_INCLUDE_INFINIBAND_VERBS_H
