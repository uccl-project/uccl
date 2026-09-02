#ifndef RDMA_ADDRESSING_HPP
#define RDMA_ADDRESSING_HPP

#include <infiniband/verbs.h>
#include <cstdint>
#include <cstring>

static inline bool rdma_port_uses_grh(struct ibv_port_attr const& port_attr) {
  return port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ||
         (port_attr.flags & IBV_QPF_GRH_REQUIRED) != 0;
}

static inline bool rdma_port_requires_gid(
    struct ibv_port_attr const& port_attr) {
  return port_attr.link_layer == IBV_LINK_LAYER_UNSPECIFIED ||
         rdma_port_uses_grh(port_attr);
}

static inline int rdma_gid_index_for_port(struct ibv_port_attr const& port_attr,
                                          int selected_gid_index) {
  return port_attr.link_layer == IBV_LINK_LAYER_UNSPECIFIED
             ? 0
             : selected_gid_index;
}

static inline void configure_qp_address_vector(
    struct ibv_ah_attr* ah_attr, struct ibv_port_attr const& port_attr,
    uint16_t remote_lid, uint8_t const remote_gid[16], int gid_index,
    int service_level, int traffic_class) {
  memset(ah_attr, 0, sizeof(*ah_attr));
  ah_attr->is_global = rdma_port_uses_grh(port_attr);
  ah_attr->dlid = remote_lid;
  ah_attr->port_num = 1;
  ah_attr->sl = service_level;

  if (!ah_attr->is_global) return;

  memcpy(&ah_attr->grh.dgid, remote_gid, 16);
  ah_attr->grh.sgid_index = gid_index;
  ah_attr->grh.hop_limit = 255;
  ah_attr->grh.traffic_class = traffic_class;
}

#endif  // RDMA_ADDRESSING_HPP
