#include "rdma_addressing.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace {

std::array<uint8_t, 16> remote_gid() {
  return {0x20, 0x01, 0x0d, 0xb8, 0x00, 0x01, 0x00, 0x02,
          0x00, 0x03, 0x00, 0x04, 0x00, 0x05, 0x00, 0x06};
}

void test_required_gid_policy() {
  struct ibv_port_attr port_attr = {};
  port_attr.link_layer = IBV_LINK_LAYER_INFINIBAND;
  assert(!rdma_port_requires_gid(port_attr));
  assert(rdma_gid_index_for_port(port_attr, 5) == 5);

  port_attr.flags = IBV_QPF_GRH_REQUIRED;
  assert(rdma_port_requires_gid(port_attr));
  assert(rdma_gid_index_for_port(port_attr, 7) == 7);

  port_attr.flags = 0;
  port_attr.link_layer = IBV_LINK_LAYER_ETHERNET;
  assert(rdma_port_requires_gid(port_attr));
  assert(rdma_gid_index_for_port(port_attr, 3) == 3);

  port_attr.link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  assert(rdma_port_requires_gid(port_attr));
  assert(rdma_gid_index_for_port(port_attr, 9) == 0);
}

void test_local_infiniband_address_vector() {
  struct ibv_port_attr port_attr = {};
  port_attr.link_layer = IBV_LINK_LAYER_INFINIBAND;
  auto const gid = remote_gid();
  struct ibv_ah_attr ah_attr;
  memset(&ah_attr, 0xff, sizeof(ah_attr));

  configure_qp_address_vector(&ah_attr, port_attr, 0x1234, gid.data(), 7, 0, 0);

  assert(ah_attr.is_global == 0);
  assert(ah_attr.dlid == 0x1234);
  assert(ah_attr.port_num == 1);
  assert(ah_attr.sl == 0);
  assert(ah_attr.src_path_bits == 0);
  assert(ah_attr.static_rate == 0);
  struct ibv_global_route empty_grh = {};
  assert(memcmp(&ah_attr.grh, &empty_grh, sizeof(empty_grh)) == 0);
}

void test_grh_required_infiniband_address_vector() {
  struct ibv_port_attr port_attr = {};
  port_attr.link_layer = IBV_LINK_LAYER_INFINIBAND;
  port_attr.flags = IBV_QPF_GRH_REQUIRED;
  auto const gid = remote_gid();
  struct ibv_ah_attr ah_attr;
  memset(&ah_attr, 0xff, sizeof(ah_attr));

  configure_qp_address_vector(&ah_attr, port_attr, 0x1234, gid.data(), 7, 0, 0);

  assert(ah_attr.is_global == 1);
  assert(ah_attr.dlid == 0x1234);
  assert(ah_attr.port_num == 1);
  assert(ah_attr.sl == 0);
  assert(ah_attr.src_path_bits == 0);
  assert(ah_attr.static_rate == 0);
  assert(memcmp(ah_attr.grh.dgid.raw, gid.data(), gid.size()) == 0);
  assert(ah_attr.grh.flow_label == 0);
  assert(ah_attr.grh.sgid_index == 7);
  assert(ah_attr.grh.hop_limit == 255);
  assert(ah_attr.grh.traffic_class == 0);
}

void test_roce_address_vector() {
  struct ibv_port_attr port_attr = {};
  port_attr.link_layer = IBV_LINK_LAYER_ETHERNET;
  auto const gid = remote_gid();
  struct ibv_ah_attr ah_attr;
  memset(&ah_attr, 0xff, sizeof(ah_attr));

  configure_qp_address_vector(&ah_attr, port_attr, 0, gid.data(), 3, 5, 104);

  assert(ah_attr.is_global == 1);
  assert(ah_attr.port_num == 1);
  assert(ah_attr.sl == 5);
  assert(ah_attr.src_path_bits == 0);
  assert(ah_attr.static_rate == 0);
  assert(memcmp(ah_attr.grh.dgid.raw, gid.data(), gid.size()) == 0);
  assert(ah_attr.grh.flow_label == 0);
  assert(ah_attr.grh.sgid_index == 3);
  assert(ah_attr.grh.hop_limit == 255);
  assert(ah_attr.grh.traffic_class == 104);
}

void test_asymmetric_infiniband_address_vectors() {
  struct ibv_port_attr local_port = {};
  local_port.link_layer = IBV_LINK_LAYER_INFINIBAND;
  local_port.flags = IBV_QPF_GRH_REQUIRED;
  struct ibv_port_attr remote_port = {};
  remote_port.link_layer = IBV_LINK_LAYER_INFINIBAND;
  auto const gid_from_remote_selected_index = remote_gid();

  assert(rdma_gid_index_for_port(remote_port, 11) == 11);

  struct ibv_ah_attr local_ah_attr;
  configure_qp_address_vector(&local_ah_attr, local_port, 0x1234,
                              gid_from_remote_selected_index.data(), 7, 0, 0);
  assert(local_ah_attr.is_global == 1);
  assert(local_ah_attr.dlid == 0x1234);
  assert(memcmp(local_ah_attr.grh.dgid.raw,
                gid_from_remote_selected_index.data(),
                gid_from_remote_selected_index.size()) == 0);
  assert(local_ah_attr.grh.sgid_index == 7);

  struct ibv_ah_attr remote_ah_attr;
  configure_qp_address_vector(&remote_ah_attr, remote_port, 0x5678,
                              remote_gid().data(), 11, 0, 0);
  assert(remote_ah_attr.is_global == 0);
  assert(remote_ah_attr.dlid == 0x5678);
}

}  // namespace

int main() {
  test_required_gid_policy();
  test_local_infiniband_address_vector();
  test_grh_required_infiniband_address_vector();
  test_roce_address_vector();
  test_asymmetric_infiniband_address_vectors();
  return 0;
}
