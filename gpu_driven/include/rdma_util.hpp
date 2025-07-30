#ifndef RDMA_UTIL_HPP
#define RDMA_UTIL_HPP

#include "rdma.hpp"
#include <infiniband/verbs.h>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <vector>

void fill_local_gid(ProxyCtx& S, RDMAConnectionInfo* local_info) {
  if (!S.context) {
    fprintf(stderr, "Error: context not initialized when filling GID\n");
    exit(1);
  }

  // Query port attributes to determine if this is RoCE (Ethernet) or InfiniBand
  struct ibv_port_attr port_attr;
  if (ibv_query_port(S.context, 1, &port_attr)) {
    perror("Failed to query port for GID");
    exit(1);
  }

  // For RoCE (Ethernet), we need to fill the GID
  if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
    union ibv_gid local_gid;
    int gid_index = 1;
    if (ibv_query_gid(S.context, 1, gid_index, &local_gid)) {
      perror("Failed to query GID");
      exit(1);
    }

    // Copy the GID to the connection info
    memcpy(local_info->gid, &local_gid, 16);
    printf(
        "[RDMA] Local GID filled for RoCE (Ethernet) connection: "
        "%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%"
        "02x\n",
        local_info->gid[0], local_info->gid[1], local_info->gid[2],
        local_info->gid[3], local_info->gid[4], local_info->gid[5],
        local_info->gid[6], local_info->gid[7], local_info->gid[8],
        local_info->gid[9], local_info->gid[10], local_info->gid[11],
        local_info->gid[12], local_info->gid[13], local_info->gid[14],
        local_info->gid[15]);
  } else {
    // For InfiniBand, GID is not strictly required, but we can still fill it
    union ibv_gid local_gid;
    if (ibv_query_gid(S.context, 1, 0, &local_gid) == 0) {
      memcpy(local_info->gid, &local_gid, 16);
      printf(
          "[RDMA] Local GID filled for InfiniBand connection: "
          "%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%"
          "02x\n",
          local_info->gid[0], local_info->gid[1], local_info->gid[2],
          local_info->gid[3], local_info->gid[4], local_info->gid[5],
          local_info->gid[6], local_info->gid[7], local_info->gid[8],
          local_info->gid[9], local_info->gid[10], local_info->gid[11],
          local_info->gid[12], local_info->gid[13], local_info->gid[14],
          local_info->gid[15]);
    } else {
      // If GID query fails for InfiniBand, zero it out
      memset(local_info->gid, 0, 16);
      printf(
          "[RDMA] GID zeroed for InfiniBand connection (GID query failed)\n");
    }
  }
}

#endif  // RDMA_UTIL_HPP