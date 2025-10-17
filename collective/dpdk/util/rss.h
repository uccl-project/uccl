#pragma once

#include "util.h"
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <shared_mutex>
#include <vector>
#include <sys/ioctl.h>
#include <unistd.h>

namespace uccl {

// Function to retrieve the redirection table and RSS key
inline bool get_rss_config(std::string const& interface_name,
                           std::vector<uint32_t>& redir_table,
                           std::vector<uint8_t>& rss_key) {
  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0) {
    perror("Socket creation failed");
    return false;
  }

  // Prepare structures for ioctl
  struct ifreq ifr;
  memset(&ifr, 0, sizeof(ifr));
  strncpy(ifr.ifr_name, interface_name.c_str(), IFNAMSIZ - 1);

  // Step 1: Query the sizes of the RSS key and redirection table
  struct ethtool_rxfh* rss_query =
      (struct ethtool_rxfh*)malloc(sizeof(struct ethtool_rxfh));
  memset(rss_query, 0, sizeof(struct ethtool_rxfh));
  rss_query->cmd = ETHTOOL_GRSSH;  // Get RSS configuration command

  ifr.ifr_data = reinterpret_cast<char*>(rss_query);

  if (ioctl(sockfd, SIOCETHTOOL, &ifr) < 0) {
    perror("Failed to query RSS configuration sizes");
    free(rss_query);
    close(sockfd);
    return false;
  }

  uint32_t indir_size = rss_query->indir_size;
  uint32_t key_size = rss_query->key_size;
  free(rss_query);

  LOG(INFO) << "Interface " << interface_name << ": RSS indirection table size "
            << indir_size << ", RSS key size " << key_size;

  if (indir_size == 0 && key_size == 0) {
    std::cerr << "RSS configuration not supported on this NIC." << std::endl;
    close(sockfd);
    return false;
  }

  // Step 2: Allocate memory for the full RSS configuration
  size_t struct_size =
      sizeof(struct ethtool_rxfh) + indir_size * sizeof(uint32_t) + key_size;
  struct ethtool_rxfh* rss_config = (struct ethtool_rxfh*)malloc(struct_size);
  memset(rss_config, 0, struct_size);

  rss_config->cmd = ETHTOOL_GRSSH;
  rss_config->indir_size = indir_size;
  rss_config->key_size = key_size;

  ifr.ifr_data = reinterpret_cast<char*>(rss_config);

  // Step 3: Perform ioctl to retrieve the RSS configuration
  if (ioctl(sockfd, SIOCETHTOOL, &ifr) < 0) {
    perror("Failed to retrieve RSS configuration");
    free(rss_config);
    close(sockfd);
    return false;
  }

  // Copy the redirection table
  if (indir_size > 0) {
    uint32_t* indir_table_start = (uint32_t*)rss_config->rss_config;
    redir_table.assign(indir_table_start, indir_table_start + indir_size);
  }

  // Copy the RSS key
  if (key_size > 0) {
    uint8_t* key_start =
        (uint8_t*)((uint32_t*)rss_config->rss_config + indir_size);
    rss_key.assign(key_start, key_start + key_size);
  }

  free(rss_config);
  close(sockfd);
  return true;
}

inline void rte_convert_rss_key(uint32_t const* orig, uint32_t* targ, int len) {
  int i;
  for (i = 0; i < (len >> 2); i++) targ[i] = __builtin_bswap32(orig[i]);
}

inline uint32_t rte_softrss(uint32_t* input_tuple, uint32_t input_len,
                            uint8_t const* rss_key) {
  uint32_t i, j, map, ret = 0;

  for (j = 0; j < input_len; j++) {
    for (map = input_tuple[j]; map; map &= (map - 1)) {
      i = __builtin_ctz(map);
      ret ^= __builtin_bswap32(((uint32_t const*)rss_key)[j]) << (31 - i) |
             (uint32_t)((uint64_t)(__builtin_bswap32(
                            ((uint32_t const*)rss_key)[j + 1])) >>
                        (i + 1));
    }
  }
  return ret;
}

struct rte_ipv4_tuple {
  uint32_t src_addr;
  uint32_t dst_addr;
  uint16_t dport;
  uint16_t sport;
};

constexpr int RTE_THASH_V4_L4_LEN = ((sizeof(struct rte_ipv4_tuple)) / 4);

inline uint32_t calculate_rss_hash(uint32_t src_ip, uint32_t dst_ip,
                                   uint16_t src_port, uint16_t dst_port,
                                   std::vector<uint8_t> const& rss_key) {
  // TODO(yang): Why using struct will given a wrong and inconsistent result?
  // struct rte_ipv4_tuple tuple = {
  //     .src_addr = src_ip,
  //     .dst_addr = dst_ip,
  //     .dport = dst_port,
  //     .sport = src_port,
  // };
  uint32_t tuple[RTE_THASH_V4_L4_LEN];
  tuple[0] = src_ip;
  tuple[1] = dst_ip;
  tuple[2] = (uint32_t)dst_port | ((uint32_t)src_port << 16);
  return rte_softrss((uint32_t*)&tuple, RTE_THASH_V4_L4_LEN, rss_key.data());
}

// Function to calculate queue ID directly from IPs and ports
inline uint32_t calculate_queue_id(uint32_t src_ip, uint32_t dst_ip,
                                   uint16_t src_port, uint16_t dst_port,
                                   std::vector<uint8_t> const& rss_key,
                                   std::vector<uint32_t> const& redir_table) {
  // Step 1: Calculate the RSS hash
  auto rss_hash =
      calculate_rss_hash(src_ip, dst_ip, src_port, dst_port, rss_key);

  // Step 2: Map the hash to the redirection table
  uint32_t queue_id = redir_table[rss_hash % redir_table.size()];

  return queue_id;
}

inline bool get_dst_ports_with_target_queueid(
    uint32_t src_ip, uint32_t dst_ip, uint16_t src_port,
    uint32_t target_queue_id, std::vector<uint8_t> const& rss_key,
    std::vector<uint32_t> const& redir_table, int num_dst_ports,
    std::vector<uint16_t>& dst_ports) {
  for (int port = BASE_PORT; port < 65536; port++) {
    uint16_t dst_port = port;
    uint32_t queue_id = calculate_queue_id(src_ip, dst_ip, src_port, dst_port,
                                           rss_key, redir_table);
    if (queue_id == target_queue_id) {
      dst_ports.push_back(dst_port);
      if (dst_ports.size() == num_dst_ports) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace uccl
