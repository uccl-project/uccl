#include "utils.h"
#include <arpa/inet.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <netdb.h>
#include <unistd.h>
#include <infiniband/verbs.h>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

// helper: check if RDMA device (by name) has at least one port UP & LinkUp
static bool rdma_device_has_up_port(const std::string &devname) {
  struct ibv_device **dev_list = ibv_get_device_list(nullptr);
  if (!dev_list) {
    std::cerr << "[rdma] ibv_get_device_list() returned null\n";
    return false;
  }

  struct ibv_device *target = nullptr;
  for (int i = 0; dev_list[i] != nullptr; ++i) {
    const char *name = ibv_get_device_name(dev_list[i]);
    if (name && devname == name) {
      target = dev_list[i];
      break;
    }
  }

  if (!target) {
    ibv_free_device_list(dev_list);
    // no need to log excessively in success path; caller may handle
    return false;
  }

  struct ibv_context *ctx = ibv_open_device(target);
  // free device list as soon as possible
  ibv_free_device_list(dev_list);
  if (!ctx) {
    std::cerr << "[rdma] ibv_open_device(" << devname << ") failed\n";
    return false;
  }

  struct ibv_device_attr dev_attr;
  if (ibv_query_device(ctx, &dev_attr)) {
    std::cerr << "[rdma] ibv_query_device failed for " << devname << "\n";
    ibv_close_device(ctx);
    return false;
  }

  bool any_up = false;
  for (uint8_t port = 1; port <= dev_attr.phys_port_cnt; ++port) {
    struct ibv_port_attr port_attr;
    if (ibv_query_port(ctx, port, &port_attr) != 0) {
      continue;
    }
    bool is_active = (port_attr.state == IBV_PORT_ACTIVE);

#ifdef IBV_PORT_PHYS_STATE_LINK_UP
    is_active = is_active && (port_attr.phys_state == IBV_PORT_PHYS_STATE_LINK_UP);
#endif

    if (is_active) {
      any_up = true;
      break;
    }
  }

  ibv_close_device(ctx);
  return any_up;
}

std::tuple<int, std::string> find_best_rdma_for_gpu(int gpu_id) {
  auto gpu_cards = uccl::get_gpu_cards();
  auto rdma_list = uccl::get_rdma_nics(); // vector<pair<name, path>> or similar

  if (gpu_id < 0 || gpu_id >= (int)gpu_cards.size() || rdma_list.empty())
    return {-1, ""};

  auto gpu_path = gpu_cards[gpu_id];

  // First, build list of candidate indices that are UP
  std::vector<int> up_indices;
  for (size_t i = 0; i < rdma_list.size(); ++i) {
    const std::string &devname = rdma_list[i].first;
    bool up = rdma_device_has_up_port(devname);
    if (up) up_indices.push_back(int(i));
    else {
      std::cerr << "[WARN] RDMA device " << devname << " appears DOWN/Disabled - skipping\n";
    }
  }

  int best_index = -1;
  int best_distance = std::numeric_limits<int>::max();

  auto consider = [&](size_t i) {
    int dist = uccl::cal_pcie_distance(gpu_path, rdma_list[i].second);
    if (dist < best_distance) {
      best_distance = dist;
      best_index = int(i);
    }
  };

  if (!up_indices.empty()) {
    // choose among up devices
    for (int idx : up_indices) consider(idx);
  } else {
    // no device is up — depending on policy either pick nearest anyway or return failure
    std::cerr << "[WARN] No RDMA devices with active/link-up ports found. Falling back to nearest device regardless of port state.\n";
    for (size_t i = 0; i < rdma_list.size(); ++i) consider(i);
    // alternatively, to fail instead of fallback, uncomment:
    // return {-1, ""};
  }

  if (best_index == -1) return {-1, ""};
  return {best_index, rdma_list[best_index].first};
}

std::string get_hostname() {
  char buf[256] = {0};
  if (gethostname(buf, sizeof(buf)) == 0) {
    return std::string(buf);
  }
  return "unknown";
}

std::string get_primary_ip() {
  char hostname[256] = {0};
  if (gethostname(hostname, sizeof(hostname)) != 0) return "0.0.0.0";

  struct hostent* he = gethostbyname(hostname);
  if (!he || !he->h_addr_list[0]) return "0.0.0.0";

  struct in_addr** addr_list = (struct in_addr**)he->h_addr_list;
  return std::string(inet_ntoa(*addr_list[0]));
}

std::string read_machine_id() {
  std::ifstream f("/etc/machine-id");
  if (!f.is_open()) return "";
  std::string id;
  std::getline(f, id);
  f.close();
  return id;
}

std::string generate_host_id(bool include_ip) {
  std::string id = read_machine_id();
  if (!id.empty()) {
    if (include_ip) {
      id += "-" + get_primary_ip();
    }
    return id;
  }

  id = get_hostname();
  if (include_ip) {
    id += "-" + get_primary_ip();
  }
  return id;
}