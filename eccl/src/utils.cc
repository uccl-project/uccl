#include "utils.h"
#include <arpa/inet.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <netdb.h>
#include <unistd.h>

std::tuple<int, std::string> find_best_rdma_for_gpu(int gpu_id) {
  auto gpu_cards = uccl::get_gpu_cards();
  auto rdma_list = uccl::get_rdma_nics();

  if (gpu_id < 0 || gpu_id >= (int)gpu_cards.size() || rdma_list.empty())
    return {-1, ""};

  auto gpu_path = gpu_cards[gpu_id];

  int best_index = -1;
  int best_distance = std::numeric_limits<int>::max();

  for (size_t i = 0; i < rdma_list.size(); ++i) {
    int dist = uccl::cal_pcie_distance(gpu_path, rdma_list[i].second);
    if (dist < best_distance) {
      best_distance = dist;
      best_index = int(i);
    }
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