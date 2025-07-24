#include "uccl_engine.h"
#include "engine.h"
#include <arpa/inet.h>
#include <cstring>
#include <string>
#include <vector>

struct uccl_engine {
  Endpoint* endpoint;
};

struct uccl_conn {
  uint64_t conn_id;
  uccl_engine* engine;
};

struct uccl_mr {
  uint64_t mr_id;
  uccl_engine* engine;
};

uccl_engine_t* uccl_engine_create(int local_gpu_idx, int num_cpus) {
  uccl_engine_t* eng = new uccl_engine;
  eng->endpoint = new Endpoint(local_gpu_idx, num_cpus);
  return eng;
}

void uccl_engine_destroy(uccl_engine_t* engine) {
  if (engine) {
    delete engine->endpoint;
    delete engine;
  }
}

uccl_conn_t* uccl_engine_connect(uccl_engine_t* engine, char const* ip_addr,
                                 int remote_gpu_idx, int remote_port) {
  if (!engine || !ip_addr) return nullptr;
  uccl_conn_t* conn = new uccl_conn;
  uint64_t conn_id;
  bool ok = engine->endpoint->connect(std::string(ip_addr), remote_gpu_idx,
                                      conn_id, remote_port);
  if (!ok) {
    delete conn;
    return nullptr;
  }
  conn->conn_id = conn_id;
  conn->engine = engine;
  return conn;
}

uccl_conn_t* uccl_engine_accept(uccl_engine_t* engine, char* ip_addr_buf,
                                size_t ip_addr_buf_len, int* remote_gpu_idx) {
  if (!engine || !ip_addr_buf || !remote_gpu_idx) return nullptr;
  uccl_conn_t* conn = new uccl_conn;
  std::string ip_addr;
  uint64_t conn_id;
  int gpu_idx;
  bool ok = engine->endpoint->accept(ip_addr, gpu_idx, conn_id);
  if (!ok) {
    delete conn;
    return nullptr;
  }
  std::strncpy(ip_addr_buf, ip_addr.c_str(), ip_addr_buf_len);
  *remote_gpu_idx = gpu_idx;
  conn->conn_id = conn_id;
  conn->engine = engine;
  return conn;
}

uccl_mr_t* uccl_engine_reg(uccl_engine_t* engine, void* data, size_t size) {
  if (!engine || !data) return nullptr;
  uccl_mr_t* mr = new uccl_mr;
  uint64_t mr_id;
  bool ok = engine->endpoint->reg(data, size, mr_id);
  if (!ok) {
    delete mr;
    return nullptr;
  }
  mr->mr_id = mr_id;
  mr->engine = engine;
  return mr;
}

int uccl_engine_send(uccl_conn_t* conn, uccl_mr_t* mr, void const* data,
                     size_t size) {
  if (!conn || !mr || !data) return -1;
  return conn->engine->endpoint->send(conn->conn_id, mr->mr_id, data, size)
             ? 0
             : -1;
}

int uccl_engine_recv(uccl_conn_t* conn, uccl_mr_t* mr, void* data,
                     size_t max_size, size_t* recv_size) {
  if (!conn || !mr || !data || !recv_size) return -1;
  return conn->engine->endpoint->recv(conn->conn_id, mr->mr_id, data, max_size,
                                      recv_size)
             ? 0
             : -1;
}

void uccl_engine_conn_destroy(uccl_conn_t* conn) { delete conn; }

void uccl_engine_mr_destroy(uccl_mr_t* mr) { delete mr; }

int uccl_engine_get_metadata(uccl_engine_t* engine, char** metadata) {
  if (!engine || !metadata) return -1;

  try {
    std::vector<uint8_t> metadata_vec =
        engine->endpoint->get_endpoint_metadata();

    // Build string representation
    std::string result;
    if (metadata_vec.size() == 10) {  // IPv4 format
      std::string ip_addr = std::to_string(metadata_vec[0]) + "." +
                            std::to_string(metadata_vec[1]) + "." +
                            std::to_string(metadata_vec[2]) + "." +
                            std::to_string(metadata_vec[3]);
      uint16_t port = (metadata_vec[4] << 8) | metadata_vec[5];
      int gpu_idx;
      std::memcpy(&gpu_idx, &metadata_vec[6], sizeof(int));

      result = "" + ip_addr + ":" + std::to_string(port) + "?" +
               std::to_string(gpu_idx);
    } else if (metadata_vec.size() == 22) {  // IPv6 format
      char ip6_str[INET6_ADDRSTRLEN];
      struct in6_addr ip6_addr;
      std::memcpy(&ip6_addr, metadata_vec.data(), 16);
      inet_ntop(AF_INET6, &ip6_addr, ip6_str, sizeof(ip6_str));

      uint16_t port = (metadata_vec[16] << 8) | metadata_vec[17];
      int gpu_idx;
      std::memcpy(&gpu_idx, &metadata_vec[18], sizeof(int));

      result = "" + std::string(ip6_str) + "]:" + std::to_string(port) + "?" +
               std::to_string(gpu_idx);
    } else {  // Fallback: return hex representation
      result = "";
      for (size_t i = 0; i < metadata_vec.size(); ++i) {
        char hex[3];
        snprintf(hex, sizeof(hex), "%02x", metadata_vec[i]);
        result += hex;
      }
    }

    *metadata = new char[result.length() + 1];
    std::strcpy(*metadata, result.c_str());

    return 0;
  } catch (...) {
    return -1;
  }
}