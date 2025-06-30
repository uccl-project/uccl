#pragma once

#include "transport.h"
#include "util/jring.h"
#include "util/shared_pool.h"
#include "util/util.h"
#include <infiniband/verbs.h>
#include <pybind11/pybind11.h>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

struct MR {
  uint64_t mr_id_;
  uccl::Mhandle* mhandle_;
};

struct Conn {
  uint64_t conn_id_;
  uccl::ConnID uccl_conn_id_;
  std::string ip_addr_;
  int remote_gpu_idx_;
};

class Endpoint {
 public:
  /*
   * Create engine threads running in background for a single interface. It also
   * opens a TCP listening thread waiting for incoming connections.
   *
   * input:
   *   local_gpu_idx: the GPU index to use for the engine
   *   num_cpus: the number of CPUs to use for the engine
   */
  Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus);

  ~Endpoint();

  /*
   * Connect to a remote server via TCP, then build RDMA QP connections.
   *
   * input:
   *   ip_addr: the IP address of the remote server
   *   remote_gpu_idx: the GPU index of the remote server
   * output:
   *   conn_id: the ID of the connection
   */
  bool connect(std::string const& ip_addr, int const& remote_gpu_idx,
               uint64_t& conn_id);

  /*
   * Accept an incoming connection via TCP, then build RDMA QP connections.
   *
   * output:
   *   ip_addr: the IP address of the remote server
   *   remote_gpu_idx: the GPU index of the remote server
   *   conn_id: the ID of the connection
   */
  bool accept(std::string& ip_addr, int& remote_gpu_idx, uint64_t& conn_id);

  /*
   * Register the data with a specific interface. Typically, one data residing
   * on one GPU only needs to register to one NIC. Even if the data is
   * registered to multiple NICs, the GPU wouldn't have enough PCIe bandwidth
   * for multiple NICs.
   *
   * input:
   *   data: the data to register
   *   size: the size of the data
   * output:
   *   mr_id: the ID of the MR
   */
  bool reg(void const* data, size_t size, uint64_t& mr_id);

  /*
   * Send data to the remote server. Blocking.
   *
   * input:
   *   conn_id: the ID of the connection
   *   mr_id: the ID of the data
   *   data: the data to send
   *   size: the size of the data
   */
  bool send(uint64_t conn_id, uint64_t mr_id, void const* data, size_t size);

  /*
   * Receive data from the remote server. Blocking.
   *
   * input:
   *   conn_id: the ID of the connection
   *   mr_id: the ID of the data
   * output:
   *   data: the data to receive
   *   size: the size of the data
   */
  bool recv(uint64_t conn_id, uint64_t mr_id, void* data, size_t max_size,
            size_t& recv_size);

 private:
  int local_gpu_idx_;
  uint32_t num_cpus_;

  uccl::RDMAEndpoint* ep_;

  std::atomic<uint64_t> next_conn_id_ = 0;
  std::atomic<uint64_t> next_mr_id_ = 0;

  // TODO(yang): add mutex to protect the maps.
  std::unordered_map<uint64_t, Conn*> conn_id_to_conn_;
  std::unordered_map<uint64_t, MR*> mr_id_to_mr_;
};