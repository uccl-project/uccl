#pragma once

#include "util/cb.h"
#include "util/jring.h"
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

// This is a workaround to avoid blocking the main thread.
inline void long_running_func() {
  py::gil_scoped_release release;
  std::this_thread::sleep_for(std::chrono::seconds(2));
}

struct alignas(64) UcclRequest {
  uint32_t ip_addr;
  uint16_t port;
  int conn_id;
  uint64_t kv_id;
  void* data;
  size_t size;
};

struct alignas(64) UcclResponse {
  uint32_t ip_addr;
  uint16_t port;
  int conn_id;
  uint64_t kv_id;
  bool success;
};

class Channel {
  constexpr static uint32_t kChannelSize = 1024;

 public:
  struct alignas(64) CMD {
    enum Opcode : uint8_t { kConnect, kAccept, kRegKV, kSendKV, kRecvKV };
    Opcode opcode;
    UcclRequest* ureq;
    UcclResponse* uresp;
    uccl::PollCtx* poll_ctx;
  };
  static_assert(sizeof(CMD) % 4 == 0, "Channel::CMD must be 32-bit aligned");

  Channel() {
    tx_cmdq_ = uccl::create_ring(sizeof(CMD), kChannelSize);
    rx_cmdq_ = uccl::create_ring(sizeof(CMD), kChannelSize);
    ctrl_cmdq_ = uccl::create_ring(sizeof(CMD), kChannelSize);
  }

  ~Channel() {
    free(tx_cmdq_);
    free(rx_cmdq_);
    free(ctrl_cmdq_);
  }

  jring_t* tx_cmdq_;
  jring_t* rx_cmdq_;
  jring_t* ctrl_cmdq_;
};

struct MR {
  int mr_id_;
  struct ibv_mr* mr_;
};

struct Conn {
  int conn_id_;
  int port_;
  struct ibv_context* ctx_;
  struct ibv_pd* pd_;
  struct ibv_qp* qp_;
  struct ibv_cq* cq_;
  struct ibv_srq* srq_;
};

class Engine {
 public:
  /*
   * Create engine threads running in background for a single interface. It also
   * opens a TCP listening thread waiting for incoming connections.
   *
   * input:
   *   if_name: the name of the interface to listen on
   *   ncpus: the number of CPUs to use for the engine
   *   nconn_per_cpu: the number of connections per CPU
   *   listen_port: the port to listen on
   */
  Engine(std::string const& if_name, const uint32_t ncpus,
         const uint32_t nconn_per_cpu, const uint16_t listen_port);

  ~Engine();

  /*
   * Connect to a remote server via TCP, then build RDMA QP connections.
   *
   * input:
   *   ip_addr: the IP address of the remote server
   *   port: the port of the remote server
   * output:
   *   conn_id: the ID of the connection
   */
  bool connect(std::string const& ip_addr, uint16_t const& port, int& conn_id);

  /*
   * Accept an incoming connection via TCP, then build RDMA QP connections.
   *
   * output:
   *   ip_addr: the IP address of the remote server
   *   port: the port of the remote server
   *   conn_id: the ID of the connection
   */
  bool accept(std::string& ip_addr, uint16_t& port, int& conn_id);

  /*
   * Register the KV a specific interface. Typically, one KV residing on one GPU
   * only needs to register to one NIC. Even if one KV registers to multiple
   * NICs, the GPU wouldn't have enough PCIe bandwidth for multiple NICs.
   *
   * input:
   *   conn_id: the ID of the connection
   *   data: the data to register
   *   size: the size of the data
   * output:
   *   mr_id: the ID of the MR
   */
  bool reg_kv(int conn_id, void const* data, size_t size, uint64_t& kv_id);

  /*
   * Send a KV to the remote server. Blocking.
   *
   * input:
   *   kv_id: the ID of the KV
   *   data: the data to send
   *   size: the size of the data
   */
  bool send_kv(uint64_t kv_id, void const* data, size_t size);

  /*
   * Receive a KV from the remote server. Blocking.
   *
   * input:
   *   kv_id: the ID of the KV
   * output:
   *   data: the data to receive
   *   size: the size of the data
   */
  bool recv_kv(uint64_t kv_id, void* data, size_t& size);

 private:
  static void* engine_thread(void* arg);

  std::string if_name_;
  uint32_t ncpus_;
  uint32_t nconn_per_cpu_;
  uint16_t listen_port_;
  int listen_fd_;

  std::mutex conn_id_to_conn_mutex_;
  std::unordered_map<int, Conn*> conn_id_to_conn_;
  std::mutex kv_id_to_conn_and_mr_mutex_;
  std::unordered_map<uint64_t, std::tuple<Conn*, MR*>> kv_id_to_conn_and_mr_;

  std::vector<Channel*> channels_;
  std::vector<std::thread> engine_threads_;
};