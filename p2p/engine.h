#pragma once

#include "transport.h"
#include "util/gpu_rt.h"
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
// #define USE_REDIS
#ifdef USE_REDIS
#include <sw/redis++/redis++.h>
#endif

namespace py = pybind11;
constexpr uint64_t kNvlinkConn = UINT64_MAX;

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

struct PeerInfo {
  std::string ip_addr;  // IP address of the peer
  int gpu_idx;          // GPU index of the peer
};

class Endpoint {
  const uint64_t kRTTBytes = 1024 * 1024;
  const uint64_t kChunkSize = 512 * 1024;
  const uint32_t kMaxInflightChunks = 8;

 public:
  gpuStream_t pick_stream() {
    if (streams_.empty()) return nullptr;
    uint32_t i =
        rr_stream_.fetch_add(1, std::memory_order_relaxed) % streams_.size();
    return streams_[i];
  }

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
   *   remote_port: the port of the remote server (optional)
   * output:
   *   conn_id: the ID of the connection
   */
  bool connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
               uint64_t& conn_id);

  bool connect(py::bytes const& metadata, uint64_t& conn_id);

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

  bool regv(std::vector<void const*> const& data_v,
            std::vector<size_t> const& size_v, std::vector<uint64_t>& mr_id_v);

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
  bool recv(uint64_t conn_id, uint64_t mr_id, void* data, size_t size);

  bool send_ipc(uint64_t conn_id, uint64_t mr_id, void const* data, size_t size,
                void const* meta, size_t meta_len);

  /* Send data to the remote server asynchronously. */
  bool send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                  size_t size, uint64_t* transfer_id);

  /* Receive data from the remote server asynchronously. */
  bool recv_async(uint64_t conn_id, uint64_t mr_id, void* data, size_t size,
                  uint64_t* transfer_id);

  /* Send a vector of data chunks. Blocking. */
  bool sendv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<void const*> data_v, std::vector<size_t> size_v,
             size_t num_iovs);

  /* Receive a vector of data chunks. Blocking. */
  bool recvv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<void*> data_v, std::vector<size_t> size_v,
             size_t num_iovs);

  /* Read data from the remote server. Blocking.
   *
   * input:
   *   conn_id: the ID of the connection
   *   mr_id: the ID of the data
   *   dst: the destination buffer
   *   size: the size of the data
   *   slot_item: the slot item to use for the transfer
   */
  bool read(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
            uccl::FifoItem const& slot_item);

  /* Read data from the remote server asynchronously. */
  bool read_async(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                  uccl::FifoItem const& slot_item, uint64_t* transfer_id);

  bool advertise(uint64_t conn_id, uint64_t mr_id, void* addr, size_t len,
                 char* out_buf);

  /* Poll the status of the asynchronous receive. */
  bool poll_async(uint64_t transfer_id, bool* is_done);

  /**
   * Join a logical rendezvous group and connect to every other member.
   *
   * This helper publishes (ip, gpu_idx) to an external discovery service (e.g.,
   * Redis, a Ray named actor, etc.) under the given @group_name.  All callers
   * block until @world_size peers have registered.  Connections are then
   * established in rank‑ascending order (lower rank initiates), guaranteeing a
   * fully‑connected clique without duplicate dials.
   *
   * @param discovery_uri  URI for discovery backend. Examples:
   *                       "redis://127.0.0.1:6379" or "ray://actor:Store".
   * @param group_name     Logical namespace so multiple groups can coexist.
   * @param world_size     Total number of expected ranks in the group.
   * @param my_rank        Caller’s rank (0‑based). Must be unique.
   *
   * @returns true on success, false otherwise.
   */
  bool join_group(std::string const& discovery_uri,
                  std::string const& group_name, int world_size, int my_rank,
                  int remote_gpu_idx);

  /**
   * Convenience constructor: create Endpoint and immediately join a group.
   * You may prefer this factory in Ray where each actor knows its rank and the
   * rendezvous, but not its peers’ IP addresses.
   */
  static std::unique_ptr<Endpoint> create_and_join(
      std::string const& discovery_uri, std::string const& group_name,
      int world_size, int my_rank, uint32_t local_gpu_idx, uint32_t num_cpus,
      int remote_gpu_idx);

  /** Returns conn_id for @rank, or UINT64_MAX if unknown. */
  uint64_t conn_id_of_rank(int rank) const;

  std::vector<uint8_t> get_endpoint_metadata();

 private:
  /** Rank‑indexed view of established connections (read‑only). */
  std::unordered_map<int, uint64_t> const& rank2conn() const {
    return rank2conn_;
  }

#ifdef USE_REDIS
  bool publish_redis(std::string const& redis_uri, std::string const& key,
                     PeerInfo const& info);
  bool fetch_all_redis(std::string const& redis_uri,
                       std::string const& key_prefix, int world_size,
                       std::vector<PeerInfo>& out);
#endif

  bool publish_peer(std::string const& discovery_uri,
                    std::string const& group_name, int rank,
                    PeerInfo const& info);
  bool collect_peers(std::string const& discovery_uri,
                     std::string const& group_name, int world_size,
                     std::vector<PeerInfo>& out);

  int local_gpu_idx_;
  int remote_gpu_idx_;
  uint32_t num_cpus_;

  uccl::RDMAEndpoint* ep_;

  std::atomic<uint64_t> next_conn_id_ = 0;
  std::atomic<uint64_t> next_mr_id_ = 0;
  std::atomic<uint64_t> next_transfer_id_ = 0;

  // TODO(yang): add mutex to protect the maps.
  mutable std::mutex conn_mu_;

  std::unordered_map<uint64_t, Conn*> conn_id_to_conn_;
  std::unordered_map<uint64_t, MR*> mr_id_to_mr_;
  std::unordered_map<uint64_t, uccl::ucclRequest*> transfer_id_to_ureq_;

  std::unordered_map<int, uint64_t> rank2conn_;

  // Assuming 1TB GPU memory, 128KB KV block size.
  static constexpr size_t kMaxNumChunksPerTransfer = 1024ul * 1024 * 1024 / 128;
  std::atomic<uint32_t> rr_stream_{0};
  std::vector<gpuStream_t> streams_;
};