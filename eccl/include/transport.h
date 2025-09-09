#pragma once

#include "config.h"
#include "oob.h"
#include "request.h"
#include "util/gpu_rt.h"
#include <infiniband/verbs.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

class Communicator;

class EndpointBase {
  virtual bool send_async(int to_rank, std::shared_ptr<Request> creq) = 0;
  virtual bool recv_async(int from_rank, std::shared_ptr<Request> creq) = 0;
  virtual bool poll_reqs(std::vector<std::shared_ptr<Request>>& creqs) = 0;
};

class RDMAEndpoint : public EndpointBase {
 public:
  RDMAEndpoint(std::shared_ptr<Config> config, Communicator* comm);
  ~RDMAEndpoint();

  bool connect_to(int rank);
  bool accept_from(int rank);

  bool send_async(int to_rank, std::shared_ptr<Request> creq) override;
  bool recv_async(int from_rank, std::shared_ptr<Request> creq) override;
  bool poll_reqs(std::vector<std::shared_ptr<Request>>& creqs) override;

 private:
  std::vector<ibv_qp*> qp_list_;
  mutable std::mutex qp_list_mu_;
  std::vector<QpInfo> qp_info_list_;
  mutable std::mutex qp_info_list_mu_;
  std::vector<QpInfo> remote_qp_info_list_;
  mutable std::mutex remote_qp_info_list_mu_;

  std::shared_ptr<Config> config_;
  Communicator* comm_;
};

struct IPCcontext {
  gpuIpcMemHandle_t handle;
  bool is_send;
  void* direct_ptr;
  uintptr_t offset;
  size_t size;
};

class IPCEndpoint : public EndpointBase {
 public:
  IPCEndpoint(std::shared_ptr<Config> config, Communicator* comm);
  ~IPCEndpoint();

  bool connect_to(int rank);
  bool accept_from(int rank);

  bool send_async(int to_rank, std::shared_ptr<Request> creq) override;
  bool recv_async(int from_rank, std::shared_ptr<Request> creq) override;
  bool poll_reqs(std::vector<std::shared_ptr<Request>>& creqs) override;

 private:
  std::shared_ptr<IPCcontext> ipc_context;
  std::shared_ptr<Config> config_;
  Communicator* comm_;
};

struct IpcCache {
  gpuIpcMemHandle_t handle;
  bool is_send;
  void* direct_ptr;  // for remote
  uintptr_t offset;
  size_t size;
};

// one gpu with best nic
class Communicator {
 public:
  Communicator(int gpu_id, int rank, int world_size,
               std::shared_ptr<Config> config = std::make_shared<Config>());
  ~Communicator();

  // ---------- Endpoint -------------
  bool connect_to(int rank);
  bool accept_from(int rank);
  std::tuple<std::shared_ptr<EndpointBase>, bool> get_endpoint_by_rank(
      int rank);

  // ---------- Communication ----------
  bool isend(int rank, std::shared_ptr<Request> req);
  bool irecv(int rank, std::shared_ptr<Request> req);
  bool irecv_red(int rank, std::shared_ptr<Request> req);
  bool poll_finish(std::vector<std::shared_ptr<Request>>& reqs);

  // ---------- Meta info -------------
  void set_communicator_meta_with_rank(int rank, CommunicatorMeta const& meta);
  std::shared_ptr<CommunicatorMeta> get_communicator_meta_by_rank(int rank);
  bool check_ready();

  // ---------- RDMA / IPC helpers ----
  bool register_local_mr(void* local_buf, size_t len);
  bool register_remote_mr(int remote_rank, void* local_buf, ibv_mr* remote_mr);
  ibv_mr* get_local_mr(void* local_buf);
  ibv_mr* get_remote_mr(int remote_rank, void* local_buf);

  bool register_local_ipc_cache(void* local_buf);
  bool register_remote_ipc_cache(int remote_rank, void* local_buf,
                                 IpcCache const& cache);
  IpcCache get_local_ipc_cache(void* local_buf);
  IpcCache get_remote_ipc_cache(int remote_rank, void* local_buf);

  ibv_cq* get_cq_by_index(int index);

 private:
  // ---------- Endpoint -------------
  std::unordered_map<int, std::shared_ptr<EndpointBase>> rank_to_endpoint_;
  mutable std::mutex ep_mu_;

  // ---------- Communicator Meta ----
  std::unordered_map<int, std::shared_ptr<CommunicatorMeta>> rank_to_comm_meta_;
  mutable std::mutex meta_mu_;

  // ---------- GPU / NIC info --------
  int gpu_id_;
  int local_rank_;
  int world_size_;
  bool support_rdma;
  bool support_rdma_roce;
  struct ibv_context* nic_ibv_ctx_{nullptr};
  uint16_t lid;     // Local ID
  uint8_t gid[16];  // Global ID for RoCE (optional)
  enum ibv_mtu active_mtu;

  // ---------- RDMA resources --------
  ibv_pd* pd_{nullptr};
  std::vector<ibv_cq*> cq_list_;
  std::unordered_map<void*, ibv_mr*> ptr_to_local_mr_;  // local ptr -> local mr
  std::unordered_map<int, std::unordered_map<void*, ibv_mr*>>
      rank_ptr_to_remote_mr_;  // to_rank remote_ptr -> remote_mr
  mutable std::mutex local_mr_mu_;
  mutable std::mutex remote_mr_mu_;

  // ---------- IPC resources ---------
  std::unordered_map<void*, IpcCache> ptr_to_local_ipc_cache_;
  std::unordered_map<int, std::unordered_map<void*, IpcCache>>
      rank_ptr_to_ipc_cache_;
  mutable std::mutex local_ipc_cache_mu_;
  mutable std::mutex remote_ipc_cache_mu_;

  // ---------- Config & Redis --------
  std::shared_ptr<Config> config_;
  std::shared_ptr<RedisExchanger> redis_client_;
  mutable std::mutex redis_client_mu_;

  friend class RDMAEndpoint;
  friend class IPCEndpoint;
};
