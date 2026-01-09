#pragma once
#include "define.h"
#include "epoll_client.h"
#include "epoll_server.h"
#include "memory_allocator.h"
#include "rdma_channel_group.h"
#include "rdma_context.h"
#include "rdma_ctrl_channel.h"
#include "rdma_device.h"
#include "util/net.h"
#include <glog/logging.h>

class NICEndpoint {
 public:
  explicit NICEndpoint(
      int gpu_index = INVALID_GPU, uint64_t rank_id = INVALID_RANK_ID,
      uint64_t port = 0, bool auto_start_polling = true,
      std::vector<size_t> const& device_ids = std::vector<size_t>())
      : gpu_index_(gpu_index),
        rank_id_(rank_id),
        auto_start_polling_(auto_start_polling),
        send_id_(0),
        recv_id_(0) {
    if (gpu_index != INVALID_GPU) {
      std::vector<size_t> actual_device_ids;
      if (device_ids.size() == 0) {
        actual_device_ids =
            RdmaDeviceManager::instance().get_best_dev_idx(gpu_index);
      } else {
        actual_device_ids = device_ids;
      }
      initializeContexts(actual_device_ids);
      LOG(INFO) << "NICEndpoint initialized with " << contexts_.size()
                << " context(s) for GPU " << gpu_index;
    }

    oob_server_ = std::make_shared<EpollServer>(
        port, [this](std::string const& input, std::string& output,
                     std::string const& ip, int port) {
          this->process_meta(input, output, ip, port);
        });
    oob_client_ = std::make_shared<EpollClient>();

    allocator_ = std::make_shared<MemoryAllocator>();
    assert(oob_server_->start());
    assert(oob_client_->start());
  }

  // Destructor
  ~NICEndpoint() {
    if (oob_client_) {
      oob_client_->stop();
    }
    if (oob_server_) {
      oob_server_->stop();
    }
  }

  int gpuIndex() const { return gpu_index_; }

  size_t contextCount() const { return contexts_.size(); }

  bool regMem(std::shared_ptr<RegMemBlock> reg_block) {
    if (unlikely(!reg_block)) {
      LOG(ERROR) << "Error: regMem called with null reg_block";
      return false;
    }

    for (size_t context_id = 0; context_id < contexts_.size(); ++context_id) {
      auto context = contexts_[context_id];
      if (unlikely(!context)) {
        LOG(ERROR) << "Error: context at context_id " << context_id
                   << " is null";
        return false;
      }

      struct ibv_mr* mr = context->regMem(reg_block->addr, reg_block->size);

      if (unlikely(!mr)) {
        LOG(ERROR) << "Error: ibv_reg_mr failed for block at "
                   << reg_block->addr << " size " << reg_block->size
                   << " context_id " << context_id;
        return false;
      }
      reg_block->setMRByContextID(context_id, mr);
    }

    return true;
  }

  bool deregMem(std::shared_ptr<RegMemBlock> reg_block) {
    if (unlikely(!reg_block)) {
      LOG(ERROR) << "Error: deregMem called with null reg_block";
      return false;
    }
    for (uint32_t ctx = 0; ctx < kNICContextNumber; ++ctx) {
      ibv_mr* mr = reg_block->getMRByContextID(ctx);
      if (mr) {
        RdmaContext::deregMem(mr);
      }
    }
    return true;
  }

  int build_connect(uint64_t rank_id, bool sync = true,
                    int timeout_ms = 10000) {
    std::string const oob_con = build_oob_connect(rank_id);
    uint64_t receved_rank_id =
        this->build_control_channel(oob_con, rank_id, sync, timeout_ms);
    if (receved_rank_id < 0) {
      return -1;
    }
    if (!this->build_normal_channels(oob_con, rank_id, sync, timeout_ms)) {
      return -1;
    }
    return static_cast<int>(receved_rank_id);
  }

  // Blocking check for send completion
  void checkSendComplete(uint64_t rank_id, int64_t wr_id) {
    LOG(INFO) << "checkSendComplete - rank_id: " << rank_id
              << ", wr_id: " << wr_id;

    auto it = send_channel_groups_.find(rank_id);
    if (it == send_channel_groups_.end()) {
      throw std::runtime_error("Send channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto send_group = it->second;
    while (!send_group->check(wr_id)) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    LOG(INFO) << "checkSendComplete - Completed for rank_id: " << rank_id
              << ", wr_id: " << wr_id;
  }

  bool checkSendComplete_once(uint64_t rank_id, int64_t wr_id) {
    // LOG(INFO) << "checkSendComplete - rank_id: " << rank_id
    //           << ", wr_id: " << wr_id;

    auto it = send_channel_groups_.find(rank_id);
    if (unlikely(it == send_channel_groups_.end())) {
      throw std::runtime_error("Send channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto send_group = it->second;
    return send_group->check(wr_id);
  }

  bool checkRecvComplete_once(uint64_t rank_id, uint64_t index) {
    // LOG(INFO) << "checkRecvComplete - Checking for rank_id: " << rank_id
    //           << ", index: " << index;
    auto it = recv_channel_groups_.find(rank_id);
    if (unlikely(it == recv_channel_groups_.end())) {
      throw std::runtime_error("Recv channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto recv_group = it->second;
    return recv_group->check(index);
  }

  // Blocking check for recv completion
  void checkRecvComplete(uint64_t rank_id, uint64_t index) {
    LOG(INFO) << "checkRecvComplete - Checking for rank_id: " << rank_id
              << ", index: " << index;
    auto it = recv_channel_groups_.find(rank_id);
    if (it == recv_channel_groups_.end()) {
      throw std::runtime_error("Recv channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto recv_group = it->second;
    while (!recv_group->check(index)) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    LOG(INFO) << "checkRecvComplete - Completed for rank_id: " << rank_id
              << ", index: " << index;
  }

  int64_t writeOrRead(std::shared_ptr<RDMASendRequest> req) {
    uint64_t rank_id = req->to_rank_id;
    auto it = send_channel_groups_.find(rank_id);
    if (it == send_channel_groups_.end()) {
      throw std::runtime_error("Send channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto send_group = it->second;
    int64_t wr_id = -1;

    // Blocking call until send succeeds
    while (wr_id < 0) {
      LOG(INFO) << "NICEndpoint::write - Attempting to send to rank_id: "
                << rank_id << ", peer rank_id " << rank_id;
      wr_id = send_group->postWriteOrRead(req);

      if (wr_id < 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }

    return wr_id;
  }

  // Blocking send: wraps SendChannelGroup::send with rank_id parameter
  // Returns wr_id for checking completion later
  int64_t send(uint64_t rank_id, std::shared_ptr<RDMASendRequest> req) {
    auto it = send_channel_groups_.find(rank_id);
    if (it == send_channel_groups_.end()) {
      throw std::runtime_error("Send channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto send_group = it->second;
    int64_t wr_id = -1;

    // Blocking call until send succeeds
    while (wr_id < 0) {
      LOG(INFO) << "NICEndpoint::send - Attempting to send to rank_id: "
                << rank_id << ", peer rank_id " << rank_id;
      wr_id = send_group->send(req);

      if (wr_id < 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }

    return wr_id;
  }

  // Blocking recv: wraps RecvChannelGroup::recv with rank_id parameter
  // Returns index for checking completion later
  int64_t recv(uint64_t rank_id, std::shared_ptr<RDMARecvRequest> req) {
    auto it = recv_channel_groups_.find(rank_id);
    if (it == recv_channel_groups_.end()) {
      throw std::runtime_error("Recv channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto recv_group = it->second;
    int64_t index = -1;
    // Blocking call until recv succeeds
    while (index < 0) {
      index = recv_group->recv(req);
      LOG(INFO) << "NICEndpoint::recv - Attempting to recv from rank_id: "
                << rank_id << ", peer rank_id " << rank_id;
      if (index < 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }

    return index;
  }

  // Add or update rank OOB metadata from a given map
  void add_rank_oob_meta(
      std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>> const&
          new_meta) {
    for (auto const& [rank_id, meta_ptr] : new_meta) {
      rank_oob_meta_[rank_id] = meta_ptr;
    }
  }
  ConnID uccl_connect(int dev, int local_gpuidx, int remote_dev,
                      int remote_gpuidx, std::string remote_ip,
                      uint16_t remote_port) {
    int32_t current_send_id = send_id_.fetch_add(1, std::memory_order_relaxed);

    add_rank_oob_meta({{current_send_id, std::make_shared<OOBMetaData>(
                                             remote_ip, remote_port)}});
    LOG(INFO) << "remote_gpuidx: " << remote_gpuidx
              << ", remote_ip: " << remote_ip
              << ", remote_port: " << remote_port;
    build_connect(current_send_id);  // sync mode (default)
    ConnID conn_id;
    conn_id.context =
        reinterpret_cast<void*>(static_cast<intptr_t>(current_send_id));
    conn_id.flow_id = current_send_id;
    return conn_id;
  };

  inline uint16_t get_p2p_listen_port(int dev) {
    return oob_server_->get_port();
  };

  inline int get_p2p_listen_fd(int dev) {
    return oob_server_->get_listen_fd();
  };

  inline ConnID uccl_accept(int dev, int listen_fd, int local_gpuidx,
                            std::string& remote_ip, int* remote_dev,
                            int* remote_gpuidx) {
    AcceptedMeta accepted;
    uint64_t rank_id = 0;

    // Block until there's an accepted connection
    while (true) {
      {
        {
          std::unique_lock<std::shared_mutex> lock(accepted_meta_mutex_);
          if (!accepted_meta_.empty()) {
            // Get the first accepted connection
            auto it = accepted_meta_.begin();
            rank_id = it->first;
            accepted = it->second;
            // Remove it from the map
            if (getOrCreateRecvGroup(rank_id)->channelCount() ==
                kQpNumPerChannel + 1) {
              accepted_meta_.erase(it);
              LOG(INFO) << "Accepted connection: rank_id=" << rank_id
                        << ", ip=" << accepted.ip << ", port=" << accepted.port
                        << ", gpu_id=" << accepted.gpu_id;
              break;
            }
          }
        }
      }
      // Wait before checking again
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    LOG(INFO) << "Done Accepted connection: rank_id=" << rank_id
              << ", ip=" << accepted.ip << ", port=" << accepted.port
              << ", gpu_id=" << accepted.gpu_id;
    // Assign output parameters
    remote_ip = accepted.ip;
    if (remote_gpuidx != nullptr) {
      *remote_gpuidx = accepted.gpu_id;
    }
    if (remote_dev != nullptr) {
      *remote_dev = 0;  // Default device
    }

    // Create and return ConnID
    ConnID conn_id;
    conn_id.context = reinterpret_cast<void*>(static_cast<intptr_t>(rank_id));
    conn_id.flow_id = rank_id;
    return conn_id;
  }

  inline int uccl_regmr(void* const data, size_t const len, MRArray& mr_array) {
    if (unlikely(!data)) {
      LOG(ERROR) << "Error: uccl_regmr called with null data";
      return -1;
    }

    for (size_t context_id = 0; context_id < contexts_.size(); ++context_id) {
      auto context = contexts_[context_id];
      if (unlikely(!context)) {
        LOG(ERROR) << "Error: context at context_id " << context_id
                   << " is null";
        return -1;
      }

      // Register memory region for this context
      struct ibv_mr* mr = context->regMem(data, len);

      if (unlikely(!mr)) {
        LOG(ERROR) << "Error: ibv_reg_mr failed for data at " << data
                   << " size " << len << " context_id " << context_id;
        return -1;
      }

      // Store the MR in the mr_map using context_id as key
      mr_array.setKeyByContextID(context_id, mr);
    }

    return 0;
  }

  inline void uccl_deregmr(MRArray const& mr_array) {
    for (uint32_t ctx = 0; ctx < kNICContextNumber; ++ctx) {
      ibv_mr* mr = mr_array.getKeyByContextID(ctx);
      if (likely(mr != nullptr)) {
        RdmaContext::deregMem(mr);
      }
    }
  }

  int get_best_dev_idx(int gpu_idx) { return 0; }

  bool initialize_engine_by_dev(int gpu_index, bool enable_p2p_listen) {
    (void)enable_p2p_listen;

    gpu_index_ = gpu_index;

    std::vector<size_t> device_ids =
        RdmaDeviceManager::instance().get_best_dev_idx(gpu_index_);

    initializeContexts(device_ids);
    LOG(INFO) << "NICEndpoint initialized with " << contexts_.size()
              << " context(s) for GPU " << gpu_index_;

    return true;
  }

  void create_unified_p2p_socket() {}

  // Manual polling routine for recv channels when auto_start_polling_ is false
  void recvRoutine() {
    if (auto_start_polling_) {
      return;  // Do nothing if auto polling is enabled
    }
    std::shared_lock<std::shared_mutex> lock(recv_channel_mutex_);
    for (auto& [rank_id, recv_group] : recv_channel_groups_) {
      if (recv_group) {
        recv_group->pollAndProcessCompletions();
      }
    }
  }

  void sendRoutine() {
    if (auto_start_polling_) {
      return;  // Do nothing if auto polling is enabled
    }
    std::shared_lock<std::shared_mutex> lock(send_channel_mutex_);
    for (auto& [rank_id, send_group] : send_channel_groups_) {
      if (send_group) {
        send_group->pollingLoopForMeta();
      }
    }
  }

  // Manual polling routine for send channels when auto_start_polling_ is false
  int sendWithoutInnerQueue(std::shared_ptr<RDMASendRequest> req) {
    if (auto_start_polling_) {
      return -1;  // Do nothing if auto polling is enabled
    }
    if (!req) {
      LOG(WARNING) << "NICEndpoint::sendRoutine - null request";
      return -1;
    }

    uint64_t rank_id = req->to_rank_id;
    std::shared_lock<std::shared_mutex> lock(send_channel_mutex_);
    auto it = send_channel_groups_.find(rank_id);
    if (it == send_channel_groups_.end()) {
      LOG(WARNING) << "NICEndpoint::sendRoutine - Send channel group not found "
                      "for rank_id: "
                   << rank_id;
      return -1;
    }

    auto send_group = it->second;
    if (!send_group) {
      LOG(WARNING) << "NICEndpoint::sendRoutine - Send channel group is null "
                      "for rank_id: "
                   << rank_id;
      return -1;
    }

    return send_group->processSendRequests(req);
  }

 private:
  // Get context from channel_id
  inline std::shared_ptr<RdmaContext> getContextByChannelId(
      uint32_t channel_id) const {
    return contexts_[channelIdToContextId(channel_id)];
  }

  void initializeContexts(std::vector<size_t> const& device_ids) {
    auto& device_manager = RdmaDeviceManager::instance();
    assert(!device_ids.empty() && device_ids.size() <= kNICContextNumber);

    for (int i = 0; i < kNICContextNumber; ++i) {
      size_t device_id = device_ids[i % device_ids.size()];
      auto device = device_manager.getDevice(device_id);
      if (!device) {
        LOG(ERROR) << "Error: Device " << device_id << " not found";
        throw std::runtime_error("Device " + std::to_string(device_id) +
                                 " not found");
      }
      auto context = std::make_shared<RdmaContext>(device, contexts_.size());
      contexts_.push_back(context);
      LOG(INFO) << "NICEndpoint: Created context " << i << " for device "
                << device_id << " (" << device->name() << ")";
    }

    assert(contexts_.size() == kNICContextNumber);
  }

  void process_meta(std::string const& input, std::string& output,
                    std::string const& client_ip, int client_port) {
    MetaInfoToExchange meta = deserialize<MetaInfoToExchange>(input);
    LOG(INFO) << "Received from " << client_ip << ":" << client_port << " - "
              << meta;

    auto context_id = channelIdToContextId(meta.channel_id);
    std::shared_ptr<RdmaContext> ctx_ptr = contexts_[context_id];

    if (meta.flag == ChannelType::Control) {
      uint64_t actual_rank_id = meta.rank_id;
      if (rank_id_ == INVALID_RANK_ID) {
        actual_rank_id = recv_id_.fetch_add(1, std::memory_order_relaxed);
      }

      auto ctrl_mem =
          allocator_->allocate(kRingBufferSize, MemoryType::HOST, ctx_ptr);
      LOG(INFO) << "process_meta: Allocated " << ctrl_mem->size
                << " bytes for recv control channel ring buffer at "
                << ctrl_mem->addr;

      auto recv_ctrl_channel = std::make_shared<RecvControlChannel>(
          ctx_ptr, meta, ctrl_mem, meta.channel_id);

      // Create respons
      RemoteMemInfo ctrl_info(ctrl_mem);
      MetaInfoToExchange response(rank_id_, meta.channel_id,
                                  recv_ctrl_channel->get_local_meta(), nullptr,
                                  ChannelType::Control, gpu_index_);
      response.mem_meta = ctrl_info;
      LOG(INFO) << "response (control channel):::::::" << response;
      output = serialize(response);

      // Set the control channel
      auto ctrl_ch_copy = recv_ctrl_channel;
      setRecvControlChannel(actual_rank_id, std::move(ctrl_ch_copy));

      // Store accepted connection metadata
      {
        std::unique_lock<std::shared_mutex> lock(accepted_meta_mutex_);
        AcceptedMeta accepted;
        accepted.ip = client_ip;
        accepted.port = static_cast<uint16_t>(client_port);
        accepted.gpu_id = meta.gpu_id;
        accepted.rank_id = actual_rank_id;
        accepted_meta_[actual_rank_id] = accepted;
        LOG(INFO) << "Stored accepted connection: rank_id=" << actual_rank_id
                  << ", ip=" << client_ip << ", port=" << client_port
                  << ", gpu_id=" << meta.gpu_id;
      }
    } else {
      // Normal channel
      uint64_t actual_rank_id = meta.rank_id;
      if (rank_id_ == INVALID_RANK_ID) {
        // Find matching rank_id from accepted_meta_
        std::shared_lock<std::shared_mutex> lock(accepted_meta_mutex_);
        for (auto const& [rank_id, accepted] : accepted_meta_) {
          if (accepted.ip == client_ip &&
              accepted.port == static_cast<uint16_t>(client_port) &&
              accepted.gpu_id == meta.gpu_id) {
            actual_rank_id = rank_id;
            break;
          }
        }
      }

      std::shared_ptr<RDMAChannel> new_channel = std::make_shared<RDMAChannel>(
          ctx_ptr, meta.channel_meta, meta.channel_id);
      // Create response (echo back the same data)
      MetaInfoToExchange response(rank_id_, meta.channel_id,
                                  new_channel->get_local_meta(), nullptr,
                                  ChannelType::Normal, gpu_index_);
      LOG(INFO) << "response:::::::" << response;
      output = serialize(response);
      addOneRecvChannel(actual_rank_id, meta.channel_id, new_channel);
    }
  }

  // Handle response from send_meta operation
  uint64_t handle_send_meta_response(std::shared_ptr<RDMAChannel> channel,
                                     std::string const& response) {
    // Deserialize response as MetaInfoToExchange
    MetaInfoToExchange response_meta =
        deserialize<MetaInfoToExchange>(response);
    LOG(INFO) << response_meta;
    channel->connect(response_meta.channel_meta);
    return response_meta.rank_id;
  }

  std::shared_ptr<RecvChannelGroup> getOrCreateRecvGroup(uint64_t rank_id) {
    {
      std::shared_lock read_lock(recv_channel_mutex_);
      auto it = recv_channel_groups_.find(rank_id);
      if (it != recv_channel_groups_.end()) {
        return it->second;
      }
    }
    auto numa_node = RdmaDeviceManager::instance().get_numa_node(gpu_index_);
    {
      std::unique_lock write_lock(recv_channel_mutex_);
      auto [it, inserted] = recv_channel_groups_.try_emplace(
          rank_id,
          std::make_shared<RecvChannelGroup>(
              numa_node, auto_start_polling_));  // try_emplace constructs only
                                                 // if inserting
      return it->second;
    }
  }

  void addOneRecvChannel(uint64_t rank_id, uint32_t channel_id,
                         std::shared_ptr<RDMAChannel> new_channel) {
    std::shared_ptr<RecvChannelGroup> group_ptr = getOrCreateRecvGroup(rank_id);
    group_ptr->addChannel(channel_id, new_channel);
  }

  void setRecvControlChannel(
      uint64_t rank_id, std::shared_ptr<RecvControlChannel>&& ctrl_channel) {
    auto it = getOrCreateRecvGroup(rank_id);
    recv_channel_groups_[rank_id]->setControlChannel(
        std::forward<std::shared_ptr<RecvControlChannel>>(ctrl_channel));
  }

  std::shared_ptr<SendChannelGroup> getOrCreateSendGroup(uint64_t rank_id) {
    {
      std::shared_lock read_lock(send_channel_mutex_);
      auto it = send_channel_groups_.find(rank_id);
      if (it != send_channel_groups_.end()) return it->second;
    }
    auto numa_node = RdmaDeviceManager::instance().get_numa_node(gpu_index_);
    {
      std::unique_lock write_lock(send_channel_mutex_);
      auto [it, inserted] = send_channel_groups_.try_emplace(
          rank_id,
          std::make_shared<SendChannelGroup>(numa_node, auto_start_polling_));
      return it->second;
    }
  }

  void addOneSendChannel(uint64_t rank_id, uint32_t channel_id,
                         std::shared_ptr<RDMAChannel> new_channel) {
    auto group_ptr = getOrCreateSendGroup(rank_id);
    group_ptr->addChannel(channel_id, new_channel);
  }

  void setSendControlChannel(
      uint64_t rank_id, std::shared_ptr<SendControlChannel>&& ctrl_channel) {
    auto it = getOrCreateSendGroup(rank_id);
    send_channel_groups_[rank_id]->setControlChannel(
        std::forward<std::shared_ptr<SendControlChannel>>(ctrl_channel));
  }

  std::string const build_oob_connect(uint64_t rank_id) {
    auto const& item = rank_oob_meta_.find(rank_id);
    std::shared_ptr<OOBMetaData> ip_port_ptr = item->second;
    std::string oob_con;
    while (oob_con.empty()) {
      oob_con = oob_client_->connect_to_server(ip_port_ptr->server_ip,
                                               ip_port_ptr->server_port);
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    return oob_con;
  }

  int build_control_channel(std::string const& oob_con, uint64_t rank_id,
                            bool sync = true, int timeout_ms = 10000) {
    auto ctrl_mem =
        allocator_->allocate(kRingBufferSize, MemoryType::HOST, contexts_[0]);
    auto control_channel = std::make_shared<SendControlChannel>(
        contexts_[0], ctrl_mem, kControlChannelID);
    auto ctrl_info = std::make_shared<RemoteMemInfo>(ctrl_mem);

    MetaInfoToExchange ctrl_meta(rank_id_, kControlChannelID,
                                 control_channel->get_local_meta(), ctrl_info,
                                 ChannelType::Control, gpu_index_);

    LOG(INFO) << "Control Meta: " << ctrl_meta
              << " Local Channel Meta: " << control_channel->get_local_meta()
              << std::endl;

    std::string ctrl_serialized_meta = serialize(ctrl_meta);

    auto promise = std::make_shared<std::promise<uint64_t>>();
    std::future<uint64_t> future = promise->get_future();

    bool sent = oob_client_->send_meta(
        oob_con, ctrl_serialized_meta,
        [this, control_channel, promise,
         rank_id](std::string const& response) mutable {
          uint64_t peer_rank =
              this->handle_send_meta_response(control_channel, response);
          this->setSendControlChannel(rank_id, std::move(control_channel));
          promise->set_value(peer_rank);  // no try/catch → fail-fast
        });

    if (!sent) {
      LOG(ERROR) << "Failed to send control channel metadata for rank "
                 << rank_id;
      return -1;
    }

    if (!sync) {
      return rank_id;
    }

    if (future.wait_for(std::chrono::milliseconds(timeout_ms)) ==
        std::future_status::timeout) {
      LOG(ERROR) << "Timeout waiting for control channel handshake for rank "
                 << rank_id;
      return -1;
    }

    uint64_t recv_rank_id = future.get();

    LOG(INFO) << "Control channel handshake completed successfully for rank "
              << recv_rank_id;

    return static_cast<int>(recv_rank_id);
  }

  bool build_normal_channels(std::string const& oob_con, uint64_t rank_id,
                             bool sync = true, int timeout_ms = 10000) {
    std::vector<std::future<void>> futures;
    futures.reserve(kQpNumPerChannel);

    for (int i = 0; i < kQpNumPerChannel; i++) {
      uint32_t channel_id = i + 1;

      auto channel = std::make_shared<RDMAChannel>(
          getContextByChannelId(channel_id), channel_id);

      MetaInfoToExchange meta(rank_id_, channel_id, channel->get_local_meta(),
                              nullptr, ChannelType::Normal, gpu_index_);

      LOG(INFO) << meta << std::endl;
      std::string serialized_meta = serialize(meta);

      auto promise = std::make_shared<std::promise<void>>();
      futures.emplace_back(promise->get_future());

      bool sent = oob_client_->send_meta(
          oob_con, serialized_meta,
          [this, channel, channel_id, promise,
           rank_id](std::string const& response) {
            uint64_t peer_rank =
                this->handle_send_meta_response(channel, response);
            addOneSendChannel(rank_id, channel_id, channel);
            promise->set_value();  // no try/catch → fail-fast
          });

      if (!sent) {
        LOG(ERROR) << "Failed to send metadata for channel " << channel_id;
        return false;
      }
    }

    if (!sync) {
      LOG(INFO) << "Normal channels async build initiated for rank " << rank_id;
      return true;
    }

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);

    for (size_t i = 0; i < futures.size(); i++) {
      auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - std::chrono::steady_clock::now());

      if (remaining.count() <= 0 ||
          futures[i].wait_for(remaining) == std::future_status::timeout) {
        LOG(ERROR) << "Timeout waiting for channel " << (i + 1)
                   << " to complete";
        return false;
      }

      futures[i].get();
    }

    LOG(INFO) << "All " << kQpNumPerChannel
              << " normal channels built successfully for rank " << rank_id;

    return true;
  }

  uint64_t rank_id_;
  int gpu_index_;
  std::vector<std::shared_ptr<RdmaContext>> contexts_;
  mutable std::shared_mutex recv_channel_mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<RecvChannelGroup>>
      recv_channel_groups_;
  mutable std::shared_mutex send_channel_mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<SendChannelGroup>>
      send_channel_groups_;

  std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>> rank_oob_meta_;
  std::shared_ptr<EpollClient> oob_client_;
  std::shared_ptr<EpollServer> oob_server_;
  std::shared_ptr<MemoryAllocator> allocator_;
  mutable std::shared_mutex accepted_meta_mutex_;
  std::unordered_map<uint64_t, AcceptedMeta> accepted_meta_;
  bool auto_start_polling_;
  std::atomic<int32_t> send_id_;
  std::atomic<int32_t> recv_id_;
};
