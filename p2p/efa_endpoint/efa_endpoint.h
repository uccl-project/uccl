#pragma once
#include "define.h"
#include "efa_channel_group.h"
#include "efa_ctrl_channel.h"
#include "epoll_client.h"
#include "epoll_server.h"
#include "memory_allocator.h"
#include "rdma_context.h"
#include "rdma_device.h"
#include <glog/logging.h>

// ChannelGroup manages multiple channels for a connection

// EFAEndpoint manages RDMA contexts and channel groups
class EFAEndpoint {
 public:
  // Constructor with required gpu_index and optional device_ids
  // If device_ids is empty, all available devices will be used
  explicit EFAEndpoint(
      int gpu_index, uint64_t rank_id, uint64_t port = 0,
      std::vector<size_t> const& device_ids = std::vector<size_t>())
      : gpu_index_(gpu_index), rank_id_(rank_id), port_(port) {
    uccl::pin_thread_to_numa(0);
    std::vector<size_t> actual_device_ids;
    if (device_ids.size() == 0) {
      actual_device_ids =
          RdmaDeviceManager::instance().get_best_dev_idx(gpu_index);
    } else {
      actual_device_ids = device_ids;
    }
    initializeContexts(actual_device_ids);
    UCCL_LOG_EP << "EFAEndpoint initialized with " << contexts_.size()
                << " context(s) for GPU " << gpu_index_;

    oob_server_ = std::make_shared<EpollServer>(
        port_, [this](std::string const& input, std::string& output) {
          this->process_meta(input, output);
        });
    oob_client_ = std::make_shared<EpollClient>();

    allocator_ = std::make_shared<MemoryAllocator>();
    assert(oob_server_->start());
    assert(oob_client_->start());
  }

  // Destructor
  ~EFAEndpoint() {
    if (oob_client_) {
      oob_client_->stop();
    }
    if (oob_server_) {
      oob_server_->stop();
    }
  }

  // Getters
  int gpuIndex() const { return gpu_index_; }

  size_t contextCount() const { return contexts_.size(); }

  std::shared_ptr<RdmaContext> getContext(size_t index) const {
    if (index >= contexts_.size()) return nullptr;
    return contexts_[index];
  }

  std::vector<std::shared_ptr<RdmaContext>> const& contexts() const {
    return contexts_;
  }

  bool regMem(std::shared_ptr<RegMemBlock> reg_block) {
    if (unlikely(!reg_block)) {
      UCCL_LOG_ERROR << "Error: regMem called with null reg_block";
      return false;
    }

    // Register memory with each context
    try {
      for (size_t context_id = 0; context_id < contexts_.size(); ++context_id) {
        auto context = contexts_[context_id];
        if (unlikely(!context)) {
          UCCL_LOG_ERROR << "Error: context at context_id " << context_id
                         << " is null";
          return false;
        }

        // Register memory region for this context
        struct ibv_mr* mr =
            ibv_reg_mr(context->getPD(), reg_block->addr, reg_block->size,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ);

        if (unlikely(!mr)) {
          UCCL_LOG_ERROR << "Error: ibv_reg_mr failed for block at "
                         << reg_block->addr << " size " << reg_block->size
                         << " context_id " << context_id;
          return false;
        }

        // Store the MR in the RegMemBlock's mr_map using context_id as key
        reg_block->mr_map[context_id] = mr;

        UCCL_LOG_RE << "Registered memory block at " << reg_block->addr
                    << " size " << reg_block->size << " with context_id "
                    << context_id << " (lkey: 0x" << std::hex << mr->lkey
                    << ", rkey: 0x" << mr->rkey << std::dec << ")";
      }

      return true;
    } catch (std::exception const& e) {
      UCCL_LOG_ERROR << "Exception during memory registration: " << e.what();
      return false;
    }
  }
  // Channel group management
  // std::shared_ptr<ChannelGroup> getChannelGroup(uint64_t connect_id) {
  //     auto it = channel_groups_.find(connect_id);
  //     if (it == channel_groups_.end()) return nullptr;
  //     return it->second;
  // }

  // void addChannelGroup(uint64_t connect_id, std::shared_ptr<ChannelGroup>
  // group) {
  //     channel_groups_[connect_id] = group;
  // }

  // void createChannelGroup(uint64_t connect_id) {
  //     channel_groups_[connect_id] = std::make_shared<ChannelGroup>();
  // }

  // bool hasChannelGroup(uint64_t connect_id) const {
  //     return channel_groups_.find(connect_id) != channel_groups_.end();
  // }

  // void addOneRecvChannel(uint64_t rank_id, uint32_t channel_id,
  //                        std::shared_ptr<EFAChannel> new_channel) {
  //   std::shared_ptr<ChannelGroup> group_ptr;
  //   auto it = recv_channel_groups_.find(rank_id);
  //   if (it == recv_channel_groups_.end()) {
  //     recv_channel_groups_[rank_id] = std::make_shared<RecvChannelGroup>();
  //   } else {
  //     group_ptr = it->second;
  //   }
  //   group_ptr->addChannel(channel_id, new_channel);
  // }

  // void addOneSendChannel(uint64_t rank_id, uint32_t channel_id,
  //                        std::shared_ptr<EFAChannel> new_channel) {
  //   std::shared_ptr<ChannelGroup> group_ptr;
  //   auto it = send_channel_groups_.find(rank_id);
  //   if (it == send_channel_groups_.end()) {
  //     send_channel_groups_[rank_id] = std::make_shared<SendChannelGroup>();
  //   } else {
  //     group_ptr = it->second;
  //   }
  //   group_ptr->addChannel(channel_id, new_channel);
  // }
  std::shared_ptr<RecvChannelGroup> getOrCreateRecvGroup(uint64_t rank_id) {
    // Fast path: try shared (reader) lock first
    {
      std::shared_lock read_lock(recv_channel_mutex_);
      auto it = recv_channel_groups_.find(rank_id);
      if (it != recv_channel_groups_.end()) {
        return it->second;  // found -> return (shared_ptr keeps object alive)
      }
    }

    // Not found => acquire unique lock and insert (recheck to avoid race)
    {
      std::unique_lock write_lock(recv_channel_mutex_);
      auto [it, inserted] = recv_channel_groups_.try_emplace(
          rank_id,
          std::make_shared<RecvChannelGroup>());  // try_emplace constructs only
                                                  // if inserting
      // it->second now contains the group (either existing or newly created)
      return it->second;
    }
  }

  void addOneRecvChannel(uint64_t rank_id, uint32_t channel_id,
                         std::shared_ptr<EFAChannel> new_channel) {
    // 1) Get or create the group with minimal locking
    std::shared_ptr<RecvChannelGroup> group_ptr = getOrCreateRecvGroup(rank_id);
    group_ptr->addChannel(channel_id, new_channel);
  }

  std::shared_ptr<SendChannelGroup> getOrCreateSendGroup(uint64_t rank_id) {
    {
      std::shared_lock read_lock(send_channel_mutex_);
      auto it = send_channel_groups_.find(rank_id);
      if (it != send_channel_groups_.end()) return it->second;
    }
    {
      std::unique_lock write_lock(send_channel_mutex_);
      auto [it, inserted] = send_channel_groups_.try_emplace(
          rank_id, std::make_shared<SendChannelGroup>());
      return it->second;
    }
  }

  void addOneSendChannel(uint64_t rank_id, uint32_t channel_id,
                         std::shared_ptr<EFAChannel> new_channel) {
    auto group_ptr = getOrCreateSendGroup(rank_id);
    group_ptr->addChannel(channel_id, new_channel);
  }

  void setSendControlChannel(
      uint64_t rank_id, std::shared_ptr<SendControlChannel>&& ctrl_channel) {
    std::unique_lock write_lock(send_channel_mutex_);
    auto it = send_channel_groups_.find(rank_id);
    if (it == send_channel_groups_.end()) {
      send_channel_groups_[rank_id] = std::make_shared<SendChannelGroup>();
    }
    send_channel_groups_[rank_id]->setControlChannel(
        std::forward<std::shared_ptr<SendControlChannel>>(ctrl_channel));
  }

  void setRecvControlChannel(
      uint64_t rank_id, std::shared_ptr<RecvControlChannel>&& ctrl_channel) {
    std::unique_lock write_lock(recv_channel_mutex_);
    auto it = recv_channel_groups_.find(rank_id);
    if (it == recv_channel_groups_.end()) {
      recv_channel_groups_[rank_id] = std::make_shared<RecvChannelGroup>();
    }
    recv_channel_groups_[rank_id]->setControlChannel(
        std::forward<std::shared_ptr<RecvControlChannel>>(ctrl_channel));
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

  void build_control_channel(std::string const& oob_con, uint64_t rank_id) {
    // Allocate memory for control channel ring buffer
    auto ctrl_mem =
        allocator_->allocate(kRingBufferSize, MemoryType::HOST, contexts_[0]);
    UCCL_LOG_RE << "Allocated " << ctrl_mem->size
                << " bytes for control channel ring buffer at "
                << ctrl_mem->addr;

    auto control_channel =
        std::make_shared<SendControlChannel>(contexts_[0], ctrl_mem, 0);
    std::shared_ptr<RemoteMemInfo> ctrl_info =
        std::make_shared<RemoteMemInfo>(ctrl_mem);
    MetaInfoToExchange ctrl_meta(rank_id_, 0, control_channel->get_local_meta(),
                                 ctrl_info, ChannelType::Control);
    std::string ctrl_serialized_meta = serialize(ctrl_meta);
    bool ctrl_sent = oob_client_->send_meta(
        oob_con, ctrl_serialized_meta,
        [this, control_channel, rank_id](std::string const& response) {
          this->handle_send_meta_response(control_channel, response);
          // Create a copy of shared_ptr and move it to setSendControlChannel
          auto ctrl_ch_copy = control_channel;
          this->setSendControlChannel(rank_id, std::move(ctrl_ch_copy));
        });
    UCCL_LOG_EP << "Created control channel with QPN: "
                << control_channel->get_local_meta()->qpn;
  }

  bool build_normal_channels(std::string const& oob_con, uint64_t rank_id) {
    for (int i = 0; i < kQpNumPerChannel; i++) {
      uint32_t channel_id = i + 1;
      auto new_channel = std::make_shared<EFAChannel>(
          contexts_[i % contexts_.size()], channel_id);
      addOneSendChannel(rank_id, channel_id, new_channel);
      MetaInfoToExchange meta(rank_id_, channel_id,
                              new_channel->get_local_meta());
      std::string serialized_meta = serialize(meta);
      bool sent = oob_client_->send_meta(
          oob_con, serialized_meta,
          [this, new_channel](std::string const& response) {
            this->handle_send_meta_response(new_channel, response);
          });
    }
    return true;
  }
  bool build_connect(uint64_t rank_id) {
    if (rank_id == rank_id_) {
      return false;
    }
    std::string const oob_con = build_oob_connect(rank_id);
    this->build_control_channel(oob_con, rank_id);
    this->build_normal_channels(oob_con, rank_id);
    return true;
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

  // Blocking send: wraps SendChannelGroup::send with rank_id parameter
  // Returns wr_id for checking completion later
  int64_t send(uint64_t rank_id, std::shared_ptr<EFASendRequest> req) {
    auto it = send_channel_groups_.find(rank_id);
    if (it == send_channel_groups_.end()) {
      throw std::runtime_error("Send channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto send_group = it->second;
    int64_t wr_id = -1;

    // Blocking call until send succeeds
    while (wr_id < 0) {
      wr_id = send_group->send(req);
      LOG(INFO) << "EFAEndpoint::send - Attempting to send to rank_id: "
                << rank_id << ", channel_id: " << req->channel_id;
      if (wr_id < 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }

    return wr_id;
  }

  // Blocking recv: wraps RecvChannelGroup::recv with rank_id parameter
  // Returns index for checking completion later
  int64_t recv(uint64_t rank_id, std::shared_ptr<EFARecvRequest> req) {
    auto it = recv_channel_groups_.find(rank_id);
    if (it == recv_channel_groups_.end()) {
      throw std::runtime_error("Recv channel group not found for rank_id: " +
                               std::to_string(rank_id));
    }

    auto recv_group = it->second;
    int64_t index = -1;
    // Blocking call until recv succeeds
    while (index < 0) {
      LOG(INFO) << "EFAEndpoint::recv - Attempting to recv from rank_id: "
                << rank_id << ", channel_id: " << req->channel_id;
      index = recv_group->recv(req);
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

  // void TestSendReceive2() {
  //     std::string word = "World Hello! " + std::to_string(rank_id_);
  //     if (send_test_->local_mem->type == MemoryType::GPU) {
  //       char* h_data = (char*)malloc(1024 * 1024);
  //       strcpy(h_data, word.data());
  //       cudaMemcpy(send_test_->local_mem->addr, h_data, 1024 * 1024,
  //                  cudaMemcpyHostToDevice);
  //     } else {
  //       strcpy((char*)send_test_->local_mem->addr, word.data());
  //     }

  //     auto channel_recv = recv_channel_groups_[1 - rank_id_]->getChannel(0);
  //     auto channel_send = send_channel_groups_[1 - rank_id_]->getChannel(0);
  //     // auto wr_recv = channel_recv->recv(recv_test_);
  //     auto wr_send = channel_send->send(send_test_);
  //     std::cout<<"send down"<<std::endl;
  //     std::cout<<"wr_recv: "<<wr_send<<std::endl;
  //     // std::cout<<"wr_send: "<<wr_recv<<std::endl;

  //     std::cout<<"channel_send"<<std::endl;
  //     channel_send->printQP();
  //     std::cout<<"channel_recv"<<std::endl;
  //     channel_recv->printQP();
  //     // channel_send->poll_cq(wr_send);
  //     channel_recv->poll_cq(0);
  //     char* h_data = (char*)malloc(1024 * 1024);
  //     cudaMemcpy(h_data, recv_test_->local_mem->addr, 1024 * 1024,
  //                cudaMemcpyDeviceToHost);
  //     printf("Server received: %s\n", h_data);
  // }

 private:
  // std::shared_ptr<EFASendRequest> send_test_;
  // std::shared_ptr<EFARecvRequest> recv_test_;

  // Initialize RDMA contexts based on device_ids
  void initializeContexts(std::vector<size_t> const& device_ids) {
    auto& device_manager = RdmaDeviceManager::instance();

    // Determine which devices to use
    std::vector<size_t> target_device_ids;
    if (device_ids.empty()) {
      // Use all available devices
      for (size_t i = 0; i < device_manager.deviceCount(); ++i) {
        target_device_ids.push_back(i);
      }
    } else {
      target_device_ids = device_ids;
    }

    // Create context for each device
    for (size_t device_id : target_device_ids) {
      auto device = device_manager.getDevice(device_id);
      if (!device) {
        UCCL_LOG_ERROR << "Warning: Device " << device_id
                       << " not found, skipping";
        continue;
      }

      try {
        auto context = std::make_shared<RdmaContext>(device, contexts_.size());
        contexts_.push_back(context);
        UCCL_LOG_EP << "EFAEndpoint: Created context for device " << device_id
                    << " (" << device->name() << ")";
      } catch (std::exception const& e) {
        UCCL_LOG_ERROR << "Error creating context for device " << device_id
                       << ": " << e.what();
      }
    }

    if (contexts_.empty()) {
      throw std::runtime_error("EFAEndpoint: No contexts created");
    }
  }

  void process_meta(std::string const& input, std::string& output) {
    MetaInfoToExchange meta = deserialize<MetaInfoToExchange>(input);
    UCCL_LOG_EP << meta;

    auto context_id = meta.channel_id % contexts_.size();
    std::shared_ptr<RdmaContext> ctx_ptr = contexts_[context_id];

    if (meta.flag == ChannelType::Control) {
      // This is a control channel, allocate memory and create
      // RecvControlChannel
      auto ctrl_mem =
          allocator_->allocate(kRingBufferSize, MemoryType::HOST, ctx_ptr);
      UCCL_LOG_RE << "process_meta: Allocated " << ctrl_mem->size
                  << " bytes for recv control channel ring buffer at "
                  << ctrl_mem->addr;

      auto recv_ctrl_channel = std::make_shared<RecvControlChannel>(
          ctx_ptr, meta, ctrl_mem, meta.channel_id);

      // Create respons
      RemoteMemInfo ctrl_info(ctrl_mem);
      MetaInfoToExchange response(rank_id_, meta.channel_id,
                                  recv_ctrl_channel->get_local_meta());
      response.mem_meta = ctrl_info;
      UCCL_LOG_EP << "response (control channel):::::::" << response;
      output = serialize(response);

      // Set the control channel
      auto ctrl_ch_copy = recv_ctrl_channel;
      setRecvControlChannel(meta.rank_id, std::move(ctrl_ch_copy));
    } else {
      // Normal channel
      std::shared_ptr<EFAChannel> new_channel = std::make_shared<EFAChannel>(
          ctx_ptr, meta.channel_meta, meta.channel_id);
      // Create response (echo back the same data)
      MetaInfoToExchange response(rank_id_, meta.channel_id,
                                  new_channel->get_local_meta());
      UCCL_LOG_EP << "response:::::::" << response;
      output = serialize(response);
      addOneRecvChannel(meta.rank_id, meta.channel_id, new_channel);
    }
  }

  // Handle response from send_meta operation
  void handle_send_meta_response(std::shared_ptr<EFAChannel> channel,
                                 std::string const& response) {
    // Deserialize response as MetaInfoToExchange
    MetaInfoToExchange response_meta =
        deserialize<MetaInfoToExchange>(response);
    UCCL_LOG_EP << response_meta;
    channel->connect(response_meta.channel_meta);
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

  uint64_t port_;
  std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>> rank_oob_meta_;
  std::shared_ptr<EpollClient> oob_client_;
  std::shared_ptr<EpollServer> oob_server_;
  std::shared_ptr<MemoryAllocator> allocator_;
};