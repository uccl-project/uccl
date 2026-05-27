#include "rdma_endpoint.h"
#include "util/debug.h"
#include "util/gpu_rt.h"
#include "util/util.h"

RDMAEndpoint::RDMAEndpoint(int gpu_index, uint64_t port,
                           bool auto_start_polling,
                           std::vector<size_t> const& device_ids)
    : gpu_index_(gpu_index),
      auto_start_polling_(auto_start_polling),
      next_send_peer_id_(0),
      next_recv_peer_id_(0) {
  if (gpu_index != INVALID_GPU) {
    initialize_rdma_ctx_for_gpu(gpu_index, device_ids);
  }

  oob_server_ = std::make_shared<EpollServer>(
      port, [this](std::string const& input, std::string& output,
                   std::string const& ip,
                   int port) { this->process_meta(input, output, ip, port); });
  oob_client_ = std::make_shared<EpollClient>();

  allocator_ = std::make_shared<MemoryAllocator>();
  if (unlikely(!oob_server_->start())) {
    UCCL_LOG(ERROR) << "Failed to start OOB server";
    throw std::runtime_error("Failed to start OOB server");
  }
  if (unlikely(!oob_client_->start())) {
    UCCL_LOG(ERROR) << "Failed to start OOB client";
    throw std::runtime_error("Failed to start OOB client");
  }
  initCompressor();
}

RDMAEndpoint::~RDMAEndpoint() {
  if (oob_client_) {
    oob_client_->stop();
  }
  if (oob_server_) {
    oob_server_->stop();
  }
}

void RDMAEndpoint::initCompressor() {
  Compressor& compressor = Compressor::getInstance();
  auto compress_buf = compressor.getCompressBuffer();
  auto decompress_buf = compressor.getDecompressBuffer();
  if (compress_buf) regMrForAllSlots(*compress_buf);
  if (decompress_buf) regMrForAllSlots(*decompress_buf);
  if (compressor.getCompressStrategy() == CompressStrategy::kNone) return;
  // Host-side rings used to carry per-message compression metadata and
  // completion acks. Registered on every context slot so any data/control
  // QP can target them via rkey lookup.
  ack_ring_ = allocator_->allocate(kAckRingBytes, MemoryType::HOST, nullptr);
  write_meta_ring_ =
      allocator_->allocate(kWriteMetaRingBytes, MemoryType::HOST, nullptr);
  std::memset(ack_ring_->addr, 0, kAckRingBytes);
  std::memset(write_meta_ring_->addr, 0, kWriteMetaRingBytes);
  regMrForAllSlots(*ack_ring_);
  regMrForAllSlots(*write_meta_ring_);
}

void RDMAEndpoint::regMrForAllSlots(RegMemBlock& blk) {
  std::unordered_map<RdmaContext*, struct ibv_mr*> registered;
  for (size_t slot = 0; slot < contexts_.size(); ++slot) {
    auto ctx = contexts_[slot];
    auto it = registered.find(ctx.get());
    struct ibv_mr* mr = nullptr;
    if (it != registered.end()) {
      mr = it->second;
    } else {
      mr = ctx->regMem(blk.addr, blk.size);
      if (!mr) {
        UCCL_LOG(ERROR) << "regMrForAllSlots: ibv_reg_mr FAILED for addr="
                        << blk.addr << " size=" << blk.size << " slot=" << slot
                        << " errno=" << errno << " (" << strerror(errno) << ")";
      } else {
        UCCL_LOG(INFO, UCCL_RDMA)
            << "regMrForAllSlots: registered addr=" << blk.addr
            << " size=" << blk.size << " slot=" << slot << " lkey=0x"
            << std::hex << mr->lkey << " rkey=0x" << mr->rkey << std::dec;
      }
      registered[ctx.get()] = mr;
    }
    blk.setMRByContextID(slot, mr);
  }
}

std::shared_ptr<RegMemBlock> RDMAEndpoint::ackRing() const { return ack_ring_; }

std::shared_ptr<RegMemBlock> RDMAEndpoint::writeMetaRing() const {
  return write_meta_ring_;
}

void RDMAEndpoint::fillCompressionMeta(MetaInfoToExchange& m) const {
  auto decomp = Compressor::getInstance().getDecompressBuffer();
  if (decomp) m.decompress_buf_meta = RemoteMemInfo(*decomp);
  if (write_meta_ring_)
    m.write_meta_ring_meta = RemoteMemInfo(*write_meta_ring_);
  if (ack_ring_) m.ack_ring_meta = RemoteMemInfo(*ack_ring_);
}

int RDMAEndpoint::gpuIndex() const { return gpu_index_; }

size_t RDMAEndpoint::contextCount() const { return contexts_.size(); }

bool RDMAEndpoint::regMem(std::shared_ptr<RegMemBlock> reg_block) {
  if (unlikely(!reg_block)) {
    UCCL_LOG(ERROR) << "Error: regMem called with null reg_block";
    return false;
  }

  // Register once per unique RdmaContext (contexts sharing the same NIC
  // device are the same shared_ptr - see initializeContexts).  This avoids
  // redundant MR registrations that waste resources: with DMA-BUF, each
  // duplicate consumes a GPU DMA mapping VA slot; with nvidia_peermem,
  // each duplicate pins the same pages again under a separate PD.
  std::unordered_map<RdmaContext*, struct ibv_mr*> registered;
  for (size_t context_id = 0; context_id < contexts_.size(); ++context_id) {
    auto context = contexts_[context_id];
    if (unlikely(!context)) {
      UCCL_LOG(ERROR) << "Error: context at context_id " << context_id
                      << " is null";
      return false;
    }

    auto it = registered.find(context.get());
    if (it != registered.end()) {
      // Same NIC device - reuse the already-registered MR.
      reg_block->setMRByContextID(context_id, it->second);
      continue;
    }

    struct ibv_mr* mr = context->regMem(reg_block->addr, reg_block->size);

    if (unlikely(!mr)) {
      UCCL_LOG(ERROR) << "Error: ibv_reg_mr failed for block at "
                      << reg_block->addr << " size " << reg_block->size
                      << " context_id " << context_id;
      return false;
    }
    reg_block->setMRByContextID(context_id, mr);
    registered[context.get()] = mr;
  }

  return true;
}

bool RDMAEndpoint::deregMem(std::shared_ptr<RegMemBlock> reg_block) {
  if (unlikely(!reg_block)) {
    UCCL_LOG(ERROR) << "Error: deregMem called with null reg_block";
    return false;
  }
  // Deduplicate MR pointers - shared contexts have the same MR in
  // multiple slots, so we must not double-free.
  std::unordered_set<ibv_mr*> freed;
  for (uint32_t ctx = 0; ctx < kNICContextNumber; ++ctx) {
    ibv_mr* mr = reg_block->getMRByContextID(ctx);
    if (mr && freed.insert(mr).second) {
      RdmaContext::deregMem(mr);
    }
  }
  return true;
}

int RDMAEndpoint::build_connect(uint64_t peer_id, bool sync, int timeout_ms) {
  std::string const oob_con = build_oob_connect(peer_id);
  uint64_t receved_peer_id =
      this->build_control_channel(oob_con, peer_id, sync, timeout_ms);
  if (receved_peer_id < 0) {
    return -1;
  }
  if (!this->build_data_channels(oob_con, peer_id, sync, timeout_ms)) {
    return -1;
  }
  return static_cast<int>(receved_peer_id);
}

void RDMAEndpoint::checkSendComplete(uint64_t peer_id, int64_t wr_id) {
  UCCL_LOG(INFO, UCCL_RDMA)
      << "checkSendComplete - peer_id: " << peer_id << ", wr_id: " << wr_id;

  auto it = send_channel_groups_.find(peer_id);
  if (it == send_channel_groups_.end()) {
    throw std::runtime_error("Send channel group not found for peer_id: " +
                             std::to_string(peer_id));
  }

  auto send_group = it->second;
  while (!send_group->check(wr_id)) {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }
  UCCL_LOG(INFO, UCCL_RDMA)
      << "checkSendComplete - Completed for peer_id: " << peer_id
      << ", wr_id: " << wr_id;
}

bool RDMAEndpoint::checkSendComplete_once(uint64_t peer_id, int64_t wr_id) {
  // UCCL_LOG(INFO, UCCL_RDMA) << "checkSendComplete - peer_id: " << peer_id
  //           << ", wr_id: " << wr_id;
  std::shared_lock<std::shared_mutex> lock(send_channel_mutex_);
  auto it = send_channel_groups_.find(peer_id);
  if (unlikely(it == send_channel_groups_.end())) {
    throw std::runtime_error("Send channel group not found for peer_id: " +
                             std::to_string(peer_id));
  }

  auto send_group = it->second;
  return send_group->check(wr_id);
}

SendConnection* RDMAEndpoint::getSendGroupRaw(uint64_t peer_id) {
  std::shared_lock<std::shared_mutex> lock(send_channel_mutex_);
  auto it = send_channel_groups_.find(peer_id);
  if (it == send_channel_groups_.end()) return nullptr;
  return it->second.get();
}

bool RDMAEndpoint::checkRecvComplete_once(uint64_t peer_id, uint64_t index) {
  UCCL_LOG(INFO, UCCL_RDMA)
      << "checkRecvComplete - Checking for peer_id: " << peer_id
      << ", index: " << index;
  auto it = recv_channel_groups_.find(peer_id);
  if (unlikely(it == recv_channel_groups_.end())) {
    throw std::runtime_error("Recv channel group not found for peer_id: " +
                             std::to_string(peer_id));
  }

  auto recv_group = it->second;
  return recv_group->check(index);
}

void RDMAEndpoint::checkRecvComplete(uint64_t peer_id, uint64_t index) {
  UCCL_LOG(INFO, UCCL_RDMA)
      << "checkRecvComplete - Checking for peer_id: " << peer_id
      << ", index: " << index;
  auto it = recv_channel_groups_.find(peer_id);
  if (it == recv_channel_groups_.end()) {
    throw std::runtime_error("Recv channel group not found for peer_id: " +
                             std::to_string(peer_id));
  }

  auto recv_group = it->second;
  while (!recv_group->check(index)) {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }
  UCCL_LOG(INFO, UCCL_RDMA)
      << "checkRecvComplete - Completed for peer_id: " << peer_id
      << ", index: " << index;
}

int64_t RDMAEndpoint::writeOrRead(std::shared_ptr<RDMASendRequest> req) {
  uint64_t peer_id = req->to_peer_id;
  auto it = send_channel_groups_.find(peer_id);
  if (it == send_channel_groups_.end()) {
    throw std::runtime_error("Send channel group not found for peer_id: " +
                             std::to_string(peer_id));
  }

  auto send_group = it->second;
  int64_t wr_id = -1;

  // Blocking call until send succeeds
  while (wr_id < 0) {
    // UCCL_LOG(INFO, UCCL_RDMA) << "RDMAEndpoint::write - Attempting to send
    // to peer_id:
    // "
    //           << peer_id;
    wr_id = send_group->postWriteOrRead(req);

    if (wr_id < 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  return wr_id;
}

int64_t RDMAEndpoint::send(uint64_t peer_id,
                           std::shared_ptr<RDMASendRequest> req) {
  auto it = send_channel_groups_.find(peer_id);
  if (it == send_channel_groups_.end()) {
    throw std::runtime_error("Send channel group not found for peer_id: " +
                             std::to_string(peer_id));
  }

  auto send_group = it->second;
  int64_t wr_id = -1;

  // Blocking call until send succeeds
  while (wr_id < 0) {
    UCCL_LOG(INFO, UCCL_RDMA)
        << "RDMAEndpoint::send - Attempting to send to peer_id: " << peer_id;
    wr_id = send_group->send(req);

    if (wr_id < 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  return wr_id;
}

int64_t RDMAEndpoint::recv(uint64_t peer_id,
                           std::shared_ptr<RDMARecvRequest> req) {
  auto it = recv_channel_groups_.find(peer_id);
  if (it == recv_channel_groups_.end()) {
    throw std::runtime_error("Recv channel group not found for peer_id: " +
                             std::to_string(peer_id));
  }

  auto recv_group = it->second;
  int64_t index = -1;
  // Blocking call until recv succeeds
  while (index < 0) {
    index = recv_group->recv(req);
    UCCL_LOG(INFO, UCCL_RDMA)
        << "RDMAEndpoint::recv - Attempting to recv from peer_id: " << peer_id;
    if (index < 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  return index;
}

void RDMAEndpoint::add_peer_oob_meta(
    std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>> const&
        new_meta) {
  for (auto const& [peer_id, meta_ptr] : new_meta) {
    peer_oob_meta_[peer_id] = meta_ptr;
  }
}

ConnID RDMAEndpoint::uccl_connect(int remote_gpuidx, std::string remote_ip,
                                  uint16_t remote_port) {
  int32_t current_send_peer_id =
      next_send_peer_id_.fetch_add(1, std::memory_order_relaxed);

  assert(gpu_index_ != INVALID_GPU);

  add_peer_oob_meta({{current_send_peer_id,
                      std::make_shared<OOBMetaData>(remote_ip, remote_port)}});
  UCCL_LOG(INFO, UCCL_RDMA)
      << "remote_gpuidx: " << remote_gpuidx << ", remote_ip: " << remote_ip
      << ", remote_port: " << remote_port;
  build_connect(current_send_peer_id);  // sync mode (default)
  ConnID conn_id;
  conn_id.context =
      reinterpret_cast<void*>(static_cast<intptr_t>(current_send_peer_id));
  conn_id.peer_id = current_send_peer_id;
  return conn_id;
}

uint16_t RDMAEndpoint::get_p2p_listen_port() { return oob_server_->get_port(); }

int RDMAEndpoint::get_p2p_listen_fd() { return oob_server_->get_listen_fd(); }

std::shared_ptr<EpollClient> RDMAEndpoint::get_oob_client() {
  return oob_client_;
}

std::string RDMAEndpoint::get_oob_conn_key(uint64_t peer_id) {
  std::shared_lock<std::shared_mutex> lock(peer_oob_conn_keys_mutex_);
  auto it = peer_oob_conn_keys_.find(peer_id);
  return (it != peer_oob_conn_keys_.end()) ? it->second : "";
}

ConnID RDMAEndpoint::uccl_accept(std::string& remote_ip, int* remote_gpuidx) {
  AcceptedMeta accepted;
  uint64_t peer_id = 0;

  // Block until there's an accepted connection
  while (!stop_accept_.load(std::memory_order_acquire)) {
    {
      {
        std::unique_lock<std::shared_mutex> lock(accepted_meta_mutex_);
        if (!accepted_meta_.empty()) {
          // Get the first accepted connection
          auto it = accepted_meta_.begin();
          peer_id = it->first;
          accepted = it->second;
          // Remove it from the map
          if (getOrCreateRecvGroup(peer_id)->channelCount() ==
              kQpNumPerChannel + 1) {
            accepted_meta_.erase(it);
            UCCL_LOG(INFO, UCCL_RDMA)
                << "Accepted connection: peer_id=" << peer_id
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
  UCCL_LOG(INFO, UCCL_RDMA)
      << "Done Accepted connection: peer_id=" << peer_id
      << ", ip=" << accepted.ip << ", port=" << accepted.port
      << ", gpu_id=" << accepted.gpu_id;
  // Assign output parameters
  remote_ip = accepted.ip;
  if (remote_gpuidx != nullptr) {
    *remote_gpuidx = accepted.gpu_id;
  }

  // Create and return ConnID
  ConnID conn_id;
  conn_id.context = reinterpret_cast<void*>(static_cast<intptr_t>(peer_id));
  conn_id.peer_id = peer_id;
  conn_id.sock_fd = 0;
  return conn_id;
}

int RDMAEndpoint::uccl_regmr(void* const data, size_t const len,
                             MRArray& mr_array,
                             std::vector<MrCacheHandleRef>& cache_refs,
                             CompressCtx compress_ctx) {
  if (unlikely(!data)) {
    UCCL_LOG(ERROR) << "Error: uccl_regmr called with null data";
    return -1;
  }
  Compressor::getInstance().prepareSplitContext(data, len, compress_ctx);
  // Register once per unique RdmaContext to avoid redundant MR
  // registrations: with DMA-BUF each duplicate consumes a GPU DMA mapping
  // VA slot; with nvidia_peermem each duplicate re-pins pages under a
  // separate PD.  On single-NIC systems all 4 context slots share one
  // RdmaContext, reducing registrations from 4 to 1.
  cache_refs.clear();
  std::unordered_map<RdmaContext*, struct ibv_mr*> registered;
  for (size_t context_id = 0; context_id < contexts_.size(); ++context_id) {
    auto context = contexts_[context_id];
    if (unlikely(!context)) {
      UCCL_LOG(ERROR) << "Error: context at context_id " << context_id
                      << " is null";
      for (auto const& ref : cache_refs) {
        ref.context->releaseCachedMr(ref.entry);
      }
      cache_refs.clear();
      return -1;
    }

    auto it = registered.find(context.get());
    if (it != registered.end()) {
      // Same NIC device - reuse the already-registered MR.
      mr_array.setKeyByContextID(context_id, it->second);
      continue;
    }

    MrCacheEntry* entry = context->acquireCachedMr(data, len);
    struct ibv_mr* mr = entry ? entry->mr : nullptr;

    if (unlikely(!mr)) {
      UCCL_LOG(ERROR) << "Error " << errno << " " << strerror(errno)
                      << ": ibv_reg_mr_iova2 failed for data at " << data
                      << " size " << len << " context_id " << context_id;
      for (auto const& ref : cache_refs) {
        ref.context->releaseCachedMr(ref.entry);
      }
      cache_refs.clear();
      return -1;
    }

    // Store the MR in the mr_map using context_id as key
    mr_array.setKeyByContextID(context_id, mr);
    registered[context.get()] = mr;
    cache_refs.push_back({context, entry});
  }

  return 0;
}

void RDMAEndpoint::uccl_deregmr(
    std::vector<MrCacheHandleRef> const& cache_refs) {
  for (auto const& ref : cache_refs) {
    if (likely(ref.context != nullptr)) {
      ref.context->releaseCachedMr(ref.entry);
    }
  }
}

bool RDMAEndpoint::initialize_rdma_ctx_for_gpu(
    int gpu_index, std::vector<size_t> const& device_ids) {
  gpu_index_ = gpu_index;

  // Find all devices used by the GPU
  std::vector<size_t> actual_device_ids;
  if (device_ids.size() == 0) {
    actual_device_ids =
        RdmaDeviceManager::instance().get_best_dev_idx(gpu_index);
  } else {
    actual_device_ids = device_ids;
  }

  initializeContexts(actual_device_ids);
  UCCL_LOG(INFO, UCCL_RDMA)
      << "RDMAEndpoint initialized with " << contexts_.size()
      << " context(s) for GPU " << gpu_index;

  for (auto dev : actual_device_ids) {
    auto device = RdmaDeviceManager::instance().getDevice(dev);
    std::cout << "GPU " << gpu_index << " uses device " << dev << " ("
              << device->name() << ")" << std::endl;
  }
  return true;
}

void RDMAEndpoint::create_unified_p2p_socket() {}

void RDMAEndpoint::recvRoutine() {
  if (auto_start_polling_) {
    return;  // Do nothing if auto polling is enabled
  }
  std::shared_lock<std::shared_mutex> lock(recv_channel_mutex_);
  for (auto& [peer_id, recv_group] : recv_channel_groups_) {
    if (recv_group && !recv_group->isRunning()) {
      recv_group->pollAndProcessCompletions();
    }
  }
}

void RDMAEndpoint::sendRoutine() {
  if (auto_start_polling_) {
    return;  // Do nothing if auto polling is enabled
  }
  std::shared_lock<std::shared_mutex> lock(send_channel_mutex_);
  for (auto& [peer_id, send_group] : send_channel_groups_) {
    if (send_group && !send_group->isRunning()) {
      send_group->pollingLoopForMeta();
    }
  }
}

void RDMAEndpoint::flushAllSends() {
  std::shared_lock<std::shared_mutex> lock(send_channel_mutex_);
  for (auto& [peer_id, send_group] : send_channel_groups_) {
    if (send_group) send_group->flushBatches();
  }
}

int RDMAEndpoint::sendWithoutInnerQueue(std::shared_ptr<RDMASendRequest> req) {
  if (!req) {
    UCCL_LOG(WARN) << "RDMAEndpoint::sendRoutine - null request";
    return -1;
  }

  uint64_t peer_id = req->to_peer_id;
  std::shared_lock<std::shared_mutex> lock(send_channel_mutex_);
  auto it = send_channel_groups_.find(peer_id);
  if (it == send_channel_groups_.end()) {
    UCCL_LOG(WARN)
        << "RDMAEndpoint::sendRoutine - Send channel group not found "
           "for peer_id: "
        << peer_id;
    return -1;
  }

  auto send_group = it->second;
  if (!send_group) {
    UCCL_LOG(WARN) << "RDMAEndpoint::sendRoutine - Send channel group is null "
                      "for peer_id: "
                   << peer_id;
    return -1;
  }

  // When the polling thread is active, enqueue via send() so that
  // only the polling thread touches QPs (avoids concurrent ibv_post_*
  // from two threads).
  if (send_group->isRunning()) {
    return send_group->send(req);
  }

  return send_group->processSendRequests(req);
}

void RDMAEndpoint::stop_accept() {
  stop_accept_.store(true, std::memory_order_release);
}

std::shared_ptr<RdmaContext> RDMAEndpoint::getContextByChannelId(
    uint32_t channel_id) const {
  return contexts_[channelIdToContextId(channel_id)];
}

void RDMAEndpoint::initializeContexts(std::vector<size_t> const& device_ids) {
  auto& device_manager = RdmaDeviceManager::instance();
  assert(!device_ids.empty() && device_ids.size() <= kNICContextNumber);

  // Share RdmaContext objects across context slots that map to the same
  // physical NIC device.  Each unique ibv_context + ibv_pd triggers a
  // separate memory registration (ibv_reg_mr or ibv_reg_dmabuf_mr).
  // With DMA-BUF, each registration creates a GPU DMA mapping VA entry;
  // with nvidia_peermem, each registration pins GPU pages under a separate
  // PD.  Sharing avoids exhausting these resources when multiple
  // subsystems (DeepEP, NCCL, NIXL) co-exist on the same GPU+NIC.
  std::unordered_map<size_t, std::shared_ptr<RdmaContext>> device_ctx_map;
  int unique_count = 0;

  for (int i = 0; i < kNICContextNumber; ++i) {
    size_t device_id = device_ids[i % device_ids.size()];

    auto it = device_ctx_map.find(device_id);
    if (it != device_ctx_map.end()) {
      // Same device as an earlier context - share it.
      contexts_.push_back(it->second);
      UCCL_LOG(INFO, UCCL_RDMA)
          << "RDMAEndpoint: Context " << i << " shares device " << device_id
          << " (" << device_manager.getDevice(device_id)->name() << ")";
      continue;
    }

    auto device = device_manager.getDevice(device_id);
    if (!device) {
      UCCL_LOG(ERROR) << "Error: Device " << device_id << " not found";
      throw std::runtime_error("Device " + std::to_string(device_id) +
                               " not found");
    }
    auto context = std::make_shared<RdmaContext>(device, contexts_.size());
    contexts_.push_back(context);
    device_ctx_map[device_id] = context;
    unique_count++;
    UCCL_LOG(INFO, UCCL_RDMA)
        << "RDMAEndpoint: Created context " << i << " for device " << device_id
        << " (" << device->name() << ")";
  }

  assert(contexts_.size() == kNICContextNumber);
  UCCL_LOG(INFO, UCCL_RDMA)
      << "RDMAEndpoint: " << kNICContextNumber << " context slots, "
      << unique_count << " unique device(s)";
}

void RDMAEndpoint::process_meta(std::string const& input, std::string& output,
                                std::string const& client_ip, int client_port) {
  if (input.size() >= sizeof(NotifyMsg)) {
    NotifyMsg const* notify_msg =
        reinterpret_cast<NotifyMsg const*>(input.data());
    if (notify_msg->magic == NOTIFY_MSG_MAGIC) {
      std::lock_guard<std::mutex> lock(notify_mutex);
      notify_list.push_back(*notify_msg);
      output = "";
      UCCL_LOG(INFO, UCCL_RDMA)
          << "process_meta: Received notification from" << notify_msg->name
          << " msg=" << notify_msg->msg;
      return;
    }
  }

  MetaInfoToExchange meta = deserialize<MetaInfoToExchange>(input);
  UCCL_LOG(INFO, UCCL_RDMA)
      << "Received from " << client_ip << ":" << client_port << " - " << meta;

  auto context_id = channelIdToContextId(meta.channel_id);
  std::shared_ptr<RdmaContext> ctx_ptr = contexts_[context_id];

  if (meta.flag == ChannelType::Control) {
    uint64_t actual_peer_id =
        next_recv_peer_id_.fetch_add(1, std::memory_order_relaxed);

    auto ctrl_mem =
        allocator_->allocate(kRingBufferSize, MemoryType::HOST, ctx_ptr);
    UCCL_LOG(INFO, UCCL_RDMA)
        << "process_meta: Allocated " << ctrl_mem->size
        << " bytes for recv control channel ring buffer at " << ctrl_mem->addr;

    auto recv_ctrl_channel = std::make_shared<RecvControlChannel>(
        ctx_ptr, meta, ctrl_mem, meta.channel_id);
    // Receiver owns the authoritative WriteReqMeta ring.
    if (write_meta_ring_)
      recv_ctrl_channel->bindWriteMetaRing(write_meta_ring_);

    // Create response (include our OOB port for potential future use)
    RemoteMemInfo ctrl_info(ctrl_mem);
    MetaInfoToExchange response(
        INVALID_PEER_ID, meta.channel_id, recv_ctrl_channel->get_local_meta(),
        nullptr, ChannelType::Control, gpu_index_, oob_server_->get_port());
    response.mem_meta = ctrl_info;
    fillCompressionMeta(response);
    UCCL_LOG(INFO, UCCL_RDMA)
        << "response (control channel):::::::" << response;
    output = serialize(response);

    // Set the control channel
    auto ctrl_ch_copy = recv_ctrl_channel;
    setRecvControlChannel(actual_peer_id, std::move(ctrl_ch_copy));
    // Remember peer's ack_ring address+rkey - the receive side needs it
    // when decompress finishes and we post an ack back to the sender.
    setRecvCompressionPeerMeta(actual_peer_id, meta);

    // Store accepted connection metadata
    {
      std::unique_lock<std::shared_mutex> lock(accepted_meta_mutex_);
      AcceptedMeta accepted;
      accepted.ip = client_ip;
      accepted.port = static_cast<uint16_t>(client_port);
      accepted.gpu_id = meta.gpu_id;
      accepted.peer_id = actual_peer_id;
      accepted_meta_[actual_peer_id] = accepted;
      UCCL_LOG(INFO, UCCL_RDMA)
          << "Stored accepted connection: peer_id=" << actual_peer_id
          << ", ip=" << client_ip << ", port=" << client_port
          << ", gpu_id=" << meta.gpu_id;
    }

    if (meta.oob_port > 0) {
      std::string rev_conn_key =
          oob_client_->connect_to_server(client_ip, meta.oob_port);
      if (!rev_conn_key.empty()) {
        std::unique_lock<std::shared_mutex> lock(peer_oob_conn_keys_mutex_);
        peer_oob_conn_keys_[actual_peer_id] = rev_conn_key;
        UCCL_LOG(INFO, UCCL_RDMA)
            << "Established reverse connection to " << client_ip << ":"
            << meta.oob_port << " for peer_id=" << actual_peer_id
            << ", conn_key=" << rev_conn_key;
      } else {
        UCCL_LOG(WARN) << "Failed to establish reverse connection to "
                       << client_ip << ":" << meta.oob_port;
      }
    }
  } else {
    // Data channel
    uint64_t actual_peer_id = INVALID_PEER_ID;
    // Find matching peer_id from accepted_meta_.
    std::shared_lock<std::shared_mutex> lock(accepted_meta_mutex_);
    for (auto const& [peer_id, accepted] : accepted_meta_) {
      if (accepted.ip == client_ip &&
          accepted.port == static_cast<uint16_t>(client_port) &&
          accepted.gpu_id == meta.gpu_id) {
        actual_peer_id = peer_id;
        break;
      }
    }

    std::shared_ptr<RDMADataChannel> new_channel =
        std::make_shared<RDMADataChannel>(ctx_ptr, meta.channel_meta,
                                          meta.channel_id);
    // Create response (echo back the same data)
    MetaInfoToExchange response(INVALID_PEER_ID, meta.channel_id,
                                new_channel->get_local_meta(), nullptr,
                                ChannelType::Normal, gpu_index_);
    UCCL_LOG(INFO, UCCL_RDMA) << "response:::::::" << response;
    output = serialize(response);
    addOneRecvChannel(actual_peer_id, meta.channel_id, new_channel);
  }
}

uint64_t RDMAEndpoint::handle_send_meta_response(
    std::shared_ptr<RDMADataChannel> channel, std::string const& response) {
  // Deserialize response as MetaInfoToExchange
  MetaInfoToExchange response_meta = deserialize<MetaInfoToExchange>(response);
  UCCL_LOG(INFO, UCCL_RDMA) << response_meta;
  channel->establishChannel(response_meta.channel_meta);
  return response_meta.peer_id;
}

std::shared_ptr<RecvConnection> RDMAEndpoint::getOrCreateRecvGroup(
    uint64_t peer_id) {
  {
    std::shared_lock read_lock(recv_channel_mutex_);
    auto it = recv_channel_groups_.find(peer_id);
    if (it != recv_channel_groups_.end()) {
      return it->second;
    }
  }
  auto numa_node = RdmaDeviceManager::instance().get_numa_node(
      RdmaDeviceManager::instance().get_best_dev_idx(gpu_index_)[0]);
  {
    std::unique_lock write_lock(recv_channel_mutex_);
    auto [it, inserted] = recv_channel_groups_.try_emplace(
        peer_id,
        std::make_shared<RecvConnection>(
            numa_node, auto_start_polling_));  // try_emplace constructs only
                                               // if inserting
    return it->second;
  }
}

void RDMAEndpoint::addOneRecvChannel(
    uint64_t peer_id, uint32_t channel_id,
    std::shared_ptr<RDMADataChannel> new_channel) {
  std::shared_ptr<RecvConnection> group_ptr = getOrCreateRecvGroup(peer_id);
  group_ptr->addChannel(channel_id, new_channel);
}

void RDMAEndpoint::setRecvControlChannel(
    uint64_t peer_id, std::shared_ptr<RecvControlChannel>&& ctrl_channel) {
  auto it = getOrCreateRecvGroup(peer_id);
  recv_channel_groups_[peer_id]->setControlChannel(
      std::forward<std::shared_ptr<RecvControlChannel>>(ctrl_channel));
}

std::shared_ptr<SendConnection> RDMAEndpoint::getOrCreateSendGroup(
    uint64_t peer_id) {
  {
    std::shared_lock read_lock(send_channel_mutex_);
    auto it = send_channel_groups_.find(peer_id);
    if (it != send_channel_groups_.end()) return it->second;
  }
  auto numa_node = RdmaDeviceManager::instance().get_numa_node(
      RdmaDeviceManager::instance().get_best_dev_idx(gpu_index_)[0]);
  auto* ctx =
      (!contexts_.empty() && contexts_[0]) ? contexts_[0]->getCtx() : nullptr;
  double link_bw =
      uccl::cc::get_link_bandwidth_bps(ctx, "UCCL_P2P_RDMA_LINK_GBPS");
  {
    std::unique_lock write_lock(send_channel_mutex_);
    auto [it, inserted] = send_channel_groups_.try_emplace(
        peer_id, std::make_shared<SendConnection>(
                     numa_node, auto_start_polling_, link_bw));
    return it->second;
  }
}

void RDMAEndpoint::addOneSendChannel(
    uint64_t peer_id, uint32_t channel_id,
    std::shared_ptr<RDMADataChannel> new_channel) {
  auto group_ptr = getOrCreateSendGroup(peer_id);
  group_ptr->addChannel(channel_id, new_channel);
}

void RDMAEndpoint::setSendControlChannel(
    uint64_t peer_id, std::shared_ptr<SendControlChannel>&& ctrl_channel) {
  auto it = getOrCreateSendGroup(peer_id);
  send_channel_groups_[peer_id]->setControlChannel(
      std::forward<std::shared_ptr<SendControlChannel>>(ctrl_channel));
}

void RDMAEndpoint::setSendCompressionPeerMeta(uint64_t peer_id,
                                              MetaInfoToExchange const& peer) {
  if (!ack_ring_ || peer.decompress_buf_meta.length == 0) return;
  auto group = getOrCreateSendGroup(peer_id);
  group->setRemoteDecompressBuf(peer.decompress_buf_meta);
  group->setLocalAckRing(ack_ring_);
}

void RDMAEndpoint::setRecvCompressionPeerMeta(uint64_t peer_id,
                                              MetaInfoToExchange const& peer) {
  if (peer.ack_ring_meta.length == 0) return;
  auto group = getOrCreateRecvGroup(peer_id);
  group->setRemoteAckRing(peer.ack_ring_meta);
}

std::string const RDMAEndpoint::build_oob_connect(uint64_t peer_id) {
  auto const& item = peer_oob_meta_.find(peer_id);
  std::shared_ptr<OOBMetaData> ip_port_ptr = item->second;
  std::string oob_con;
  while (oob_con.empty()) {
    oob_con = oob_client_->connect_to_server(ip_port_ptr->server_ip,
                                             ip_port_ptr->server_port);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  // Store conn_key for later use (e.g., notifications)
  std::unique_lock<std::shared_mutex> lock(peer_oob_conn_keys_mutex_);
  peer_oob_conn_keys_[peer_id] = oob_con;
  return oob_con;
}

int RDMAEndpoint::build_control_channel(std::string const& oob_con,
                                        uint64_t peer_id, bool sync,
                                        int timeout_ms) {
  auto ctrl_mem =
      allocator_->allocate(kRingBufferSize, MemoryType::HOST, contexts_[0]);
  auto control_channel = std::make_shared<SendControlChannel>(
      contexts_[0], ctrl_mem, kControlChannelID);
  auto ctrl_info = std::make_shared<RemoteMemInfo>(ctrl_mem);

  // Include OOB server port for back-connection (notifications)
  MetaInfoToExchange ctrl_meta(
      INVALID_PEER_ID, kControlChannelID, control_channel->get_local_meta(),
      ctrl_info, ChannelType::Control, gpu_index_, oob_server_->get_port());
  fillCompressionMeta(ctrl_meta);

  UCCL_LOG(INFO, UCCL_RDMA)
      << "Control Meta: " << ctrl_meta
      << " Local Channel Meta: " << control_channel->get_local_meta()
      << std::endl;

  std::string ctrl_serialized_meta = serialize(ctrl_meta);

  auto promise = std::make_shared<std::promise<uint64_t>>();
  std::future<uint64_t> future = promise->get_future();

  bool sent = oob_client_->send_meta(
      oob_con, ctrl_serialized_meta,
      [this, control_channel, promise,
       peer_id](std::string const& response) mutable {
        MetaInfoToExchange response_meta =
            deserialize<MetaInfoToExchange>(response);
        control_channel->establishChannel(response_meta.channel_meta);
        // Bind WriteReqMeta ring: local mirror (this endpoint's own) +
        // remote slot table on the peer receiver.
        if (write_meta_ring_) {
          control_channel->bindWriteMetaRing(
              write_meta_ring_, response_meta.write_meta_ring_meta);
        }
        this->setSendControlChannel(peer_id, std::move(control_channel));
        // Remember peer's decompress_buf so the SendConnection can target
        // it during compressWriteRequestSplitFirst.
        this->setSendCompressionPeerMeta(peer_id, response_meta);
        promise->set_value(response_meta.peer_id);  // no try/catch; fail-fast
      });

  if (!sent) {
    UCCL_LOG(ERROR) << "Failed to send control channel metadata for peer "
                    << peer_id;
    return -1;
  }

  if (!sync) {
    return peer_id;
  }

  if (future.wait_for(std::chrono::milliseconds(timeout_ms)) ==
      std::future_status::timeout) {
    UCCL_LOG(ERROR) << "Timeout waiting for control channel handshake for peer "
                    << peer_id;
    return -1;
  }

  uint64_t recv_peer_id = future.get();

  UCCL_LOG(INFO, UCCL_RDMA)
      << "Control channel handshake completed successfully for peer "
      << recv_peer_id;

  return static_cast<int>(recv_peer_id);
}

bool RDMAEndpoint::build_data_channels(std::string const& oob_con,
                                       uint64_t peer_id, bool sync,
                                       int timeout_ms) {
  std::vector<std::future<void>> futures;
  futures.reserve(kQpNumPerChannel);

  for (int i = 0; i < kQpNumPerChannel; i++) {
    uint32_t channel_id = i + 1;

    auto channel = std::make_shared<RDMADataChannel>(
        getContextByChannelId(channel_id), channel_id);

    MetaInfoToExchange meta(INVALID_PEER_ID, channel_id,
                            channel->get_local_meta(), nullptr,
                            ChannelType::Normal, gpu_index_);

    UCCL_LOG(INFO, UCCL_RDMA) << meta << std::endl;
    std::string serialized_meta = serialize(meta);

    auto promise = std::make_shared<std::promise<void>>();
    futures.emplace_back(promise->get_future());

    bool sent = oob_client_->send_meta(
        oob_con, serialized_meta,
        [this, channel, channel_id, promise,
         peer_id](std::string const& response) {
          uint64_t peer_rank =
              this->handle_send_meta_response(channel, response);
          addOneSendChannel(peer_id, channel_id, channel);
          promise->set_value();  // no try/catch; fail-fast
        });

    if (!sent) {
      UCCL_LOG(ERROR) << "Failed to send metadata for channel " << channel_id;
      return false;
    }
  }

  if (!sync) {
    UCCL_LOG(INFO, UCCL_RDMA)
        << "Normal channels async build initiated for peer " << peer_id;
    return true;
  }

  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

  for (size_t i = 0; i < futures.size(); i++) {
    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());

    if (remaining.count() <= 0 ||
        futures[i].wait_for(remaining) == std::future_status::timeout) {
      UCCL_LOG(ERROR) << "Timeout waiting for channel " << (i + 1)
                      << " to complete";
      return false;
    }

    futures[i].get();
  }

  UCCL_LOG(INFO, UCCL_RDMA)
      << "All " << kQpNumPerChannel
      << " normal channels built successfully for peer " << peer_id;

  return true;
}
