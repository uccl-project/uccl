#include "rdma_endpoint.h"
#include "util/debug.h"
#include "util/gpu_rt.h"
#include "util/util.h"

// ── Lifecycle ────────────────────────────────────────────────────────────────
RDMAEndpoint::RDMAEndpoint(int gpu_index, uint64_t port,
                           std::vector<size_t> const& device_ids)
    : gpu_index_(gpu_index), next_send_peer_id_(0), next_recv_peer_id_(0) {
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
  init_compressor();
}

RDMAEndpoint::~RDMAEndpoint() {
  if (oob_client_) {
    oob_client_->stop();
  }
  if (oob_server_) {
    oob_server_->stop();
  }
}

// ── Initialization ───────────────────────────────────────────────────────────
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

  initialize_contexts(actual_device_ids);
  UCCL_LOG(INFO, UCCL_RDMA)
      << "RDMAEndpoint initialized with " << contexts_.size()
      << " context(s) for GPU " << gpu_index;

  for (auto dev : actual_device_ids) {
    auto device = RdmaDeviceManager::instance().get_device(dev);
    std::cout << "GPU " << gpu_index << " uses device " << dev << " ("
              << device->name() << ")" << std::endl;
  }
  return true;
}

void RDMAEndpoint::initialize_contexts(std::vector<size_t> const& device_ids) {
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
          << " (" << device_manager.get_device(device_id)->name() << ")";
      continue;
    }

    auto device = device_manager.get_device(device_id);
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

std::shared_ptr<RdmaContext> RDMAEndpoint::get_context_by_channel_id(
    uint32_t channel_id) const {
  return contexts_[channel_id_to_context_id(channel_id)];
}

void RDMAEndpoint::init_compressor() {
  Compressor& compressor = Compressor::get_instance();
  auto compress_buf = compressor.get_compress_buffer();
  auto decompress_buf = compressor.get_decompress_buffer();
  if (compress_buf && !reg_mem(compress_buf)) {
    UCCL_LOG(ERROR) << "Failed to register compression buffer";
  }
  if (decompress_buf && !reg_mem(decompress_buf)) {
    UCCL_LOG(ERROR) << "Failed to register decompression buffer";
  }
  if (compressor.get_compress_strategy() == CompressStrategy::kNone) return;
  // Host-side rings used to carry per-message compression metadata and
  // completion acks. Registered on every context slot so any data/control
  // QP can target them via rkey lookup.
  ack_ring_ = allocator_->allocate(kAckRingBytes, MemoryType::HOST, nullptr);
  write_meta_ring_ =
      allocator_->allocate(kWriteMetaRingBytes, MemoryType::HOST, nullptr);
  std::memset(ack_ring_->addr, 0, kAckRingBytes);
  std::memset(write_meta_ring_->addr, 0, kWriteMetaRingBytes);
  if (!reg_mem(ack_ring_) || !reg_mem(write_meta_ring_)) {
    throw std::runtime_error("Failed to register compression metadata rings");
  }
}

void RDMAEndpoint::fill_compression_meta(MetaInfoToExchange& m) const {
  auto decomp = Compressor::get_instance().get_decompress_buffer();
  if (decomp) m.decompress_buf_meta = RemoteMemInfo(*decomp);
  if (write_meta_ring_)
    m.write_meta_ring_meta = RemoteMemInfo(*write_meta_ring_);
  if (ack_ring_) m.ack_ring_meta = RemoteMemInfo(*ack_ring_);
}

// ── Accessors ────────────────────────────────────────────────────────────────
int RDMAEndpoint::gpu_index() const { return gpu_index_; }

size_t RDMAEndpoint::context_count() const { return contexts_.size(); }

std::shared_ptr<RegMemBlock> RDMAEndpoint::ack_ring() const {
  return ack_ring_;
}

std::shared_ptr<RegMemBlock> RDMAEndpoint::write_meta_ring() const {
  return write_meta_ring_;
}

// ── Memory registration ──────────────────────────────────────────────────────
bool RDMAEndpoint::reg_mem(std::shared_ptr<RegMemBlock> reg_block) {
  if (unlikely(!reg_block)) {
    UCCL_LOG(ERROR) << "Error: reg_mem called with null reg_block";
    return false;
  }

  // Register once per unique RdmaContext (contexts sharing the same NIC
  // device are the same shared_ptr - see initialize_contexts).  This avoids
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
      reg_block->set_mr_by_context_id(context_id, it->second);
      continue;
    }

    struct ibv_mr* mr = context->reg_mem(reg_block->addr, reg_block->size);

    if (unlikely(!mr)) {
      UCCL_LOG(ERROR) << "Error: ibv_reg_mr failed for block at "
                      << reg_block->addr << " size " << reg_block->size
                      << " context_id " << context_id;
      return false;
    }
    reg_block->set_mr_by_context_id(context_id, mr);
    registered[context.get()] = mr;
  }

  return true;
}

bool RDMAEndpoint::dereg_mem(std::shared_ptr<RegMemBlock> reg_block) {
  if (unlikely(!reg_block)) {
    UCCL_LOG(ERROR) << "Error: dereg_mem called with null reg_block";
    return false;
  }
  // Deduplicate MR pointers - shared contexts have the same MR in
  // multiple slots, so we must not double-free.
  std::unordered_set<ibv_mr*> freed;
  for (uint32_t ctx = 0; ctx < kNICContextNumber; ++ctx) {
    ibv_mr* mr = reg_block->get_mr_by_context_id(ctx);
    if (mr && freed.insert(mr).second) {
      RdmaContext::dereg_mem(mr);
    }
  }
  return true;
}

int RDMAEndpoint::uccl_regmr(void* const data, size_t const len,
                             MRArray& mr_array,
                             std::vector<MrCacheHandleRef>& cache_refs,
                             CompressCtx compress_ctx) {
  if (unlikely(!data)) {
    UCCL_LOG(ERROR) << "Error: uccl_regmr called with null data";
    return -1;
  }
  Compressor::get_instance().prepare_split_context(data, len, compress_ctx);
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
        ref.context->release_cached_mr(ref.entry);
      }
      cache_refs.clear();
      return -1;
    }

    auto it = registered.find(context.get());
    if (it != registered.end()) {
      // Same NIC device - reuse the already-registered MR.
      mr_array.set_key_by_context_id(context_id, it->second);
      continue;
    }

    MrCacheEntry* entry = context->acquire_cached_mr(data, len);
    struct ibv_mr* mr = entry ? entry->mr : nullptr;

    if (unlikely(!mr)) {
      UCCL_LOG(ERROR) << "Error " << errno << " " << strerror(errno)
                      << ": ibv_reg_mr_iova2 failed for data at " << data
                      << " size " << len << " context_id " << context_id;
      for (auto const& ref : cache_refs) {
        ref.context->release_cached_mr(ref.entry);
      }
      cache_refs.clear();
      return -1;
    }

    // Store the MR in the mr_map using context_id as key
    mr_array.set_key_by_context_id(context_id, mr);
    registered[context.get()] = mr;
    cache_refs.push_back({context, entry});
  }

  return 0;
}

void RDMAEndpoint::uccl_deregmr(
    std::vector<MrCacheHandleRef> const& cache_refs) {
  for (auto const& ref : cache_refs) {
    if (likely(ref.context != nullptr)) {
      ref.context->release_cached_mr(ref.entry);
    }
  }
}

// ── Connection setup ─────────────────────────────────────────────────────────
void RDMAEndpoint::add_peer_oob_meta(
    std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>> const&
        new_meta) {
  for (auto const& [peer_id, meta_ptr] : new_meta) {
    peer_oob_meta_[peer_id] = meta_ptr;
  }
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
      allocator_->allocate(kWriteMetaRingBytes, MemoryType::HOST, contexts_[0]);
  auto control_channel = std::make_shared<SendControlChannel>(
      contexts_[0], ctrl_mem, kControlChannelID);
  auto ctrl_info = std::make_shared<RemoteMemInfo>(ctrl_mem);

  // Include OOB server port for back-connection (notifications)
  MetaInfoToExchange ctrl_meta(
      INVALID_PEER_ID, kControlChannelID, control_channel->get_local_meta(),
      ctrl_info, ChannelType::Control, gpu_index_, oob_server_->get_port());
  fill_compression_meta(ctrl_meta);

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
        control_channel->establish_channel(response_meta.channel_meta);
        // Bind WriteReqMeta ring: local mirror (this endpoint's own) +
        // remote slot table on the peer receiver.
        if (write_meta_ring_) {
          control_channel->bind_write_meta_ring(
              write_meta_ring_, response_meta.write_meta_ring_meta);
        }
        this->set_send_control_channel(peer_id, std::move(control_channel));
        // Remember peer's decompress_buf so the SendConnection can target
        // it during compress_write_request_split_first.
        this->set_send_compression_peer_meta(peer_id, response_meta);
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
        get_context_by_channel_id(channel_id), channel_id);

    MetaInfoToExchange meta(INVALID_PEER_ID, channel_id,
                            channel->get_local_meta(), nullptr,
                            ChannelType::Data, gpu_index_);

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
          add_one_send_channel(peer_id, channel_id, channel);
          promise->set_value();  // no try/catch; fail-fast
        });

    if (!sent) {
      UCCL_LOG(ERROR) << "Failed to send metadata for channel " << channel_id;
      return false;
    }
  }

  if (!sync) {
    UCCL_LOG(INFO, UCCL_RDMA)
        << "Data channels async build initiated for peer " << peer_id;
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
      << " data channels built successfully for peer " << peer_id;

  return true;
}

uint64_t RDMAEndpoint::handle_send_meta_response(
    std::shared_ptr<RDMADataChannel> channel, std::string const& response) {
  // Deserialize response as MetaInfoToExchange
  MetaInfoToExchange response_meta = deserialize<MetaInfoToExchange>(response);
  UCCL_LOG(INFO, UCCL_RDMA) << response_meta;
  channel->establish_channel(response_meta.channel_meta);
  return response_meta.peer_id;
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
          if (get_or_create_recv_group(peer_id)->channel_count() ==
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

  auto context_id = channel_id_to_context_id(meta.channel_id);
  std::shared_ptr<RdmaContext> ctx_ptr = contexts_[context_id];

  if (meta.flag == ChannelType::Control) {
    uint64_t actual_peer_id =
        next_recv_peer_id_.fetch_add(1, std::memory_order_relaxed);

    auto ctrl_mem =
        allocator_->allocate(kWriteMetaRingBytes, MemoryType::HOST, ctx_ptr);
    UCCL_LOG(INFO, UCCL_RDMA)
        << "process_meta: Allocated " << ctrl_mem->size
        << " bytes for recv control channel ring buffer at " << ctrl_mem->addr;

    auto recv_ctrl_channel = std::make_shared<RecvControlChannel>(
        ctx_ptr, meta, ctrl_mem, meta.channel_id);
    // Receiver owns the authoritative WriteReqMeta ring.
    if (write_meta_ring_)
      recv_ctrl_channel->bind_write_meta_ring(write_meta_ring_);

    // Create response (include our OOB port for potential future use)
    RemoteMemInfo ctrl_info(ctrl_mem);
    MetaInfoToExchange response(
        INVALID_PEER_ID, meta.channel_id, recv_ctrl_channel->get_local_meta(),
        nullptr, ChannelType::Control, gpu_index_, oob_server_->get_port());
    response.mem_meta = ctrl_info;
    fill_compression_meta(response);
    UCCL_LOG(INFO, UCCL_RDMA)
        << "response (control channel):::::::" << response;
    output = serialize(response);

    // Set the control channel
    auto ctrl_ch_copy = recv_ctrl_channel;
    set_recv_control_channel(actual_peer_id, std::move(ctrl_ch_copy));
    // Remember peer's ack_ring address+rkey - the receive side needs it
    // when decompress finishes and we post an ack back to the sender.
    set_recv_compression_peer_meta(actual_peer_id, meta);

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
                                ChannelType::Data, gpu_index_);
    UCCL_LOG(INFO, UCCL_RDMA) << "response:::::::" << response;
    output = serialize(response);
    add_one_recv_channel(actual_peer_id, meta.channel_id, new_channel);
  }
}

void RDMAEndpoint::stop_accept() {
  stop_accept_.store(true, std::memory_order_release);
}

// ── SendConnection group registry ────────────────────────────────────────────
std::shared_ptr<SendConnection> RDMAEndpoint::get_or_create_send_group(
    uint64_t peer_id) {
  {
    std::shared_lock read_lock(send_channel_mutex_);
    auto it = send_channel_groups_.find(peer_id);
    if (it != send_channel_groups_.end()) return it->second;
  }
  auto* ctx =
      (!contexts_.empty() && contexts_[0]) ? contexts_[0]->get_ctx() : nullptr;
  double link_bw =
      uccl::cc::get_link_bandwidth_bps(ctx, "UCCL_P2P_RDMA_LINK_GBPS");
  {
    std::unique_lock write_lock(send_channel_mutex_);
    auto [it, inserted] = send_channel_groups_.try_emplace(
        peer_id, std::make_shared<SendConnection>(link_bw));
    return it->second;
  }
}

void RDMAEndpoint::add_one_send_channel(
    uint64_t peer_id, uint32_t channel_id,
    std::shared_ptr<RDMADataChannel> new_channel) {
  auto group_ptr = get_or_create_send_group(peer_id);
  group_ptr->add_channel(channel_id, new_channel);
}

void RDMAEndpoint::set_send_control_channel(
    uint64_t peer_id, std::shared_ptr<SendControlChannel>&& ctrl_channel) {
  auto it = get_or_create_send_group(peer_id);
  send_channel_groups_[peer_id]->set_control_channel(
      std::forward<std::shared_ptr<SendControlChannel>>(ctrl_channel));
}

void RDMAEndpoint::set_send_compression_peer_meta(
    uint64_t peer_id, MetaInfoToExchange const& peer) {
  if (!ack_ring_ || peer.decompress_buf_meta.length == 0) return;
  auto group = get_or_create_send_group(peer_id);
  group->set_remote_decompress_buf(peer.decompress_buf_meta);
  group->set_local_ack_ring(ack_ring_);
}

// ── RecvConnection group registry ────────────────────────────────────────────
std::shared_ptr<RecvConnection> RDMAEndpoint::get_or_create_recv_group(
    uint64_t peer_id) {
  {
    std::shared_lock read_lock(recv_channel_mutex_);
    auto it = recv_channel_groups_.find(peer_id);
    if (it != recv_channel_groups_.end()) {
      return it->second;
    }
  }
  {
    std::unique_lock write_lock(recv_channel_mutex_);
    auto [it, inserted] = recv_channel_groups_.try_emplace(
        peer_id,
        std::make_shared<RecvConnection>());  // try_emplace constructs only
                                              // if inserting
    return it->second;
  }
}

void RDMAEndpoint::add_one_recv_channel(
    uint64_t peer_id, uint32_t channel_id,
    std::shared_ptr<RDMADataChannel> new_channel) {
  std::shared_ptr<RecvConnection> group_ptr = get_or_create_recv_group(peer_id);
  group_ptr->add_channel(channel_id, new_channel);
}

void RDMAEndpoint::set_recv_control_channel(
    uint64_t peer_id, std::shared_ptr<RecvControlChannel>&& ctrl_channel) {
  auto it = get_or_create_recv_group(peer_id);
  recv_channel_groups_[peer_id]->set_control_channel(
      std::forward<std::shared_ptr<RecvControlChannel>>(ctrl_channel));
}

void RDMAEndpoint::set_recv_compression_peer_meta(
    uint64_t peer_id, MetaInfoToExchange const& peer) {
  if (peer.ack_ring_meta.length == 0) return;
  auto group = get_or_create_recv_group(peer_id);
  group->set_remote_ack_ring(peer.ack_ring_meta);
}

// ── One-sided transfer and completion ────────────────────────────────────────
int64_t RDMAEndpoint::write_or_read(std::shared_ptr<RDMASendRequest> req) {
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
    wr_id = send_group->post_write_or_read(req);

    if (wr_id < 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  return wr_id;
}

void RDMAEndpoint::check_send_complete(uint64_t peer_id, int64_t wr_id) {
  UCCL_LOG(INFO, UCCL_RDMA)
      << "check_send_complete - peer_id: " << peer_id << ", wr_id: " << wr_id;

  auto it = send_channel_groups_.find(peer_id);
  if (it == send_channel_groups_.end()) {
    throw std::runtime_error("Send channel group not found for peer_id: " +
                             std::to_string(peer_id));
  }

  auto send_group = it->second;
  while (!send_group->check_completion(wr_id)) {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }
  UCCL_LOG(INFO, UCCL_RDMA)
      << "check_send_complete - Completed for peer_id: " << peer_id
      << ", wr_id: " << wr_id;
}

bool RDMAEndpoint::check_send_complete_once(uint64_t peer_id, int64_t wr_id) {
  // UCCL_LOG(INFO, UCCL_RDMA) << "check_send_complete - peer_id: " << peer_id
  //           << ", wr_id: " << wr_id;
  std::shared_lock<std::shared_mutex> lock(send_channel_mutex_);
  auto it = send_channel_groups_.find(peer_id);
  if (unlikely(it == send_channel_groups_.end())) {
    throw std::runtime_error("Send channel group not found for peer_id: " +
                             std::to_string(peer_id));
  }

  auto send_group = it->second;
  return send_group->check_completion(wr_id);
}

SendConnection* RDMAEndpoint::get_send_group_raw(uint64_t peer_id) {
  std::shared_lock<std::shared_mutex> lock(send_channel_mutex_);
  auto it = send_channel_groups_.find(peer_id);
  if (it == send_channel_groups_.end()) return nullptr;
  return it->second.get();
}

// ── Polling and batching ─────────────────────────────────────────────────────
void RDMAEndpoint::recv_routine() {
  std::shared_lock<std::shared_mutex> lock(recv_channel_mutex_);
  for (auto& [peer_id, recv_group] : recv_channel_groups_) {
    if (recv_group) recv_group->recv_routine();
  }
}

void RDMAEndpoint::send_routine() {
  std::shared_lock<std::shared_mutex> lock(send_channel_mutex_);
  for (auto& [peer_id, send_group] : send_channel_groups_) {
    if (send_group) send_group->send_routine();
  }
}

bool RDMAEndpoint::has_pending_compressed_send() const {
  std::shared_lock<std::shared_mutex> lock(send_channel_mutex_);
  for (auto const& [peer_id, send_group] : send_channel_groups_) {
    if (send_group && send_group->has_pending_compressed()) return true;
  }
  return false;
}

void RDMAEndpoint::flush_all_sends() {
  std::shared_lock<std::shared_mutex> lock(send_channel_mutex_);
  for (auto& [peer_id, send_group] : send_channel_groups_) {
    if (send_group) send_group->flush_batches();
  }
}

void RDMAEndpoint::create_unified_p2p_socket() {}
