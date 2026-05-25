#include "common.h"

void serialize_fifo_item(FifoItem const& item, char* buf) {
  std::memcpy(buf + 0, &item.addr, sizeof(uint64_t));
  std::memcpy(buf + 8, &item.size, sizeof(uint32_t));
  std::memcpy(buf + 12, &item.rkey, sizeof(uint32_t));
  std::memcpy(buf + 16, &item.nmsgs, sizeof(uint32_t));
  std::memcpy(buf + 20, &item.rid, sizeof(uint32_t));
  std::memcpy(buf + 24, &item.idx, sizeof(uint64_t));
  std::memcpy(buf + 32, &item.padding, sizeof(item.padding));
}

void deserialize_fifo_item(char const* buf, FifoItem* item) {
  std::memcpy(&item->addr, buf + 0, sizeof(uint64_t));
  std::memcpy(&item->size, buf + 8, sizeof(uint32_t));
  std::memcpy(&item->rkey, buf + 12, sizeof(uint32_t));
  std::memcpy(&item->nmsgs, buf + 16, sizeof(uint32_t));
  std::memcpy(&item->rid, buf + 20, sizeof(uint32_t));
  std::memcpy(&item->idx, buf + 24, sizeof(uint64_t));
  std::memcpy(item->padding, buf + 32, sizeof(item->padding));
}

size_t channelIdToContextId(uint32_t channel_id) {
  return (channel_id == 0) ? 0 : (channel_id - 1) % kNICContextNumber;
}

void ImmData::set_chunk_count(uint16_t chunk_count) {
  data_ = (static_cast<uint32_t>(chunk_count) << 16) | (data_ & 0xFFFF);
}

void ImmData::set_index(uint16_t index) {
  data_ = (data_ & 0xFFFF0000) | (index & kIndexMask) | (data_ & kWriteMetaBit);
}

void ImmData::set_write_meta() { data_ |= kWriteMetaBit; }

ImmData& ImmData::operator=(uint32_t raw) {
  data_ = raw;
  return *this;
}

std::ostream& operator<<(std::ostream& os, ImmData const& imm) {
  os << "ImmData{raw: " << imm.data_ << ", chunk_count: " << imm.chunk_count()
     << ", index: " << imm.index() << "}";
  return os;
}

MessageChunk::MessageChunk(uint64_t off, size_t sz) : offset(off), size(sz) {}

std::ostream& operator<<(std::ostream& os, MessageChunk const& chunk) {
  os << "MessageChunk{offset: " << chunk.offset << ", size: " << chunk.size
     << "}";
  return os;
}

size_t ChunkSplitStrategy::getMessageChunkCount(size_t message_size) {
  constexpr size_t chunk_size_bytes = kMessageChunkSizeKB * 1024;
  if (message_size == 0) return 0;
  size_t chunk_count = (message_size + chunk_size_bytes - 1) / chunk_size_bytes;
  return std::min(chunk_count, static_cast<size_t>(kMaxSplitNum));
}

std::vector<MessageChunk> ChunkSplitStrategy::splitMessageToChunks(
    size_t message_size) {
  size_t chunk_count = getMessageChunkCount(message_size);
  std::vector<MessageChunk> chunks;
  chunks.reserve(chunk_count);
  size_t actual_chunk_size = getRegularChunkSize(message_size, chunk_count);
  for (size_t i = 0; i < chunk_count; ++i) {
    uint64_t offset = i * actual_chunk_size;
    size_t size = std::min(actual_chunk_size, message_size - offset);
    chunks.emplace_back(offset, size);
  }
  return chunks;
}

size_t ChunkSplitStrategy::getRegularChunkSize(size_t message_size,
                                               size_t chunk_count) {
  if (chunk_count == 0 || message_size == 0) return 0;
  return (message_size + chunk_count - 1) / chunk_count;
}

void copyRKeyArrayFromMRArray(MRArray const& mr_array, RKeyArray& rkey_array) {
  for (uint32_t ctx = 0; ctx < kNICContextNumber; ++ctx) {
    ibv_mr* mr = mr_array.getKeyByContextID(ctx);
    uint32_t rkey = mr ? mr->rkey : 0;
    rkey_array.setKeyByContextID(ctx, rkey);
  }
}

void copyRKeysFromMRArrayToBytes(MRArray const& mr_array, char* dst,
                                 size_t dst_size) {
  constexpr size_t needed = sizeof(uint32_t) * kNICContextNumber;
  assert(dst_size >= needed);

  uint32_t* out = reinterpret_cast<uint32_t*>(dst);

  for (uint32_t ctx = 0; ctx < kNICContextNumber; ++ctx) {
    ibv_mr* mr = mr_array.getKeyByContextID(ctx);
    out[ctx] = mr ? mr->rkey : 0;
  }
}

std::ostream& operator<<(std::ostream& os, ChannelMetaData const& meta) {
  os << "ChannelMetaData{qpn: " << meta.qpn << ", gid: ";
  std::ios_base::fmtflags flags = os.flags();
  for (int i = 0; i < 16; ++i) {
    os << std::hex << std::setw(2) << std::setfill('0')
       << static_cast<int>(meta.gid.raw[i]);
    if (i == 7) os << ":";
  }
  os.flags(flags);
  os << "}";
  return os;
}

OOBMetaData::OOBMetaData() : server_ip(""), server_port(0), gpu_id(0) {}

OOBMetaData::OOBMetaData(std::string conn_key_, uint16_t remote_port_)
    : server_ip(std::move(conn_key_)), server_port(remote_port_), gpu_id(0) {}

std::ostream& operator<<(std::ostream& os, OOBMetaData const& meta) {
  os << "OOBMetaData{server_ip: " << meta.server_ip
     << ", server_port: " << meta.server_port << ", gpu_id: " << meta.gpu_id
     << "}";
  return os;
}

bool CQMeta::hasIMM() const { return op_code == IBV_WC_RECV_RDMA_WITH_IMM; }

std::ostream& operator<<(std::ostream& os, CQMeta const& meta) {
  os << "CQMeta{wr_id: " << meta.wr_id << ", op_code: " << meta.op_code
     << ", len: " << meta.len << ", imm: " << meta.imm
     << ", hasIMM: " << (meta.op_code == IBV_WC_RECV_RDMA_WITH_IMM) << "}";
  return os;
}

RegMemBlock::RegMemBlock() : addr(nullptr), size(0), type(MemoryType::HOST) {}

RegMemBlock::RegMemBlock(void* a, size_t s, MemoryType t)
    : addr(a), size(s), type(t) {}

RegMemBlock::RegMemBlock(void* a, size_t s, MRArray const& mr_array_in,
                         MemoryType t)
    : addr(a), size(s), type(t) {
  mr_array.copyFrom(mr_array_in);
}

bool RegMemBlock::operator==(RegMemBlock const& other) const {
  return addr == other.addr && size == other.size && type == other.type;
}

void RegMemBlock::setMRByContextID(uint32_t context_id, struct ibv_mr* mr) {
  mr_array.setKeyByContextID(context_id, mr);
}

void RegMemBlock::setMRByChannelID(uint32_t channel_id, struct ibv_mr* mr) {
  mr_array.setKeyByChannelID(channel_id, mr);
}

struct ibv_mr* RegMemBlock::getMRByChannelID(uint32_t channel_id) const {
  return mr_array.getKeyByChannelID(channel_id);
}

struct ibv_mr* RegMemBlock::getMRByContextID(uint32_t context_id) const {
  return mr_array.getKeyByContextID(context_id);
}

uint32_t RegMemBlock::getKeyByChannelID(uint32_t channel_id) const {
  struct ibv_mr* mr = getMRByChannelID(channel_id);
  return mr ? mr->rkey : 0;
}

uint32_t RegMemBlock::getKeyByContextID(uint32_t context_id) const {
  struct ibv_mr* mr = getMRByContextID(context_id);
  return mr ? mr->rkey : 0;
}

std::ostream& operator<<(std::ostream& os, RegMemBlock const& block) {
  os << "RegMemBlock{addr: " << block.addr << ", size: " << block.size
     << ", type: " << (block.type == MemoryType::HOST ? "HOST" : "GPU")
     << block.mr_array << std::endl;

  os << "}";
  return os;
}

RemoteMemInfo::RemoteMemInfo() : addr(0), length(0), type(MemoryType::HOST) {}

RemoteMemInfo::RemoteMemInfo(uint64_t a, size_t len, RKeyArray const& rkey,
                             MemoryType t)
    : addr(a), length(len), type(t) {
  rkey_array.copyFrom(rkey);
}

RemoteMemInfo::RemoteMemInfo(uint64_t a, size_t len, MRArray const& mrs,
                             MemoryType t)
    : addr(a), length(len), type(t) {
  copyRKeyArrayFromMRArray(mrs, rkey_array);
}

RemoteMemInfo::RemoteMemInfo(RegMemBlock const& block)
    : addr(reinterpret_cast<uint64_t>(block.addr)),
      length(block.size),
      type(block.type) {
  copyRKeyArrayFromMRArray(block.mr_array, rkey_array);
}

RemoteMemInfo::RemoteMemInfo(std::shared_ptr<RegMemBlock> const block)
    : addr(reinterpret_cast<uint64_t>(block->addr)),
      length(block->size),
      type(block->type) {
  copyRKeyArrayFromMRArray(block->mr_array, rkey_array);
}

uint32_t RemoteMemInfo::getKeyByChannelID(uint32_t channel_id) const {
  return rkey_array.getKeyByChannelID(channel_id);
}

uint32_t RemoteMemInfo::getKeyByContextID(size_t context_id) const {
  return rkey_array.getKeyByContextID(context_id);
}

std::ostream& operator<<(std::ostream& os, RemoteMemInfo const& info) {
  os << "RemoteMemInfo{addr: 0x" << std::hex << info.addr << std::dec
     << ", length: " << info.length
     << ", type: " << (info.type == MemoryType::HOST ? "HOST" : "GPU")
     << "rkey_array " << info.rkey_array << "}" << std::endl;
  return os;
}

RDMARecvRequest::RDMARecvRequest(std::shared_ptr<RegMemBlock> local)
    : local_mem(local) {}

uint32_t RDMARecvRequest::getLocalKey() const {
  return local_mem->getKeyByChannelID(channel_id);
}

uint64_t RDMARecvRequest::getLocalAddress() const {
  return reinterpret_cast<uint64_t>(local_mem->addr);
}

uint32_t RDMARecvRequest::getLocalLen() const { return local_mem->size; }

std::ostream& operator<<(std::ostream& os, RDMARecvRequest const& req) {
  os << "RDMARecvRequest{";
  os << "from_peer_id: " << req.from_peer_id
     << ", to_peer_id: " << req.to_peer_id
     << ", channel_id: " << req.channel_id;
  if (req.local_mem) {
    os << ", local_mem: " << *req.local_mem;
  } else {
    os << ", local_mem: nullptr";
  }
  os << "}";
  return os;
}

SendReqMeta::SendReqMeta()
    : peer_id(0),
      channel_id(0),
      remote_mem{},
      expected_chunk_count(0),
      received_chunk_count(0) {}

SendReqMeta::SendReqMeta(uint32_t pid, uint32_t cid, RemoteMemInfo const& rmem,
                         uint32_t expected, uint32_t received)
    : peer_id(pid),
      channel_id(cid),
      remote_mem(rmem),
      expected_chunk_count(expected),
      received_chunk_count(received) {}

SendReqMeta::SendReqMeta(std::shared_ptr<RDMARecvRequest> rev_req) {
  peer_id = rev_req->from_peer_id;
  channel_id = rev_req->channel_id;
  remote_mem = rev_req->local_mem;
  local_mem = *(rev_req->local_compression_mem ? rev_req->local_compression_mem
                                               : rev_req->local_mem);
  float_type = rev_req->compress_ctx ? rev_req->compress_ctx->getFloatType()
                                     : FloatType::kUndefined;
  expected_chunk_count =
      ChunkSplitStrategy::getMessageChunkCount(local_mem.size);
  received_chunk_count = 0;
}

std::ostream& operator<<(std::ostream& os, SendReqMeta const& meta) {
  os << "SendReqMeta{peer_id: " << meta.peer_id
     << ", channel_id: " << meta.channel_id
     << ", remote_mem: " << meta.remote_mem
     << ", expected_chunk_count: " << meta.expected_chunk_count
     << ", received_chunk_count: " << meta.received_chunk_count << "}";
  return os;
}

SendReqMetaOnRing::SendReqMetaOnRing() : meta{}, flag{ReqFlag::PENDING} {}

SendReqMetaOnRing::SendReqMetaOnRing(SendReqMetaOnRing const& other)
    : meta(other.meta), flag{other.flag.load(std::memory_order_relaxed)} {}

SendReqMetaOnRing& SendReqMetaOnRing::operator=(
    SendReqMetaOnRing const& other) {
  meta = other.meta;
  flag.store(other.flag.load(std::memory_order_relaxed),
             std::memory_order_relaxed);
  return *this;
}

SendReqMeta SendReqMetaOnRing::getSendReqMeta() const { return meta; }

void SendReqMetaOnRing::setSendReqMeta(SendReqMeta const& m) { meta = m; }

std::ostream& operator<<(std::ostream& os, SendReqMetaOnRing const& ring) {
  auto flag_val = ring.flag.load(std::memory_order_relaxed);
  os << "SendReqMetaOnRing{meta: " << ring.meta << ", flag: "
     << (flag_val == ReqFlag::PENDING       ? "PENDING"
         : flag_val == ReqFlag::IN_PROGRESS ? "IN_PROGRESS"
                                            : "IS_DONE")
     << "}";
  return os;
}

RDMASendRequest::RDMASendRequest(std::shared_ptr<RegMemBlock> local,
                                 std::shared_ptr<RemoteMemInfo> remote,
                                 ImmData imm, bool signaled)
    : local_mem(local),
      remote_mem(remote),
      imm_data(imm),
      need_signaled(signaled) {}

RDMASendRequest::RDMASendRequest(std::shared_ptr<RDMASendRequest> other,
                                 std::shared_ptr<RegMemBlock> local,
                                 ImmData imm, bool signaled)
    : local_mem(local),
      remote_mem(other->remote_mem),
      imm_data(imm),
      need_signaled(signaled) {}

RDMASendRequest::RDMASendRequest(RDMASendRequest const& other,
                                 std::shared_ptr<RegMemBlock> local,
                                 ImmData imm, bool signaled)
    : local_mem(local),
      remote_mem(other.remote_mem),
      imm_data(imm),
      need_signaled(signaled) {}

uint32_t RDMASendRequest::getLocalKey() const {
  return local_mem->getKeyByChannelID(channel_id);
}

uint32_t RDMASendRequest::getRemoteKey() const {
  return remote_mem->getKeyByChannelID(channel_id);
}

uint64_t RDMASendRequest::getLocalAddress() const {
  return reinterpret_cast<uint64_t>(local_mem->addr);
}

uint64_t RDMASendRequest::getRemoteAddress() const { return remote_mem->addr; }

ImmData RDMASendRequest::getImm() const { return imm_data; }

uint32_t RDMASendRequest::getLocalLen() const { return local_mem->size; }

std::ostream& operator<<(std::ostream& os, RDMASendRequest const& req) {
  os << "RDMASendRequest{";
  os << "from_peer_id: " << req.from_peer_id
     << ", to_peer_id: " << req.to_peer_id << ", channel_id: " << req.channel_id
     << ", imm_data: " << req.imm_data
     << ", need_signaled: " << req.need_signaled;
  if (req.local_mem) {
    os << ", local_mem: " << *req.local_mem;
  } else {
    os << ", local_mem: nullptr";
  }
  if (req.remote_mem) {
    os << ", remote_mem: " << *req.remote_mem;
  } else {
    os << ", remote_mem: nullptr";
  }
  os << "}";
  return os;
}

MetaInfoToExchange::MetaInfoToExchange()
    : peer_id(0),
      channel_id(0),
      flag(ChannelType::Normal),
      channel_meta{},
      mem_meta{},
      gpu_id(0),
      oob_port(0) {}

MetaInfoToExchange::MetaInfoToExchange(
    int32_t pid, int32_t cid, std::shared_ptr<ChannelMetaData> ch_meta,
    std::shared_ptr<RemoteMemInfo> mem_meta_ptr, ChannelType flag_in, int gid,
    uint16_t oob_p)
    : peer_id(pid),
      channel_id(cid),
      flag(flag_in),
      channel_meta{},
      mem_meta{},
      gpu_id(gid),
      oob_port(oob_p) {
  if (ch_meta) {
    channel_meta = *ch_meta;
  }
  if (mem_meta_ptr) {
    mem_meta = *mem_meta_ptr;
  }
}

std::ostream& operator<<(std::ostream& os, MetaInfoToExchange const& meta) {
  os << "=== MetaInfoToExchange ===" << std::endl;
  os << "  peer_id: " << meta.peer_id << std::endl;
  os << "  channel_id: " << meta.channel_id << std::endl;

  os << "  channel_meta:" << std::endl;
  os << "    qpn: " << meta.channel_meta.qpn << std::endl;
  os << "    gid: ";

  std::ios_base::fmtflags flags = os.flags();

  for (int i = 0; i < 16; ++i) {
    os << std::hex << std::setw(2) << std::setfill('0')
       << static_cast<int>(meta.channel_meta.gid.raw[i]);
    if (i == 7) os << ":";
  }

  os.flags(flags);
  os << std::endl;

  os << "  mem_meta:" << meta.mem_meta << std::endl;
  os << "  gpu_id: " << meta.gpu_id << std::endl;
  os << "=========================" << std::endl;

  return os;
}

int make_socket_non_blocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags == -1) return -1;
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) return -1;
  return 0;
}

ssize_t try_send(int fd, char const* buf, size_t len) {
  ssize_t n = ::send(fd, buf, len, MSG_NOSIGNAL);
  if (n < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
    return -1;
  }
  return n;
}

bool is_nic_usable(std::string const& nic_name, NicMode mode) {
  if (mode == NicMode::EFA && strncmp(nic_name.c_str(), "rdmap", 5) != 0) {
    return false;
  }

  int dev_count = 0;
  ibv_device** dev_list = ibv_get_device_list(&dev_count);
  if (!dev_list) {
    return false;
  }
  bool usable = false;
  for (int i = 0; i < dev_count; ++i) {
    if (std::strcmp(ibv_get_device_name(dev_list[i]), nic_name.c_str()) != 0) {
      continue;
    }
    ibv_context* ctx = ibv_open_device(dev_list[i]);
    if (!ctx) break;

    ibv_port_attr port_attr{};
    if (ibv_query_port(ctx, kPortNum, &port_attr) == 0) {
      if (mode == NicMode::EFA) {
        if (port_attr.state == IBV_PORT_ACTIVE &&
            (port_attr.link_layer == IBV_LINK_LAYER_UNSPECIFIED ||
             port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ||
             port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND)) {
          usable = true;
        }
      } else {
        if (port_attr.state == IBV_PORT_ACTIVE &&
            (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ||
             port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) &&
            (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET ||
             port_attr.gid_tbl_len > 0)) {
          usable = true;
        }
      }
    }
    ibv_close_device(ctx);
    break;
  }
  ibv_free_device_list(dev_list);
  return usable;
}

namespace std {
size_t hash<RegMemBlock>::operator()(RegMemBlock const& block) const {
  size_t h1 = hash<size_t>{}(block.size);
  size_t h2 = hash<int>{}(static_cast<int>(block.type));
  return h1 ^ (h2 << 1);
}
}  // namespace std
