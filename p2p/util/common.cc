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

namespace std {
size_t hash<RegMemBlock>::operator()(RegMemBlock const& block) const {
  size_t h1 = hash<size_t>{}(block.size);
  size_t h2 = hash<int>{}(static_cast<int>(block.type));
  return h1 ^ (h2 << 1);
}
}  // namespace std
