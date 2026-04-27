#include "../util/utils.h"
#include "oob.h"
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <sstream>
#include <string>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

namespace {

constexpr int kSocketPollMs = 200;
constexpr int kRelayWaitMs = 50;
constexpr size_t kBufferCompactMinPrefix = 4096;
constexpr int kLeaderReadyWaitDefaultMs = 30000;

void compact_prefixed_buffer(std::string& buffer, size_t& offset) {
  if (offset == 0) return;
  if (offset == buffer.size()) {
    buffer.clear();
    offset = 0;
    return;
  }
  if (offset >= kBufferCompactMinPrefix && offset * 2 >= buffer.size()) {
    buffer.erase(0, offset);
    offset = 0;
  }
}

enum class SocketFrameType : uint8_t {
  SyncRequest = 1,
  SyncDone = 2,
  Publish = 3,
};

int env_int_or_default(char const* name, int default_value) {
  char const* v = std::getenv(name);
  if (!v || v[0] == '\0') return default_value;
  try {
    return std::stoi(v);
  } catch (...) {
    return default_value;
  }
}

bool env_bool_or_default(char const* name, bool default_value) {
  char const* v = std::getenv(name);
  if (!v || v[0] == '\0') return default_value;
  if (v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F') {
    return false;
  }
  return true;
}

bool connect_with_timeout(int fd, sockaddr_in const& addr, int timeout_ms) {
  int const current_flags = ::fcntl(fd, F_GETFL, 0);
  if (current_flags < 0) return false;
  if (::fcntl(fd, F_SETFL, current_flags | O_NONBLOCK) != 0) return false;

  int rc =
      ::connect(fd, reinterpret_cast<sockaddr const*>(&addr), sizeof(addr));
  if (rc == 0) {
    (void)::fcntl(fd, F_SETFL, current_flags | O_NONBLOCK);
    return true;
  }
  if (errno != EINPROGRESS) return false;

  pollfd pfd{};
  pfd.fd = fd;
  pfd.events = POLLOUT;
  rc = ::poll(&pfd, 1, timeout_ms > 0 ? timeout_ms : kSocketPollMs);
  if (rc <= 0) return false;

  int so_error = 0;
  socklen_t so_error_len = sizeof(so_error);
  if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &so_error_len) != 0) {
    return false;
  }
  if (so_error != 0) {
    errno = so_error;
    return false;
  }

  return ::fcntl(fd, F_SETFL, current_flags | O_NONBLOCK) == 0;
}

std::string kv_namespace_for_port(int port) {
  char const* override_ns = std::getenv("UHM_OOB_NAMESPACE");
  std::string seed =
      std::string(override_ns && override_ns[0] != '\0' ? override_ns
                                                        : generate_host_id()) +
      ":" + std::to_string(port);
  uint64_t h = static_cast<uint64_t>(std::hash<std::string>{}(seed));
  std::ostringstream oss;
  oss << "/uk_oob_kv_" << std::hex << h;
  return oss.str();
}

bool read_process_start_ticks(pid_t pid, uint64_t& out_ticks) {
  out_ticks = 0;
  std::ostringstream path;
  path << "/proc/" << static_cast<long long>(pid) << "/stat";
  int const fd = ::open(path.str().c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  char buf[2048] = {};
  ssize_t const n = ::read(fd, buf, sizeof(buf) - 1);
  ::close(fd);
  if (n <= 0) return false;
  std::string const stat(buf, static_cast<size_t>(n));
  size_t const right_paren = stat.rfind(')');
  if (right_paren == std::string::npos || right_paren + 2 >= stat.size()) {
    return false;
  }
  std::istringstream iss(stat.substr(right_paren + 2));
  std::string token;
  for (int i = 0; i < 20; ++i) {
    if (!(iss >> token)) return false;
  }
  try {
    out_ticks = static_cast<uint64_t>(std::stoull(token));
    return out_ticks != 0;
  } catch (...) {
    return false;
  }
}

bool pid_is_alive(pid_t pid) {
  if (pid <= 1) return false;
  if (::kill(pid, 0) == 0) return true;
  return errno == EPERM;
}

struct SocketFrame {
  SocketFrameType type = SocketFrameType::SyncRequest;
  std::string key;
  std::string value;
};

void append_net_u32(std::string& out, uint32_t value) {
  uint32_t const net_value = htonl(value);
  out.append(reinterpret_cast<char const*>(&net_value), sizeof(net_value));
}

bool read_net_u32(std::string const& in, size_t& pos, uint32_t& value) {
  if (in.size() < pos + sizeof(uint32_t)) return false;
  uint32_t net_value = 0;
  std::memcpy(&net_value, in.data() + pos, sizeof(net_value));
  value = ntohl(net_value);
  pos += sizeof(uint32_t);
  return true;
}

bool encode_socket_frame(SocketFrame const& frame, std::string& out) {
  out.clear();
  if (frame.type == SocketFrameType::SyncRequest ||
      frame.type == SocketFrameType::SyncDone) {
    append_net_u32(out, 1);
    out.push_back(static_cast<char>(frame.type));
    return true;
  }
  if (frame.type != SocketFrameType::Publish) return false;
  if (frame.key.size() > std::numeric_limits<uint32_t>::max() ||
      frame.value.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  uint32_t const payload_len =
      1 + sizeof(uint32_t) + static_cast<uint32_t>(frame.key.size()) +
      sizeof(uint32_t) + static_cast<uint32_t>(frame.value.size());
  append_net_u32(out, payload_len);
  out.push_back(static_cast<char>(frame.type));
  append_net_u32(out, static_cast<uint32_t>(frame.key.size()));
  out.append(frame.key);
  append_net_u32(out, static_cast<uint32_t>(frame.value.size()));
  out.append(frame.value);
  return true;
}

bool decode_socket_frame(std::string const& payload, SocketFrame& frame) {
  frame = {};
  if (payload.empty()) return false;
  frame.type =
      static_cast<SocketFrameType>(static_cast<uint8_t>(payload.front()));
  if (frame.type == SocketFrameType::SyncRequest ||
      frame.type == SocketFrameType::SyncDone) {
    return payload.size() == 1;
  }
  if (frame.type != SocketFrameType::Publish) return false;

  size_t pos = 1;
  uint32_t key_len = 0;
  uint32_t value_len = 0;
  if (!read_net_u32(payload, pos, key_len)) return false;
  if (payload.size() < pos + key_len) return false;
  frame.key.assign(payload.data() + pos, key_len);
  pos += key_len;
  if (!read_net_u32(payload, pos, value_len)) return false;
  if (payload.size() != pos + value_len) return false;
  frame.value.assign(payload.data() + pos, value_len);
  return true;
}

}  // namespace

uint64_t ShmExchanger::SlotCacheHash::hash_key(std::string_view key) {
  // 64-bit FNV-1a, stable and allocation-free for small key caching.
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : key) {
    h ^= static_cast<uint64_t>(c);
    h *= 1099511628211ULL;
  }
  return h;
}

ShmExchanger::ShmExchanger(std::string kv_namespace, bool create_if_missing,
                           int open_timeout_ms)
    : kv_namespace_(std::move(kv_namespace)),
      create_if_missing_(create_if_missing),
      open_timeout_ms_(open_timeout_ms > 0 ? open_timeout_ms : 30000) {
  (void)init_shared_store();
}

ShmExchanger::~ShmExchanger() { close_shared_store(); }

bool ShmExchanger::valid() const { return shm_ != nullptr; }

bool ShmExchanger::apply_remote(std::string_view key, std::string_view value) {
  return put_encoded(key, value, /*mark_relayed=*/true);
}

bool ShmExchanger::begin_leader_run() {
  if (!lock_store()) return false;
  uint32_t const owner_pid = shm_->owner_pid;
  uint64_t const owner_start_ticks = shm_->owner_start_ticks;
  bool owner_alive = false;
  if (owner_pid > 1 && pid_is_alive(static_cast<pid_t>(owner_pid))) {
    if (owner_start_ticks == 0) {
      owner_alive = true;
    } else {
      uint64_t current_ticks = 0;
      owner_alive = read_process_start_ticks(static_cast<pid_t>(owner_pid),
                                             current_ticks) &&
                    current_ticks == owner_start_ticks;
    }
  }
  if (owner_alive) {
    unlock_store();
    return false;
  }

  for (uint32_t i = 0; i < shm_->slot_capacity; ++i) {
    shm_->slots[i] = KvSlot{};
  }
  shm_->used_slots = 0;
  shm_->first_free_hint = 0;
  shm_->write_seq = 0;
  shm_->init_done = 0;
  shm_->owner_pid = static_cast<uint32_t>(::getpid());
  uint64_t start_ticks = 0;
  (void)read_process_start_ticks(::getpid(), start_ticks);
  shm_->owner_start_ticks = start_ticks;
  next_unrelayed_scan_index_ = 0;
  slot_index_cache_.clear();
  unlock_store();
  maybe_post_sem();
  return true;
}

bool ShmExchanger::mark_run_ready() {
  if (!shm_ || !lock_store()) return false;
  shm_->init_done = 1;
  shm_->owner_pid = static_cast<uint32_t>(::getpid());
  uint64_t start_ticks = 0;
  (void)read_process_start_ticks(::getpid(), start_ticks);
  shm_->owner_start_ticks = start_ticks;
  unlock_store();
  maybe_post_sem();
  return true;
}

bool ShmExchanger::wait_until_ready(int timeout_ms) const {
  if (!shm_) return false;
  int const wait_ms = timeout_ms > 0 ? timeout_ms : kLeaderReadyWaitDefaultMs;
  auto const deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!lock_store()) return false;
    uint32_t const owner_pid = shm_->owner_pid;
    uint64_t const owner_start_ticks = shm_->owner_start_ticks;
    bool const init_done = shm_->init_done != 0;
    unlock_store();

    if (!init_done || owner_pid <= 1 ||
        !pid_is_alive(static_cast<pid_t>(owner_pid))) {
      (void)wait_for_update(10);
      continue;
    }
    if (owner_start_ticks == 0) return true;

    uint64_t current_ticks = 0;
    if (read_process_start_ticks(static_cast<pid_t>(owner_pid),
                                 current_ticks) &&
        current_ticks == owner_start_ticks) {
      return true;
    }
    (void)wait_for_update(10);
  }
  return false;
}

size_t ShmExchanger::collect_unrelayed(
    std::vector<std::pair<std::string, std::string>>& out, size_t max_items) {
  size_t const collected =
      collect_entries(out, max_items, next_unrelayed_scan_index_,
                      /*only_unrelayed=*/true);
  if (shm_ && shm_->slot_capacity > 0) {
    next_unrelayed_scan_index_ =
        (next_unrelayed_scan_index_ + (collected == 0 ? 1 : collected)) %
        shm_->slot_capacity;
  }
  return collected;
}

size_t ShmExchanger::collect_all(
    std::vector<std::pair<std::string, std::string>>& out,
    size_t max_items) const {
  return collect_entries(out, max_items, 0, /*only_unrelayed=*/false);
}

bool ShmExchanger::mark_relayed(std::string_view key) {
  if (!shm_ || !lock_store()) return false;
  int idx = find_slot_locked(key);
  if (idx < 0) {
    unlock_store();
    return false;
  }
  shm_->slots[static_cast<size_t>(idx)].relayed = 1;
  unlock_store();
  return true;
}

bool ShmExchanger::wait_raw(std::string_view key, std::string& value,
                            WaitOptions const& options) {
  if (options.max_retries == 0) {
    return get_raw(key, value);
  }

  int const delay_ms = options.delay_ms > 0 ? options.delay_ms : 1;
  auto try_once = [&]() { return get_raw(key, value); };
  if (try_once()) return true;

  if (options.max_retries > 0) {
    for (int i = 0; i < options.max_retries; ++i) {
      if (!valid()) return false;
      (void)wait_for_update(delay_ms);
      if (try_once()) return true;
    }
    return false;
  }

  while (valid()) {
    (void)wait_for_update(delay_ms);
    if (try_once()) return true;
  }
  return false;
}

bool ShmExchanger::put_raw(std::string_view key, std::string_view value) {
  return put_encoded(key, value, /*mark_relayed=*/false);
}

bool ShmExchanger::get_raw(std::string_view key, std::string& value) {
  value.clear();
  if (!shm_ || !lock_store()) return false;
  int idx = find_slot_locked(key);
  if (idx < 0) {
    unlock_store();
    return false;
  }
  auto const& slot = shm_->slots[static_cast<size_t>(idx)];
  value.assign(slot.value, slot.value + slot.value_len);
  unlock_store();
  return true;
}

bool ShmExchanger::init_shared_store() {
  bool created = false;
  if (create_if_missing_) {
    shm_fd_ =
        ::shm_open(kv_namespace_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (shm_fd_ >= 0) {
      created = true;
    } else if (errno == EEXIST) {
      shm_fd_ = ::shm_open(kv_namespace_.c_str(), O_CREAT | O_RDWR, 0600);
    }
  } else {
    auto const deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(open_timeout_ms_);
    while (std::chrono::steady_clock::now() < deadline) {
      shm_fd_ = ::shm_open(kv_namespace_.c_str(), O_RDWR, 0600);
      if (shm_fd_ >= 0) break;
      if (errno != ENOENT) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  if (shm_fd_ < 0) return false;

  if (::flock(shm_fd_, LOCK_EX) != 0) {
    ::close(shm_fd_);
    shm_fd_ = -1;
    return false;
  }

  if (created &&
      ::ftruncate(shm_fd_, static_cast<off_t>(sizeof(KvShmHeader))) != 0) {
    (void)::flock(shm_fd_, LOCK_UN);
    ::close(shm_fd_);
    shm_fd_ = -1;
    return false;
  }

  void* mapped = ::mmap(nullptr, sizeof(KvShmHeader), PROT_READ | PROT_WRITE,
                        MAP_SHARED, shm_fd_, 0);
  if (mapped == MAP_FAILED) {
    (void)::flock(shm_fd_, LOCK_UN);
    ::close(shm_fd_);
    shm_fd_ = -1;
    return false;
  }
  shm_ = reinterpret_cast<KvShmHeader*>(mapped);

  bool need_init =
      created || shm_->magic != kShmMagic || shm_->slot_capacity != kMaxSlots;
  if (need_init) {
    shm_->magic = kShmMagic;
    shm_->slot_capacity = kMaxSlots;
    shm_->write_seq = 0;
    shm_->owner_pid = 0;
    shm_->init_done = 0;
    shm_->owner_start_ticks = 0;
    shm_->used_slots = 0;
    shm_->first_free_hint = 0;
    shm_->reserved0 = 0;
    for (uint32_t i = 0; i < kMaxSlots; ++i) {
      shm_->slots[i] = KvSlot{};
    }

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
#ifdef PTHREAD_MUTEX_ROBUST
    pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
#endif
    pthread_mutex_init(&shm_->mu, &attr);
    pthread_mutexattr_destroy(&attr);
    sem_init(&shm_->notify_sem, 1, 0);
  } else {
    if (!lock_store()) {
      (void)::flock(shm_fd_, LOCK_UN);
      ::munmap(shm_, sizeof(KvShmHeader));
      shm_ = nullptr;
      ::close(shm_fd_);
      shm_fd_ = -1;
      return false;
    }
    rebuild_metadata_locked();
    unlock_store();
  }

  (void)::flock(shm_fd_, LOCK_UN);
  return true;
}

void ShmExchanger::close_shared_store() {
  slot_index_cache_.clear();
  if (shm_) {
    ::munmap(shm_, sizeof(KvShmHeader));
    shm_ = nullptr;
  }
  if (shm_fd_ >= 0) {
    ::close(shm_fd_);
    shm_fd_ = -1;
  }
}

bool ShmExchanger::put_encoded(std::string_view key, std::string_view value,
                               bool mark_relayed) {
  if (!shm_) return false;
  if (key.empty() || key.size() >= kMaxKeyBytes) return false;
  if (value.size() >= kMaxValueBytes) return false;
  if (!lock_store()) return false;

  int slot_idx = find_slot_locked(key);
  if (slot_idx < 0) {
    if (shm_->used_slots >= shm_->slot_capacity) {
      unlock_store();
      return false;
    }
    slot_idx = find_free_slot_locked();
  }
  if (slot_idx < 0) {
    unlock_store();
    return false;
  }

  auto& slot = shm_->slots[static_cast<size_t>(slot_idx)];
  bool const is_new_slot = slot.state != kSlotUsed;
  slot.state = kSlotUsed;
  slot.relayed = mark_relayed ? 1 : 0;
  slot.write_seq = ++shm_->write_seq;
  slot.key_len = static_cast<uint32_t>(key.size());
  slot.value_len = static_cast<uint32_t>(value.size());
  std::memcpy(slot.key, key.data(), key.size());
  std::memcpy(slot.value, value.data(), value.size());
  if (is_new_slot) {
    shm_->used_slots += 1;
  }
  if (static_cast<uint32_t>(slot_idx) == shm_->first_free_hint) {
    shm_->first_free_hint =
        static_cast<uint32_t>((slot_idx + 1) % shm_->slot_capacity);
  }
  slot_index_cache_[SlotCacheHash::hash_key(key)] =
      static_cast<uint32_t>(slot_idx);
  unlock_store();
  maybe_post_sem();
  return true;
}

size_t ShmExchanger::collect_entries(
    std::vector<std::pair<std::string, std::string>>& out, size_t max_items,
    size_t start_index, bool only_unrelayed) const {
  out.clear();
  if (!shm_ || max_items == 0) return 0;
  if (!lock_store()) return 0;

  size_t const slot_count = shm_->slot_capacity;
  if (slot_count == 0) {
    unlock_store();
    return 0;
  }

  size_t const begin = start_index % slot_count;
  for (size_t visited = 0; visited < slot_count && out.size() < max_items;
       ++visited) {
    size_t const index = (begin + visited) % slot_count;
    auto const& slot = shm_->slots[index];
    if (slot.state != kSlotUsed) continue;
    if (only_unrelayed && slot.relayed) continue;
    out.emplace_back(std::string(slot.key, slot.key + slot.key_len),
                     std::string(slot.value, slot.value + slot.value_len));
  }
  unlock_store();
  return out.size();
}

bool ShmExchanger::wait_for_update(int delay_ms) const {
  if (!shm_) return false;
  if (delay_ms <= 0) delay_ms = 1;

  auto deadline =
      std::chrono::system_clock::now() + std::chrono::milliseconds(delay_ms);
  auto sec = std::chrono::time_point_cast<std::chrono::seconds>(deadline);
  auto nsec =
      std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - sec)
          .count();
  struct timespec ts {};
  ts.tv_sec = static_cast<time_t>(sec.time_since_epoch().count());
  ts.tv_nsec = static_cast<long>(nsec);

  while (true) {
    if (sem_timedwait(&shm_->notify_sem, &ts) == 0) return true;
    if (errno == EINTR) continue;
    if (errno == ETIMEDOUT) return false;
    return false;
  }
}

int ShmExchanger::find_slot_locked(std::string_view key) const {
  if (!shm_) return -1;
  uint64_t const key_hash = SlotCacheHash::hash_key(key);
  auto const cache_it = slot_index_cache_.find(key_hash);
  if (cache_it != slot_index_cache_.end()) {
    uint32_t const idx = cache_it->second;
    if (idx < shm_->slot_capacity) {
      auto const& slot = shm_->slots[idx];
      if (slot.state == kSlotUsed && slot.key_len == key.size() &&
          std::memcmp(slot.key, key.data(), key.size()) == 0) {
        return static_cast<int>(idx);
      }
    }
    slot_index_cache_.erase(cache_it);
  }
  for (uint32_t i = 0; i < shm_->slot_capacity; ++i) {
    auto const& slot = shm_->slots[i];
    if (slot.state != kSlotUsed || slot.key_len != key.size()) continue;
    if (std::memcmp(slot.key, key.data(), key.size()) == 0) {
      slot_index_cache_[key_hash] = i;
      return static_cast<int>(i);
    }
  }
  return -1;
}

int ShmExchanger::find_free_slot_locked() const {
  if (!shm_) return -1;
  for (uint32_t offset = 0; offset < shm_->slot_capacity; ++offset) {
    uint32_t const index =
        (shm_->first_free_hint + offset) % shm_->slot_capacity;
    if (shm_->slots[index].state != kSlotUsed) return static_cast<int>(index);
  }
  return -1;
}

void ShmExchanger::rebuild_metadata_locked() {
  if (!shm_) return;
  slot_index_cache_.clear();
  shm_->used_slots = 0;
  shm_->first_free_hint = 0;
  uint64_t max_write_seq = 0;
  bool found_free_slot = false;
  for (uint32_t i = 0; i < shm_->slot_capacity; ++i) {
    auto const& slot = shm_->slots[i];
    if (slot.state != kSlotUsed) {
      if (!found_free_slot) {
        shm_->first_free_hint = i;
        found_free_slot = true;
      }
      continue;
    }
    shm_->used_slots += 1;
    slot_index_cache_[SlotCacheHash::hash_key(
        std::string_view(slot.key, slot.key_len))] = i;
    if (slot.write_seq > max_write_seq) max_write_seq = slot.write_seq;
  }
  if (!found_free_slot) {
    shm_->first_free_hint = 0;
  }
  shm_->write_seq = std::max(shm_->write_seq, max_write_seq);
}

bool ShmExchanger::lock_store() const {
  if (!shm_) return false;
  int rc = pthread_mutex_lock(&shm_->mu);
  if (rc == 0) return true;
#ifdef EOWNERDEAD
  if (rc == EOWNERDEAD) {
    (void)pthread_mutex_consistent(&shm_->mu);
    return true;
  }
#endif
  return false;
}

void ShmExchanger::unlock_store() const {
  if (!shm_) return;
  (void)pthread_mutex_unlock(&shm_->mu);
}

void ShmExchanger::maybe_post_sem() const {
  if (!shm_) return;
  (void)sem_post(&shm_->notify_sem);
}

SocketExchanger::SocketExchanger(bool is_root, std::string host, int port,
                                 int timeout_ms, size_t max_line_bytes,
                                 ReceiveCallback on_receive)
    : host_(std::move(host)),
      port_(port),
      timeout_ms_(timeout_ms),
      is_root_(is_root),
      max_line_bytes_(max_line_bytes),
      on_receive_(std::move(on_receive)) {}

SocketExchanger::~SocketExchanger() {
  running_.store(false, std::memory_order_release);
  ready_cv_.notify_all();
  pending_cv_.notify_all();
  (void)notify_root_wakeup();
  if (root_.listen_fd >= 0) {
    ::shutdown(root_.listen_fd, SHUT_RDWR);
  }
  {
    std::lock_guard<std::mutex> lk(upstream_mu_);
    if (upstream_fd_ >= 0) {
      ::shutdown(upstream_fd_, SHUT_RDWR);
    }
  }
  if (upstream_thread_.joinable()) upstream_thread_.join();
  if (root_.thread.joinable()) root_.thread.join();
  stop_root_server();

  {
    std::lock_guard<std::mutex> lk(upstream_mu_);
    close_upstream_locked();
  }

  {
    std::lock_guard<std::mutex> lk(root_.peers_mu);
    for (auto const& [fd, _] : root_.peers) {
      (void)_;
      ::shutdown(fd, SHUT_RDWR);
    }
  }
}

bool SocketExchanger::valid() const {
  return started_.load(std::memory_order_acquire) &&
         running_.load(std::memory_order_acquire);
}

bool SocketExchanger::start() {
  if (started_.exchange(true, std::memory_order_acq_rel)) return valid();
  running_.store(true, std::memory_order_release);
  if (is_root_) {
    if (!start_root_server()) {
      running_.store(false, std::memory_order_release);
      ready_.store(false, std::memory_order_release);
      return false;
    }
    ready_.store(true, std::memory_order_release);
    connection_epoch_.fetch_add(1, std::memory_order_acq_rel);
    ready_cv_.notify_all();
    return true;
  }
  upstream_thread_ = std::thread(&SocketExchanger::upstream_loop, this);
  if (wait_until_ready(timeout_ms_)) return true;

  running_.store(false, std::memory_order_release);
  ready_cv_.notify_all();
  pending_cv_.notify_all();
  {
    std::lock_guard<std::mutex> lk(upstream_mu_);
    if (upstream_fd_ >= 0) {
      ::shutdown(upstream_fd_, SHUT_RDWR);
    }
  }
  if (upstream_thread_.joinable()) upstream_thread_.join();
  {
    std::lock_guard<std::mutex> lk(upstream_mu_);
    close_upstream_locked();
  }
  started_.store(false, std::memory_order_release);
  return false;
}

uint64_t SocketExchanger::connection_epoch() const {
  return connection_epoch_.load(std::memory_order_acquire);
}

bool SocketExchanger::put_raw(std::string_view key, std::string_view value) {
  if (!running_.load(std::memory_order_acquire)) return false;
  if (is_root_ && !ready_.load(std::memory_order_acquire)) return false;

  std::string key_copy(key);
  std::string value_copy(value);
  {
    std::lock_guard<std::mutex> lk(entries_mu_);
    entries_[key_copy] = value_copy;
  }
  if (is_root_) {
    root_broadcast_pub(key_copy, value_copy, -1);
    return true;
  }
  {
    std::lock_guard<std::mutex> lk(pending_mu_);
    pending_pubs_.emplace_back(std::move(key_copy), std::move(value_copy));
  }
  pending_cv_.notify_all();
  return running_.load(std::memory_order_acquire);
}

bool SocketExchanger::get_raw(std::string_view key, std::string& value) {
  std::lock_guard<std::mutex> lk(entries_mu_);
  auto it = entries_.find(std::string(key));
  if (it == entries_.end()) return false;
  value = it->second;
  return true;
}

bool SocketExchanger::start_root_server() {
  root_.listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (root_.listen_fd < 0) return false;

  int opt = 1;
  (void)::setsockopt(root_.listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt,
                     sizeof(opt));
  (void)::fcntl(root_.listen_fd, F_SETFL, O_NONBLOCK);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(port_));
  if (::bind(root_.listen_fd, reinterpret_cast<sockaddr*>(&addr),
             sizeof(addr)) != 0) {
    ::close(root_.listen_fd);
    root_.listen_fd = -1;
    return false;
  }
  if (::listen(root_.listen_fd, 64) != 0) {
    ::close(root_.listen_fd);
    root_.listen_fd = -1;
    return false;
  }
  int wake_pipe[2] = {-1, -1};
  if (::pipe(wake_pipe) != 0) {
    ::close(root_.listen_fd);
    root_.listen_fd = -1;
    return false;
  }
  int const wake_read_flags = ::fcntl(wake_pipe[0], F_GETFL, 0);
  int const wake_write_flags = ::fcntl(wake_pipe[1], F_GETFL, 0);
  if (wake_read_flags < 0 || wake_write_flags < 0 ||
      ::fcntl(wake_pipe[0], F_SETFL, wake_read_flags | O_NONBLOCK) != 0 ||
      ::fcntl(wake_pipe[1], F_SETFL, wake_write_flags | O_NONBLOCK) != 0) {
    ::close(wake_pipe[0]);
    ::close(wake_pipe[1]);
    ::close(root_.listen_fd);
    root_.listen_fd = -1;
    return false;
  }
  root_.wakeup_read_fd = wake_pipe[0];
  root_.wakeup_write_fd = wake_pipe[1];
  try {
    root_.thread = std::thread(&SocketExchanger::root_accept_loop, this);
  } catch (...) {
    stop_root_server();
    return false;
  }
  return true;
}

void SocketExchanger::stop_root_server() {
  if (root_.wakeup_write_fd >= 0) {
    ::close(root_.wakeup_write_fd);
    root_.wakeup_write_fd = -1;
  }
  if (root_.wakeup_read_fd >= 0) {
    ::close(root_.wakeup_read_fd);
    root_.wakeup_read_fd = -1;
  }
  if (root_.listen_fd >= 0) {
    ::shutdown(root_.listen_fd, SHUT_RDWR);
    ::close(root_.listen_fd);
    root_.listen_fd = -1;
  }
}

void SocketExchanger::root_accept_loop() {
  int epoll_fd = ::epoll_create1(EPOLL_CLOEXEC);
  if (epoll_fd < 0) return;

  auto set_peer_events = [&](int fd, bool want_write) -> bool {
    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLRDHUP | (want_write ? EPOLLOUT : 0);
    ev.data.fd = fd;
    return ::epoll_ctl(epoll_fd, EPOLL_CTL_MOD, fd, &ev) == 0;
  };

  auto add_epoll_in = [&](int fd) -> bool {
    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLRDHUP;
    ev.data.fd = fd;
    return ::epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev) == 0;
  };

  auto sync_peer_interest = [&](std::shared_ptr<PeerConn> const& peer) -> bool {
    bool want_write = false;
    {
      std::lock_guard<std::mutex> send_lk(peer->send_mu);
      want_write = !peer->write_buffer.empty();
    }
    return set_peer_events(peer->fd, want_write);
  };

  if (root_.listen_fd < 0 || root_.wakeup_read_fd < 0 ||
      !add_epoll_in(root_.listen_fd) || !add_epoll_in(root_.wakeup_read_fd)) {
    ::close(epoll_fd);
    return;
  }

  constexpr int kMaxEvents = 64;
  epoll_event events[kMaxEvents];
  while (running_.load(std::memory_order_acquire)) {
    int const ready_count = ::epoll_wait(epoll_fd, events, kMaxEvents, -1);
    if (ready_count < 0) {
      if (errno == EINTR) continue;
      break;
    }
    for (int i = 0; i < ready_count; ++i) {
      int const fd = events[i].data.fd;
      uint32_t const ev = events[i].events;
      if (fd == root_.listen_fd) {
        while (running_.load(std::memory_order_acquire)) {
          sockaddr_in cli{};
          socklen_t len = sizeof(cli);
          int conn_fd = ::accept(root_.listen_fd,
                                 reinterpret_cast<sockaddr*>(&cli), &len);
          if (conn_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
              break;
            }
            break;
          }
          int const current_flags = ::fcntl(conn_fd, F_GETFL, 0);
          if (current_flags < 0 ||
              ::fcntl(conn_fd, F_SETFL, current_flags | O_NONBLOCK) != 0) {
            ::shutdown(conn_fd, SHUT_RDWR);
            ::close(conn_fd);
            continue;
          }
          auto peer = std::make_shared<PeerConn>();
          peer->fd = conn_fd;
          {
            std::lock_guard<std::mutex> lk(root_.peers_mu);
            root_.peers[conn_fd] = peer;
          }
          if (!add_epoll_in(conn_fd)) {
            close_root_peer(conn_fd);
            continue;
          }
        }
        continue;
      }
      if (fd == root_.wakeup_read_fd) {
        char buf[128];
        while (::read(root_.wakeup_read_fd, buf, sizeof(buf)) > 0) {
        }
        std::vector<int> pending_interest_fds;
        drain_root_peer_write_interest(pending_interest_fds);
        for (int pending_fd : pending_interest_fds) {
          std::shared_ptr<PeerConn> pending_peer;
          {
            std::lock_guard<std::mutex> lk(root_.peers_mu);
            auto it = root_.peers.find(pending_fd);
            if (it != root_.peers.end()) pending_peer = it->second;
          }
          if (!pending_peer) continue;
          if (!sync_peer_interest(pending_peer)) {
            close_root_peer(pending_fd);
          }
        }
        continue;
      }

      if (ev & (EPOLLERR | EPOLLHUP | EPOLLRDHUP)) {
        close_root_peer(fd);
        continue;
      }
      if (!(ev & (EPOLLIN | EPOLLOUT))) continue;

      std::shared_ptr<PeerConn> peer;
      {
        std::lock_guard<std::mutex> lk(root_.peers_mu);
        auto it = root_.peers.find(fd);
        if (it != root_.peers.end()) peer = it->second;
      }
      if (!peer) continue;

      bool close_peer_now = false;
      if (ev & EPOLLOUT) {
        std::lock_guard<std::mutex> send_lk(peer->send_mu);
        if (!peer->write_buffer.empty() &&
            !flush_write_buffer_locked(fd, peer->write_buffer,
                                       peer->write_offset) &&
            errno != EAGAIN && errno != EWOULDBLOCK) {
          close_peer_now = true;
        }
      }
      if (close_peer_now) {
        close_root_peer(fd);
        continue;
      }

      for (;;) {
        std::string frame;
        bool const got_frame = recv_frame_buffered(fd, peer->read_buffer,
                                                   peer->read_offset, frame, 0);
        if (!got_frame) {
          if (errno == EAGAIN || errno == EWOULDBLOCK) break;
          close_peer_now = true;
          break;
        }
        SocketFrame decoded;
        if (!decode_socket_frame(frame, decoded)) continue;
        if (decoded.type == SocketFrameType::SyncRequest) {
          std::vector<std::pair<std::string, std::string>> snapshot;
          snapshot_entries(snapshot);
          bool snapshot_ok = true;
          {
            std::lock_guard<std::mutex> send_lk(peer->send_mu);
            for (auto const& entry : snapshot) {
              std::string publish_frame;
              if (!encode_socket_frame(SocketFrame{SocketFrameType::Publish,
                                                   entry.first, entry.second},
                                       publish_frame) ||
                  !enqueue_frame_locked(peer->write_buffer, publish_frame)) {
                snapshot_ok = false;
                break;
              }
            }
            if (snapshot_ok) {
              std::string sync_done_frame;
              snapshot_ok =
                  encode_socket_frame(
                      SocketFrame{SocketFrameType::SyncDone, {}, {}},
                      sync_done_frame) &&
                  enqueue_frame_locked(peer->write_buffer, sync_done_frame);
            }
          }
          if (!snapshot_ok) {
            close_peer_now = true;
            break;
          }
          if (!sync_peer_interest(peer)) {
            close_peer_now = true;
            break;
          }
          continue;
        }
        if (decoded.type == SocketFrameType::Publish) {
          handle_publish(decoded.key, decoded.value,
                         /*broadcast_from_root=*/true, fd);
        }
      }
      if (close_peer_now) {
        close_root_peer(fd);
      } else if (!sync_peer_interest(peer)) {
        close_root_peer(fd);
      }
    }
  }

  std::vector<int> stale_fds;
  {
    std::lock_guard<std::mutex> lk(root_.peers_mu);
    stale_fds.reserve(root_.peers.size());
    for (auto const& [fd, _] : root_.peers) stale_fds.push_back(fd);
  }
  for (int fd : stale_fds) {
    close_root_peer(fd);
  }
  ::close(epoll_fd);
}

void SocketExchanger::root_broadcast_pub(std::string const& key,
                                         std::string const& value,
                                         int exclude_fd) {
  std::vector<std::shared_ptr<PeerConn>> peers;
  std::vector<int> stale_fds;
  std::vector<int> activate_write_fds;
  {
    std::lock_guard<std::mutex> lk(root_.peers_mu);
    peers.reserve(root_.peers.size());
    for (auto const& [fd, peer] : root_.peers) {
      if (fd == exclude_fd) continue;
      peers.push_back(peer);
    }
  }

  std::string frame;
  if (!encode_socket_frame(SocketFrame{SocketFrameType::Publish, key, value},
                           frame)) {
    return;
  }
  for (auto const& peer : peers) {
    std::lock_guard<std::mutex> lk(peer->send_mu);
    bool const was_empty = peer->write_buffer.empty();
    if (!enqueue_frame_locked(peer->write_buffer, frame)) {
      stale_fds.push_back(peer->fd);
      continue;
    }
    if (was_empty && !peer->write_buffer.empty()) {
      activate_write_fds.push_back(peer->fd);
    }
  }
  for (int fd : activate_write_fds) {
    queue_root_peer_write_interest(fd);
  }

  if (!stale_fds.empty()) {
    for (int fd : stale_fds) {
      close_root_peer(fd);
    }
  }
  (void)notify_root_wakeup();
}

void SocketExchanger::close_root_peer(int fd) {
  std::shared_ptr<PeerConn> peer;
  {
    std::lock_guard<std::mutex> lk(root_.peers_mu);
    auto it = root_.peers.find(fd);
    if (it == root_.peers.end()) return;
    peer = it->second;
    root_.peers.erase(it);
  }
  if (peer) {
    std::lock_guard<std::mutex> send_lk(peer->send_mu);
    peer->write_buffer.clear();
    peer->write_offset = 0;
    peer->read_buffer.clear();
    peer->read_offset = 0;
  }
  {
    std::lock_guard<std::mutex> lk(root_.pending_mu);
    root_.pending_write_interest_set.erase(fd);
  }
  ::shutdown(fd, SHUT_RDWR);
  ::close(fd);
}

void SocketExchanger::queue_root_peer_write_interest(int fd) {
  std::lock_guard<std::mutex> lk(root_.pending_mu);
  auto const [_, inserted] = root_.pending_write_interest_set.insert(fd);
  if (!inserted) return;
  root_.pending_write_interest.push_back(fd);
}

void SocketExchanger::drain_root_peer_write_interest(std::vector<int>& fds) {
  fds.clear();
  std::lock_guard<std::mutex> lk(root_.pending_mu);
  fds.reserve(root_.pending_write_interest.size());
  while (!root_.pending_write_interest.empty()) {
    int const fd = root_.pending_write_interest.front();
    root_.pending_write_interest.pop_front();
    root_.pending_write_interest_set.erase(fd);
    fds.push_back(fd);
  }
}

bool SocketExchanger::notify_root_wakeup() const {
  if (root_.wakeup_write_fd < 0) return false;
  char one = 1;
  ssize_t n = ::write(root_.wakeup_write_fd, &one, sizeof(one));
  return n == static_cast<ssize_t>(sizeof(one)) || errno == EAGAIN ||
         errno == EWOULDBLOCK;
}

bool SocketExchanger::enqueue_frame_locked(std::string& write_buffer,
                                           std::string const& frame) const {
  if (frame.size() > max_line_bytes_) {
    errno = EMSGSIZE;
    return false;
  }
  if (write_buffer.size() + frame.size() > max_line_bytes_ * 4) {
    errno = ENOBUFS;
    return false;
  }
  write_buffer.append(frame);
  return true;
}

bool SocketExchanger::flush_write_buffer_locked(int fd,
                                                std::string& write_buffer,
                                                size_t& write_offset) const {
  int send_flags = 0;
#ifdef MSG_NOSIGNAL
  send_flags |= MSG_NOSIGNAL;
#endif
  while (write_offset < write_buffer.size()) {
    char const* data = write_buffer.data() + write_offset;
    size_t const pending = write_buffer.size() - write_offset;
    ssize_t n = ::send(fd, data, pending, send_flags);
    if (n > 0) {
      write_offset += static_cast<size_t>(n);
      compact_prefixed_buffer(write_buffer, write_offset);
      continue;
    }
    if (n == 0) {
      errno = ECONNRESET;
      return false;
    }
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) return false;
    return false;
  }
  write_buffer.clear();
  write_offset = 0;
  return true;
}

bool SocketExchanger::recv_frame_buffered(int fd, std::string& buffer,
                                          size_t& read_offset,
                                          std::string& frame,
                                          int timeout_ms) const {
  auto extract_frame = [&]() -> bool {
    if (read_offset > buffer.size()) {
      errno = EPROTO;
      return false;
    }
    size_t const available = buffer.size() - read_offset;
    if (available < sizeof(uint32_t)) return false;
    size_t pos = read_offset;
    uint32_t payload_len = 0;
    if (!read_net_u32(buffer, pos, payload_len)) return false;
    if (payload_len > max_line_bytes_) {
      errno = EMSGSIZE;
      return false;
    }
    if (available < sizeof(uint32_t) + payload_len) return false;
    frame.assign(buffer.data() + read_offset + sizeof(uint32_t), payload_len);
    read_offset += sizeof(uint32_t) + payload_len;
    compact_prefixed_buffer(buffer, read_offset);
    return true;
  };

  if (extract_frame()) return true;

  while (running_.load(std::memory_order_acquire)) {
    if (timeout_ms > 0) {
      pollfd pfd{};
      pfd.fd = fd;
      pfd.events = POLLIN;
      int rc = ::poll(&pfd, 1, timeout_ms);
      if (rc == 0) {
        errno = EAGAIN;
        return false;
      }
      if (rc < 0) {
        if (errno == EINTR) continue;
        return false;
      }
      if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
        errno = ECONNRESET;
        return false;
      }
      if (!(pfd.revents & POLLIN)) {
        errno = EAGAIN;
        return false;
      }
    }

    char chunk[4096];
    ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
    if (n == 0) {
      errno = ECONNRESET;
      return false;
    }
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    compact_prefixed_buffer(buffer, read_offset);
    buffer.append(chunk, chunk + n);
    if (buffer.size() - read_offset > max_line_bytes_ * 2) {
      errno = EMSGSIZE;
      return false;
    }
    if (extract_frame()) return true;
    if (timeout_ms == 0) {
      errno = EAGAIN;
      return false;
    }
  }
  return false;
}

bool SocketExchanger::connect_upstream_locked() {
  if (upstream_fd_ >= 0) return true;

  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return false;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port_));
  if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) <= 0) {
    ::close(fd);
    return false;
  }
  if (!connect_with_timeout(fd, addr, timeout_ms_)) {
    ::close(fd);
    return false;
  }

  upstream_fd_ = fd;
  upstream_read_buffer_.clear();
  upstream_read_offset_ = 0;
  upstream_write_buffer_.clear();
  upstream_write_offset_ = 0;
  upstream_sync_complete_.store(false, std::memory_order_release);
  std::string sync_request_frame;
  if (!encode_socket_frame(SocketFrame{SocketFrameType::SyncRequest, {}, {}},
                           sync_request_frame) ||
      !enqueue_frame_locked(upstream_write_buffer_, sync_request_frame)) {
    close_upstream_locked();
    return false;
  }
  if (!flush_write_buffer_locked(upstream_fd_, upstream_write_buffer_,
                                 upstream_write_offset_) &&
      errno != EAGAIN && errno != EWOULDBLOCK) {
    close_upstream_locked();
    return false;
  }
  return true;
}

void SocketExchanger::close_upstream_locked() {
  if (upstream_fd_ >= 0) {
    ::shutdown(upstream_fd_, SHUT_RDWR);
    ::close(upstream_fd_);
    upstream_fd_ = -1;
  }
  upstream_read_buffer_.clear();
  upstream_read_offset_ = 0;
  upstream_write_buffer_.clear();
  upstream_write_offset_ = 0;
  upstream_sync_complete_.store(false, std::memory_order_release);
  ready_.store(false, std::memory_order_release);
  ready_cv_.notify_all();
}

void SocketExchanger::upstream_loop() {
  while (running_.load(std::memory_order_acquire)) {
    {
      std::lock_guard<std::mutex> lk(upstream_mu_);
      if (upstream_fd_ < 0 && !connect_upstream_locked()) {
        std::unique_lock<std::mutex> pending_lk(pending_mu_);
        pending_cv_.wait_for(pending_lk,
                             std::chrono::milliseconds(kSocketPollMs));
        continue;
      }
    }

    for (;;) {
      bool queued_any = false;
      {
        std::lock_guard<std::mutex> lk(pending_mu_);
        while (!pending_pubs_.empty()) {
          std::string frame;
          auto const& pending = pending_pubs_.front();
          if (!encode_socket_frame(SocketFrame{SocketFrameType::Publish,
                                               pending.first, pending.second},
                                   frame)) {
            pending_pubs_.pop_front();
            continue;
          }
          std::lock_guard<std::mutex> upstream_lk(upstream_mu_);
          if (!enqueue_frame_locked(upstream_write_buffer_, frame)) {
            if (errno == EMSGSIZE) {
              // Oversized payload can never be sent; drop it to avoid
              // head-of-line blocking in pending_pubs_.
              pending_pubs_.pop_front();
              continue;
            }
            break;
          }
          pending_pubs_.pop_front();
          queued_any = true;
        }
      }
      if (!queued_any) break;
    }

    std::string frame;
    bool progressed = false;
    {
      std::lock_guard<std::mutex> lk(upstream_mu_);
      if (upstream_fd_ >= 0) {
        bool const got_buffered_frame =
            recv_frame_buffered(upstream_fd_, upstream_read_buffer_,
                                upstream_read_offset_, frame, 0);
        if (got_buffered_frame) {
          progressed = true;
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
          close_upstream_locked();
          continue;
        }

        pollfd pfd{};
        pfd.fd = upstream_fd_;
        pfd.events = POLLIN;
        if (!upstream_write_buffer_.empty()) pfd.events |= POLLOUT;
        if (!progressed) {
          int rc = ::poll(&pfd, 1, kSocketPollMs);
          if (rc < 0) {
            if (errno == EINTR) continue;
            close_upstream_locked();
            continue;
          }
          if (rc == 0) continue;
          if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            close_upstream_locked();
            continue;
          }
          if (pfd.revents & POLLOUT) {
            if (!flush_write_buffer_locked(upstream_fd_, upstream_write_buffer_,
                                           upstream_write_offset_) &&
                errno != EAGAIN && errno != EWOULDBLOCK) {
              close_upstream_locked();
              continue;
            }
            progressed = true;
          }
          if (pfd.revents & POLLIN) {
            bool const got_frame =
                recv_frame_buffered(upstream_fd_, upstream_read_buffer_,
                                    upstream_read_offset_, frame, 0);
            if (!got_frame && errno != EAGAIN && errno != EWOULDBLOCK) {
              close_upstream_locked();
              continue;
            }
            progressed = got_frame || progressed;
          }
        }
      }
    }

    if (!progressed || frame.empty()) {
      continue;
    }

    SocketFrame decoded;
    if (!decode_socket_frame(frame, decoded)) continue;
    if (decoded.type == SocketFrameType::SyncDone) {
      if (!upstream_sync_complete_.load(std::memory_order_acquire)) {
        upstream_sync_complete_.store(true, std::memory_order_release);
        ready_.store(true, std::memory_order_release);
        connection_epoch_.fetch_add(1, std::memory_order_acq_rel);
        ready_cv_.notify_all();
      }
      continue;
    }
    if (decoded.type == SocketFrameType::Publish) {
      handle_publish(decoded.key, decoded.value, /*broadcast_from_root=*/false,
                     -1);
    }
  }
}

void SocketExchanger::handle_publish(std::string const& key,
                                     std::string const& value,
                                     bool broadcast_from_root, int exclude_fd) {
  {
    std::lock_guard<std::mutex> lk(entries_mu_);
    entries_[key] = value;
  }
  if (on_receive_) on_receive_(key, value);
  if (is_root_ && broadcast_from_root) {
    root_broadcast_pub(key, value, exclude_fd);
  }
}

bool SocketExchanger::wait_until_ready(int timeout_ms) {
  if (ready_.load(std::memory_order_acquire)) return true;
  std::unique_lock<std::mutex> lk(ready_mu_);
  if (timeout_ms <= 0) {
    ready_cv_.wait(lk, [this] {
      return !running_.load(std::memory_order_acquire) ||
             ready_.load(std::memory_order_acquire);
    });
  } else {
    ready_cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this] {
      return !running_.load(std::memory_order_acquire) ||
             ready_.load(std::memory_order_acquire);
    });
  }
  return ready_.load(std::memory_order_acquire);
}

void SocketExchanger::snapshot_entries(
    std::vector<std::pair<std::string, std::string>>& out) {
  out.clear();
  std::lock_guard<std::mutex> lk(entries_mu_);
  out.reserve(entries_.size());
  for (auto const& [key, value] : entries_) {
    out.emplace_back(key, value);
  }
}

HierarchicalExchanger::HierarchicalExchanger(bool is_server,
                                             std::string const& host, int port,
                                             int timeout_ms,
                                             size_t max_line_bytes,
                                             int local_id)
    : host_(host),
      port_(port),
      is_server_(is_server),
      local_id_(local_id),
      node_leader_(false) {
  if (local_id_ < 0) {
    local_id_ = env_int_or_default("UHM_LOCAL_ID",
                                   env_int_or_default("LOCAL_RANK", -1));
  }
  node_leader_ = is_node_leader();

  std::string const ns = kv_namespace();
  if (node_leader_) {
    shm_ = std::make_unique<ShmExchanger>(ns, /*create_if_missing=*/true);
    if (!shm_ || !shm_->valid()) return;
    if (!shm_->begin_leader_run()) return;
    running_.store(true, std::memory_order_release);
    socket_ = std::make_unique<SocketExchanger>(
        is_server_, host_, port_, timeout_ms, max_line_bytes,
        [this](std::string const& key, std::string const& value) {
          apply_remote_entry(key, value);
        });
    if (!socket_->start()) {
      running_.store(false, std::memory_order_release);
      return;
    }
    if (!shm_->mark_run_ready()) {
      running_.store(false, std::memory_order_release);
      return;
    }
    last_replayed_epoch_ = 0;
    relay_thread_ = std::thread(&HierarchicalExchanger::relay_loop, this);
  } else {
    shm_ = std::make_unique<ShmExchanger>(ns, /*create_if_missing=*/false,
                                          timeout_ms);
    if (!shm_ || !shm_->valid()) return;
    int const wait_ms =
        env_int_or_default("UHM_OOB_LEADER_READY_TIMEOUT_MS", timeout_ms);
    if (!shm_->wait_until_ready(wait_ms)) return;
    running_.store(true, std::memory_order_release);
  }
}

HierarchicalExchanger::~HierarchicalExchanger() {
  running_.store(false, std::memory_order_release);
  if (relay_thread_.joinable()) relay_thread_.join();
  if (node_leader_) {
    (void)::shm_unlink(kv_namespace().c_str());
  }
}

bool HierarchicalExchanger::valid() const {
  if (!running_.load(std::memory_order_acquire) || !shm_ || !shm_->valid()) {
    return false;
  }
  if (node_leader_ && (!socket_ || !socket_->valid())) return false;
  return true;
}

bool HierarchicalExchanger::wait_raw(std::string_view key, std::string& value,
                                     WaitOptions const& options) {
  if (!shm_) return false;
  return shm_->wait_raw(key, value, options);
}

bool HierarchicalExchanger::put_raw(std::string_view key,
                                    std::string_view value) {
  if (!shm_) return false;
  if (!shm_->put_raw(key, value)) return false;
  if (node_leader_ && socket_) {
    if (socket_->put_raw(key, value)) {
      (void)shm_->mark_relayed(key);
    }
  }
  return true;
}

bool HierarchicalExchanger::get_raw(std::string_view key, std::string& value) {
  if (!shm_) return false;
  return shm_->get_raw(key, value);
}

bool HierarchicalExchanger::is_node_leader() const {
  if (local_id_ >= 0) return local_id_ == 0;
  return is_server_;
}

void HierarchicalExchanger::relay_loop() {
  std::vector<std::pair<std::string, std::string>> pending;
  while (running_.load(std::memory_order_acquire)) {
    if (!socket_ || !shm_) break;
    uint64_t const socket_epoch = socket_->connection_epoch();
    if (socket_epoch != 0 && socket_epoch != last_replayed_epoch_) {
      replay_local_state();
      last_replayed_epoch_ = socket_epoch;
    }
    if (shm_->collect_unrelayed(pending, 32) == 0) {
      (void)shm_->wait_for_update(kRelayWaitMs);
      continue;
    }

    for (auto const& [key, value] : pending) {
      if (!socket_->put_raw(key, value)) break;
      (void)shm_->mark_relayed(key);
    }
  }
}

void HierarchicalExchanger::replay_local_state() {
  if (!socket_ || !shm_) return;
  std::vector<std::pair<std::string, std::string>> batch;
  if (shm_->collect_all(batch, std::numeric_limits<size_t>::max()) == 0) {
    return;
  }
  for (auto const& [key, value] : batch) {
    if (!socket_->put_raw(key, value)) return;
    (void)shm_->mark_relayed(key);
  }
}

std::string HierarchicalExchanger::kv_namespace() const {
  return kv_namespace_for_port(port_);
}

void HierarchicalExchanger::apply_remote_entry(std::string const& key,
                                               std::string const& value) {
  if (!running_.load(std::memory_order_acquire) || !shm_) return;
  (void)shm_->apply_remote(key, value);
}

}  // namespace Transport
}  // namespace UKernel
