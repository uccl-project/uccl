#pragma once

#include "../../include/config.h"
#include "../../include/gpu_rt.h"
#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <semaphore.h>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace UKernel {
namespace Transport {

enum class PeerTransportKind { Unknown, Uccl, Ipc, Tcp };

struct CommunicatorMeta;

char const* peer_transport_kind_name(PeerTransportKind kind);

PeerTransportKind resolve_peer_transport_kind(
    CommunicatorConfig const& config, CommunicatorMeta const& local_meta,
    CommunicatorMeta const& peer_meta);

template <typename T>
bool serialize_object(T const& obj, std::string& out);
template <typename T>
bool deserialize_object(std::string const& in, T& obj);

#define UK_OOB_SERDE_METHODS(Type)                                             \
  bool serialize(std::string& out) const { return serialize_object(*this, out); } \
  bool deserialize(std::string const& in) {                                    \
    return deserialize_object(in, *this);                                      \
  }

struct CommunicatorMeta {
  std::string host_id;
  std::string ip;
  int local_id = -1;
  bool rdma_capable = false;

  UK_OOB_SERDE_METHODS(CommunicatorMeta)
};

struct MR {
  uint32_t id = 0;
  uint64_t address = 0;
  uint64_t length = 0;
  uint32_t lkey = 0;
  uint32_t key = 0;

  UK_OOB_SERDE_METHODS(MR)
};

struct NamedMR {
  uint32_t buffer_id = 0;
  MR mr{};

  UK_OOB_SERDE_METHODS(NamedMR)
};

struct NamedMRInfos {
  uint64_t generation = 0;
  std::vector<NamedMR> entries;

  UK_OOB_SERDE_METHODS(NamedMRInfos)
};

struct UCCLP2PInfo {
  std::string ip;
  uint16_t port = 0;
  int dev_idx = -1;
  int gpu_idx = -1;

  UCCLP2PInfo() = default;
  UCCLP2PInfo(std::string ip_, uint16_t port_, int dev_idx_, int gpu_idx_)
      : ip(std::move(ip_)), port(port_), dev_idx(dev_idx_), gpu_idx(gpu_idx_) {}

  UK_OOB_SERDE_METHODS(UCCLP2PInfo)
};

struct TcpP2PInfo {
  std::string ip;
  uint16_t port = 0;

  TcpP2PInfo() = default;
  TcpP2PInfo(std::string ip_, uint16_t port_)
      : ip(std::move(ip_)), port(port_) {}

  UK_OOB_SERDE_METHODS(TcpP2PInfo)
};

struct IpcBufferInfo {
  uint32_t ipc_id = 0;
  uint64_t binding_version = 0;
  gpuIpcMemHandle_t handle{};
  uint64_t base_offset = 0;
  uint64_t bytes = 0;
  int32_t device_idx = -1;
  bool valid = false;

  UK_OOB_SERDE_METHODS(IpcBufferInfo)
};

#define UK_OOB_DEFINE_VISIT_FIELDS(Type, BODY)                                 \
  template <class F>                                                           \
  inline void visit_fields(Type& v, F&& f) { BODY }                            \
  template <class F>                                                           \
  inline void visit_fields(Type const& v, F&& f) { BODY }

UK_OOB_DEFINE_VISIT_FIELDS(CommunicatorMeta, f("host_id", v.host_id);
                           f("ip", v.ip); f("local_id", v.local_id);
                           f("rdma_capable", v.rdma_capable);)
UK_OOB_DEFINE_VISIT_FIELDS(MR, f("id", v.id); f("address", v.address);
                           f("length", v.length); f("lkey", v.lkey);
                           f("key", v.key);)
UK_OOB_DEFINE_VISIT_FIELDS(NamedMR, f("buffer_id", v.buffer_id);
                           f("mr", v.mr);)
UK_OOB_DEFINE_VISIT_FIELDS(NamedMRInfos, f("generation", v.generation);
                           f("entries", v.entries);)
UK_OOB_DEFINE_VISIT_FIELDS(UCCLP2PInfo, f("ip", v.ip); f("port", v.port);
                           f("dev_idx", v.dev_idx); f("gpu_idx", v.gpu_idx);)
UK_OOB_DEFINE_VISIT_FIELDS(TcpP2PInfo, f("ip", v.ip); f("port", v.port);)
UK_OOB_DEFINE_VISIT_FIELDS(IpcBufferInfo, f("ipc_id", v.ipc_id);
                           f("binding_version", v.binding_version);
                           f("handle", v.handle);
                           f("base_offset", v.base_offset);
                           f("bytes", v.bytes);
                           f("device_idx", v.device_idx); f("valid", v.valid);)

#undef UK_OOB_DEFINE_VISIT_FIELDS
#undef UK_OOB_SERDE_METHODS

namespace detail {

template <typename T>
struct is_vector : std::false_type {};
template <typename T, typename A>
struct is_vector<std::vector<T, A>> : std::true_type {};

struct ByteWriter {
  std::string out;
  bool ok = true;

  void append(void const* src, size_t n) {
    if (!ok) return;
    auto const* p = reinterpret_cast<char const*>(src);
    out.append(p, n);
  }
};

struct ByteReader {
  std::string const& in;
  size_t pos = 0;
  bool ok = true;

  explicit ByteReader(std::string const& src) : in(src) {}

  bool take(void* dst, size_t n) {
    if (!ok || pos > in.size() || n > in.size() - pos) {
      ok = false;
      return false;
    }
    std::memcpy(dst, in.data() + pos, n);
    pos += n;
    return true;
  }
};

template <typename T>
void write_value(ByteWriter& w, T const& value);

template <typename T>
void read_value(ByteReader& r, T& value);

template <typename T>
void write_pod(ByteWriter& w, T const& value) {
  static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
  w.append(&value, sizeof(T));
}

template <typename T>
void read_pod(ByteReader& r, T& value) {
  static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
  (void)r.take(&value, sizeof(T));
}

inline void write_u32(ByteWriter& w, uint32_t value) { write_pod(w, value); }
inline void read_u32(ByteReader& r, uint32_t& value) { read_pod(r, value); }

template <typename T>
void write_value(ByteWriter& w, T const& value) {
  if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
    if (value.size() > static_cast<size_t>(UINT32_MAX)) {
      w.ok = false;
      return;
    }
    write_u32(w, static_cast<uint32_t>(value.size()));
    w.append(value.data(), value.size());
  } else if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
    uint8_t v = value ? 1 : 0;
    write_pod(w, v);
  } else if constexpr (std::is_enum_v<std::decay_t<T>>) {
    using U = std::underlying_type_t<std::decay_t<T>>;
    write_pod(w, static_cast<U>(value));
  } else if constexpr (std::is_arithmetic_v<std::decay_t<T>> ||
                       std::is_same_v<std::decay_t<T>, gpuIpcMemHandle_t>) {
    write_pod(w, value);
  } else if constexpr (is_vector<std::decay_t<T>>::value) {
    if (value.size() > static_cast<size_t>(UINT32_MAX)) {
      w.ok = false;
      return;
    }
    write_u32(w, static_cast<uint32_t>(value.size()));
    for (auto const& item : value) {
      write_value(w, item);
      if (!w.ok) return;
    }
  } else {
    visit_fields(value, [&](char const*, auto const& field) {
      write_value(w, field);
    });
  }
}

template <typename T>
void read_value(ByteReader& r, T& value) {
  if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
    uint32_t size = 0;
    read_u32(r, size);
    if (!r.ok || size > r.in.size() - r.pos) {
      r.ok = false;
      return;
    }
    value.assign(r.in.data() + r.pos, r.in.data() + r.pos + size);
    r.pos += size;
  } else if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
    uint8_t v = 0;
    read_pod(r, v);
    if (!r.ok) return;
    value = (v != 0);
  } else if constexpr (std::is_enum_v<std::decay_t<T>>) {
    using U = std::underlying_type_t<std::decay_t<T>>;
    U raw{};
    read_pod(r, raw);
    if (!r.ok) return;
    value = static_cast<T>(raw);
  } else if constexpr (std::is_arithmetic_v<std::decay_t<T>> ||
                       std::is_same_v<std::decay_t<T>, gpuIpcMemHandle_t>) {
    read_pod(r, value);
  } else if constexpr (is_vector<std::decay_t<T>>::value) {
    uint32_t n = 0;
    read_u32(r, n);
    if (!r.ok) return;
    value.clear();
    value.resize(n);
    for (auto& item : value) {
      read_value(r, item);
      if (!r.ok) return;
    }
  } else {
    visit_fields(value, [&](char const*, auto& field) { read_value(r, field); });
  }
}

}  // namespace detail

template <typename T>
inline bool serialize_object(T const& obj, std::string& out) {
  detail::ByteWriter writer;
  constexpr uint32_t kSerdeMagic = 0x4F4F4253;  // "OOBS"
  constexpr uint32_t kSerdeVersion = 1;
  detail::write_u32(writer, kSerdeMagic);
  detail::write_u32(writer, kSerdeVersion);
  detail::write_value(writer, obj);
  if (!writer.ok) return false;
  out.swap(writer.out);
  return true;
}

template <typename T>
inline bool deserialize_object(std::string const& in, T& obj) {
  detail::ByteReader reader(in);
  uint32_t magic = 0;
  uint32_t version = 0;
  detail::read_u32(reader, magic);
  detail::read_u32(reader, version);
  if (!reader.ok) return false;
  if (magic != 0x4F4F4253 || version != 1) return false;
  detail::read_value(reader, obj);
  return reader.ok && reader.pos == in.size();
}

class Exchanger {
 public:
  struct WaitOptions {
    WaitOptions(int max_retries_ = -1, int delay_ms_ = 100)
        : max_retries(max_retries_), delay_ms(delay_ms_) {}
    int max_retries;  // <0 means wait until available or invalid.
    int delay_ms;
  };

  virtual ~Exchanger() = default;

  virtual bool valid() const = 0;
  template <typename T,
            typename std::enable_if<
                !std::is_same<typename std::decay<T>::type, std::string>::value,
                int>::type = 0>
  bool put(std::string_view key, T const& obj) {
    std::string encoded;
    if (!serialize_object(obj, encoded)) return false;
    return put_raw(key, encoded);
  }
  template <typename T,
            typename std::enable_if<
                !std::is_same<typename std::decay<T>::type, std::string>::value,
                int>::type = 0>
  bool get(std::string_view key, T& obj) {
    std::string encoded;
    if (!get_raw(key, encoded)) return false;
    return deserialize_object(encoded, obj);
  }
  template <typename T,
            typename std::enable_if<
                !std::is_same<typename std::decay<T>::type, std::string>::value,
                int>::type = 0>
  bool wait(std::string_view key, T& obj,
            WaitOptions const& options = WaitOptions{}) {
    std::string encoded;
    if (!wait_raw(key, encoded, options)) return false;
    return deserialize_object(encoded, obj);
  }

 protected:
  virtual bool wait_raw(std::string_view key, std::string& value,
                        WaitOptions const& options = WaitOptions{});
  virtual bool put_raw(std::string_view key, std::string_view value) = 0;
  virtual bool get_raw(std::string_view key, std::string& value) = 0;
};

class ShmExchanger : public Exchanger {
 public:
  explicit ShmExchanger(std::string kv_namespace, bool create_if_missing = true,
                        int open_timeout_ms = 30000);
  ~ShmExchanger() override;

  bool valid() const override;
  bool apply_remote(std::string_view key, std::string_view value);
  bool begin_leader_run();
  bool mark_run_ready();
  bool wait_until_ready(int timeout_ms) const;
  size_t collect_unrelayed(
      std::vector<std::pair<std::string, std::string>>& out,
      size_t max_items = 32);
  size_t collect_all(std::vector<std::pair<std::string, std::string>>& out,
                     size_t max_items = 32) const;
  bool mark_relayed(std::string_view key);

 protected:
  bool wait_raw(std::string_view key, std::string& value,
                WaitOptions const& options) override;
  bool put_raw(std::string_view key, std::string_view value) override;
  bool get_raw(std::string_view key, std::string& value) override;

 private:
  struct SlotCacheHash {
    static uint64_t hash_key(std::string_view key);
  };
  static constexpr uint32_t kShmMagic = 0x554B4F42;  // "UKOB"
  static constexpr uint32_t kShmVersion = 6;
  static constexpr uint32_t kMaxSlots = 1024;
  static constexpr uint32_t kMaxKeyBytes = 256;
  static constexpr uint32_t kMaxValueBytes = 8192;
  static constexpr uint8_t kSlotEmpty = 0;
  static constexpr uint8_t kSlotUsed = 1;

  struct KvSlot {
    uint8_t state = kSlotEmpty;
    uint8_t relayed = 0;
    uint16_t reserved0 = 0;
    uint64_t write_seq = 0;
    uint32_t key_len = 0;
    uint32_t value_len = 0;
    char key[kMaxKeyBytes] = {};
    char value[kMaxValueBytes] = {};
  };

  struct KvShmHeader {
    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t write_seq = 0;
    uint32_t owner_pid = 0;
    uint32_t init_done = 0;
    uint64_t owner_start_ticks = 0;
    uint32_t slot_capacity = 0;
    uint32_t used_slots = 0;
    uint32_t first_free_hint = 0;
    uint32_t reserved0 = 0;
    pthread_mutex_t mu;
    sem_t notify_sem;
    KvSlot slots[kMaxSlots];
  };

  bool init_shared_store();
  void close_shared_store();
  bool put_encoded(std::string_view key, std::string_view value,
                   bool mark_relayed);
  size_t collect_entries(std::vector<std::pair<std::string, std::string>>& out,
                         size_t max_items, size_t start_index,
                         bool only_unrelayed) const;
  bool wait_for_update(int delay_ms) const;
  int find_slot_locked(std::string_view key) const;
  int find_free_slot_locked() const;
  void rebuild_metadata_locked();
  bool lock_store() const;
  void unlock_store() const;
  void maybe_post_sem() const;

  friend class HierarchicalExchanger;

  std::string kv_namespace_;
  bool create_if_missing_ = true;
  int open_timeout_ms_ = 30000;
  int shm_fd_ = -1;
  KvShmHeader* shm_ = nullptr;
  size_t next_unrelayed_scan_index_ = 0;
  mutable std::unordered_map<uint64_t, uint32_t> slot_index_cache_;
};

class SocketExchanger : public Exchanger {
 public:
  using ReceiveCallback =
      std::function<void(std::string const&, std::string const&)>;

  SocketExchanger(bool is_root, std::string host, int port, int timeout_ms,
                  size_t max_line_bytes, ReceiveCallback on_receive);
  ~SocketExchanger() override;

  bool valid() const override;
  bool start();
  uint64_t connection_epoch() const;

 protected:
  bool put_raw(std::string_view key, std::string_view value) override;
  bool get_raw(std::string_view key, std::string& value) override;

 private:
  struct PeerConn;

  struct RootReactorState {
    int listen_fd = -1;
    int wakeup_read_fd = -1;
    int wakeup_write_fd = -1;
    std::thread thread;
    std::mutex peers_mu;
    std::unordered_map<int, std::shared_ptr<PeerConn>> peers;
    std::mutex pending_mu;
    std::deque<int> pending_write_interest;
    std::unordered_set<int> pending_write_interest_set;
  };

  struct PeerConn {
    int fd = -1;
    std::mutex send_mu;
    std::string read_buffer;
    size_t read_offset = 0;
    std::string write_buffer;
    size_t write_offset = 0;
  };

  bool start_root_server();
  void stop_root_server();
  void root_accept_loop();
  void root_broadcast_pub(std::string const& key, std::string const& value,
                          int exclude_fd);
  void close_root_peer(int fd);
  void queue_root_peer_write_interest(int fd);
  void drain_root_peer_write_interest(std::vector<int>& fds);
  bool notify_root_wakeup() const;
  bool enqueue_frame_locked(std::string& write_buffer,
                            std::string const& frame) const;
  bool flush_write_buffer_locked(int fd, std::string& write_buffer,
                                 size_t& write_offset) const;
  bool recv_frame_buffered(int fd, std::string& buffer, size_t& read_offset,
                           std::string& frame, int timeout_ms) const;
  bool connect_upstream_locked();
  void close_upstream_locked();
  void upstream_loop();
  void handle_publish(std::string const& key, std::string const& value,
                      bool broadcast_from_root, int exclude_fd);
  void snapshot_entries(std::vector<std::pair<std::string, std::string>>& out);
  bool wait_until_ready(int timeout_ms);

  friend class HierarchicalExchanger;

  std::string host_;
  int port_ = 0;
  int timeout_ms_ = 0;
  bool is_root_ = false;
  size_t max_line_bytes_ = 0;
  ReceiveCallback on_receive_;

  std::atomic<bool> running_{false};
  std::atomic<bool> started_{false};
  std::atomic<bool> ready_{false};
  std::atomic<uint64_t> connection_epoch_{0};

  mutable std::mutex entries_mu_;
  std::unordered_map<std::string, std::string> entries_;

  RootReactorState root_;

  mutable std::mutex upstream_mu_;
  int upstream_fd_ = -1;
  std::thread upstream_thread_;
  std::string upstream_read_buffer_;
  size_t upstream_read_offset_ = 0;
  std::string upstream_write_buffer_;
  size_t upstream_write_offset_ = 0;
  std::atomic<bool> upstream_sync_complete_{false};

  std::mutex pending_mu_;
  std::condition_variable pending_cv_;
  std::deque<std::pair<std::string, std::string>> pending_pubs_;

  std::mutex ready_mu_;
  std::condition_variable ready_cv_;
};

class HierarchicalExchanger : public Exchanger {
 public:
  HierarchicalExchanger(bool is_server, std::string const& host, int port,
                        int timeout_ms = 3000,
                        size_t max_line_bytes = 1 * 1024 * 1024,
                        int local_id = -1);
  ~HierarchicalExchanger();

  bool valid() const override;

 private:
  bool is_node_leader() const;
  void relay_loop();
  void replay_local_state();
  std::string kv_namespace() const;
  void apply_remote_entry(std::string const& key, std::string const& value);

  bool wait_raw(std::string_view key, std::string& value,
                WaitOptions const& options) override;
  bool put_raw(std::string_view key, std::string_view value) override;
  bool get_raw(std::string_view key, std::string& value) override;

  std::string host_;
  int port_ = 0;
  bool is_server_ = false;
  int local_id_ = -1;
  bool node_leader_ = false;
  std::atomic<bool> running_{false};
  uint64_t last_replayed_epoch_ = 0;

  std::unique_ptr<ShmExchanger> shm_;
  std::unique_ptr<SocketExchanger> socket_;
  std::thread relay_thread_;
};

}  // namespace Transport
}  // namespace UKernel
