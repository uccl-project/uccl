#include "cxi_endpoint.h"
#include "engine.h"
#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {

constexpr uint32_t kCxiEndpointInfoMagic = 0x43584950;  // "CXIP"
// Version 2: one endpoint name per lane, and the FIFO metadata carries one MR
// key per lane.
constexpr uint32_t kCxiEndpointInfoVersion = 2;

// Followed by num_lanes entries of {uint16_t size; uint8_t name[size]}.
struct CxiEndpointInfoHeader {
  uint32_t magic = kCxiEndpointInfoMagic;
  uint32_t version = kCxiEndpointInfoVersion;
  int32_t gpu_index = -1;
  uint16_t oob_port = 0;
  uint16_t num_lanes = 0;
};

// RMA pieces start at multiples of this. Pieces that start at unaligned
// offsets were measured 4 to 5 times slower.
constexpr size_t kCxiPieceAlign = 4096;

void check_fi(char const* what, int ret) {
  if (ret != 0) {
    throw std::runtime_error(std::string(what) + ": " + fi_strerror(-ret));
  }
}

fi_threading threading_hint() {
  // Default to FI_THREAD_SAFE: libfabric >= 2.x rejects FI_THREAD_ENDPOINT in
  // the CXI provider's domain-attr check, failing fi_getinfo with ENODATA.
  // FI_THREAD_SAFE is accepted by both libfabric 1.x and 2.x.
  char const* value = std::getenv("UCCL_CXI_THREADING");
  if (!value || std::strcmp(value, "safe") == 0) return FI_THREAD_SAFE;
  if (std::strcmp(value, "endpoint") == 0) return FI_THREAD_ENDPOINT;
  if (std::strcmp(value, "completion") == 0) return FI_THREAD_COMPLETION;
  if (std::strcmp(value, "domain") == 0) return FI_THREAD_DOMAIN;
  if (std::strcmp(value, "fid") == 0) return FI_THREAD_FID;
  if (std::strcmp(value, "unspec") == 0) return FI_THREAD_UNSPEC;
  throw std::runtime_error(
      std::string("Invalid UCCL_CXI_THREADING=") + value +
      " (expected one of: safe, endpoint, completion, domain, fid, unspec)");
}

// Reads <device_dir>/numa_node. Returns -1 if it is missing or unknown.
int sysfs_numa_node(std::filesystem::path const& device_dir) {
  std::ifstream file(device_dir / "numa_node");
  int node = -1;
  if (!(file >> node) || node < 0) return -1;
  return node;
}

std::filesystem::path cxi_device_dir(std::string const& domain_name) {
  return std::filesystem::path("/sys/class/cxi") / domain_name / "device";
}

int cxi_device_index_on_gpu_numa_node(int gpu_index) {
  auto const gpus = uccl::get_gpu_cards();
  if (static_cast<size_t>(gpu_index) >= gpus.size()) return -1;
  int const node = sysfs_numa_node(gpus[gpu_index]);
  if (node < 0) return -1;

  int found = -1;
  std::error_code ec;
  for (auto const& entry :
       std::filesystem::directory_iterator("/sys/class/cxi", ec)) {
    std::string const name = entry.path().filename().string();
    if (name.rfind("cxi", 0) != 0) continue;

    char* end = nullptr;
    long const index = std::strtol(name.c_str() + 3, &end, 10);
    if (end == name.c_str() + 3 || *end != '\0') continue;
    if (sysfs_numa_node(cxi_device_dir(name)) != node) continue;
    if (found >= 0) return -1;  // More than one NIC on this NUMA node.
    found = static_cast<int>(index);
  }
  return found;
}

int cxi_device_index_for_gpu(int gpu_index) {
  char const* value = std::getenv("UCCL_CXI_DEVICE_INDEX");
  if (value && value[0] != '\0') return std::atoi(value);
  if (gpu_index < 0 || static_cast<uint32_t>(gpu_index) == INVALID_GPU) {
    return -1;
  }
  int const local = cxi_device_index_on_gpu_numa_node(gpu_index);
  if (local >= 0) return local;
  return gpu_index % 4;
}

std::string cxi_domain_name_for_gpu(int gpu_index) {
  char const* domain = std::getenv("UCCL_CXI_DOMAIN");
  if (domain && domain[0] != '\0') return domain;

  int const device_index = cxi_device_index_for_gpu(gpu_index);
  if (device_index < 0) return "";

  std::ostringstream os;
  os << "cxi" << device_index;
  return os.str();
}

size_t cxi_size_env(char const* name, size_t fallback) {
  char const* value = std::getenv(name);
  if (!value || value[0] == '\0') return fallback;
  char* end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || parsed == 0) return fallback;
  return static_cast<size_t>(parsed);
}

int cxi_num_lanes() {
  char const* value = std::getenv("UCCL_CXI_NUM_LANES");
  if (!value || value[0] == '\0') return kCxiMaxLanes;
  char* end = nullptr;
  long const lanes = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || lanes < 1 || lanes > kCxiMaxLanes) {
    throw std::runtime_error(std::string("Invalid UCCL_CXI_NUM_LANES=") +
                             value + " (expected 1 to " +
                             std::to_string(kCxiMaxLanes) + ")");
  }
  return static_cast<int>(lanes);
}

bool is_cuda_pointer(void* ptr, int& cuda_device) {
  gpuPointerAttribute_t attrs{};
  gpuError_t err = gpuPointerGetAttributes(&attrs, ptr);
  if (err != gpuSuccess) {
    gpuGetLastError();
    return false;
  }
  if (gpuMemTypeOf(attrs) == gpuMemoryTypeDevice ||
      gpuMemTypeOf(attrs) == gpuMemoryTypeManaged) {
    cuda_device = attrs.device;
    return true;
  }
  return false;
}

void set_fi_context_owner(fi_context2& ctx, void* owner) {
  ctx = fi_context2{};
  ctx.internal[0] = owner;
}

void* get_fi_context_owner(void* context) {
  auto* fi_ctx = static_cast<fi_context2*>(context);
  return fi_ctx ? fi_ctx->internal[0] : nullptr;
}

std::string cq_error_string(fid_cq* cq, fi_cq_err_entry const& err) {
  char buf[256];
  char const* msg =
      fi_cq_strerror(cq, err.prov_errno, err.err_data, buf, sizeof(buf));
  if (msg) return msg;
  return fi_strerror(err.err);
}

// Drain and stringify any pending completion-queue error entries. Used to
// attach provider-level detail (prov_errno, op_context, ...) to a failed RMA
// post so hard failures like FI_EIO are diagnosable rather than opaque.
// Consumes the entries without marking their owning OpContexts failed, which
// starves poll_cq_locked() of error completions for unrelated inflight ops —
// only call this when the process is about to abort.
std::string drain_cq_errors(fid_cq* cq) {
  if (!cq) return "";

  std::ostringstream os;
  bool wrote = false;
  for (int i = 0; i < 16; ++i) {
    fi_cq_err_entry err{};
    ssize_t rc = fi_cq_readerr(cq, &err, 0);
    if (rc == -FI_EAGAIN || rc == 0) break;
    if (rc < 0) {
      if (wrote) os << "; ";
      os << "fi_cq_readerr failed rc=" << rc << " " << fi_strerror(-rc);
      break;
    }
    if (wrote) os << "; ";
    wrote = true;
    os << "cq_err[" << i << "] err=" << err.err << "(" << fi_strerror(err.err)
       << ") prov_errno=" << err.prov_errno << " flags=0x" << std::hex
       << err.flags << " len=0x" << err.len << " olen=0x" << err.olen
       << " data=0x" << err.data << " op_context=0x"
       << reinterpret_cast<uintptr_t>(err.op_context) << std::dec << " msg=\""
       << cq_error_string(cq, err) << "\"";
  }
  return os.str();
}

void close_mrs(std::vector<fid_mr*>& mrs) {
  for (fid_mr* mr : mrs) {
    int ret = fi_close(&mr->fid);
    if (ret != 0) {
      UCCL_LOG(ERROR) << "fi_close(mr) failed: " << fi_strerror(-ret);
    }
  }
  mrs.clear();
}

}  // namespace

CxiEndpoint::CxiEndpoint(int gpu_index, uint64_t port) : gpu_index_(gpu_index) {
  if (gpu_index != INVALID_GPU) {
    initialize_rdma_ctx_for_gpu(gpu_index);
  }

  oob_server_ = std::make_shared<EpollServer>(
      static_cast<int>(port),
      [this](std::string const& input, std::string& output,
             std::string const& ip, int client_port) {
        this->process_meta(input, output, ip, client_port);
      });
  oob_client_ = std::make_shared<EpollClient>();

  if (!oob_server_->start()) {
    UCCL_LOG(ERROR) << "Failed to start CXI OOB server";
    throw std::runtime_error("Failed to start CXI OOB server");
  }
  if (!oob_client_->start()) {
    UCCL_LOG(ERROR) << "Failed to start CXI OOB client";
    throw std::runtime_error("Failed to start CXI OOB client");
  }
}

CxiEndpoint::~CxiEndpoint() {
  if (oob_client_) oob_client_->stop();
  if (oob_server_) oob_server_->stop();

  {
    std::lock_guard<std::mutex> lock(op_mutex_);
    inflight_ops_.clear();
  }

  // Close in dependency order: the endpoint is bound to the AV and CQ, so they
  // stay busy until it is closed, and a busy domain makes fi_close(fabric)
  // corrupt libfabric's fabric list.
  for (fid_ep* ep : eps_) fi_close(&ep->fid);
  eps_.clear();
  if (av_) fi_close(&av_->fid);
  if (cq_) fi_close(&cq_->fid);
  if (domain_) fi_close(&domain_->fid);
  if (fabric_) fi_close(&fabric_->fid);
  if (info_) fi_freeinfo(info_);
}

bool CxiEndpoint::initialize_rdma_ctx_for_gpu(
    int gpu_index, std::vector<size_t> const& device_ids) {
  (void)device_ids;
  gpu_index_ = gpu_index;
  if (fabric_initialized_) return true;

  init_fabric(gpu_index);
  fabric_initialized_ = true;
  UCCL_LOG(INFO) << "CxiEndpoint initialized for GPU " << gpu_index
                 << " on domain "
                 << (info_->domain_attr->name ? info_->domain_attr->name : "?")
                 << ", NIC NUMA node " << nic_numa_node(gpu_index);
  return true;
}

// Before the domain is open (lazy engine setup), use the NIC that init_fabric
// will open for this GPU.
int CxiEndpoint::nic_numa_node(int gpu_index) const {
  bool const open = info_ && info_->domain_attr && info_->domain_attr->name;
  std::string const name =
      open ? info_->domain_attr->name : cxi_domain_name_for_gpu(gpu_index);
  return name.empty() ? -1 : sysfs_numa_node(cxi_device_dir(name));
}

void CxiEndpoint::init_fabric(int gpu_index) {
  fi_info* hints = fi_allocinfo();
  if (!hints) throw std::runtime_error("fi_allocinfo failed");

  hints->fabric_attr->prov_name = strdup("cxi");
  std::string domain_name = cxi_domain_name_for_gpu(gpu_index);
  if (!domain_name.empty()) {
    hints->domain_attr->name = strdup(domain_name.c_str());
  }
  hints->ep_attr->type = FI_EP_RDM;
  hints->caps = FI_TAGGED | FI_MSG | FI_HMEM | FI_RMA | FI_READ | FI_WRITE |
                FI_REMOTE_WRITE | FI_REMOTE_READ | FI_LOCAL_COMM |
                FI_REMOTE_COMM;
  hints->mode = FI_CONTEXT | FI_CONTEXT2;
  hints->domain_attr->threading = threading_hint();
  hints->domain_attr->control_progress = FI_PROGRESS_UNSPEC;
  hints->domain_attr->data_progress = FI_PROGRESS_UNSPEC;
  hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_ENDPOINT |
                                FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                                FI_MR_PROV_KEY;
  hints->domain_attr->mr_key_size = 2;
  hints->tx_attr->size = cxi_size_env("UCCL_CXI_TX_QUEUE_SIZE", 4096);
  hints->rx_attr->size = cxi_size_env("UCCL_CXI_RX_QUEUE_SIZE", 4096);
  hints->tx_attr->msg_order = FI_ORDER_SAS;
  hints->rx_attr->msg_order = FI_ORDER_SAS;

  try {
    check_fi("fi_getinfo(cxi)",
             fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &info_));

    uint32_t const lf_ver = fi_version();
    UCCL_LOG(INFO) << "libfabric runtime version " << FI_MAJOR(lf_ver) << "."
                   << FI_MINOR(lf_ver);
    if (lf_ver < FI_VERSION(2, 0)) {
      UCCL_LOG(WARN)
          << "libfabric " << FI_MAJOR(lf_ver) << "." << FI_MINOR(lf_ver)
          << " detected: performance may be poor for large KV pools; see "
             "uccl-project/uccl#956. Consider upgrading to libfabric >= 2.5.";
    }
    info_->tx_attr->op_flags |= FI_DELIVERY_COMPLETE;
    UCCL_LOG(INFO) << "CXI FI_DELIVERY_COMPLETE enabled";

    check_fi("fi_fabric", fi_fabric(info_->fabric_attr, &fabric_, nullptr));
    check_fi("fi_domain", fi_domain(fabric_, info_, &domain_, nullptr));

    // MR match events need the target to poll its CQ, so the idle receive
    // proxy keeps polling when they are on.
    char const* match = std::getenv("FI_CXI_MR_MATCH_EVENTS");
    mr_match_events_ =
        match && match[0] != '\0' && std::strcmp(match, "0") != 0;
    if (mr_match_events_) {
      UCCL_LOG(INFO, UCCL_INIT)
          << "CXI MR match events are on; the receive proxy keeps polling the "
             "CQ when idle";
    }

    fi_cq_attr cq_attr{};
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.size = cxi_size_env("UCCL_CXI_CQ_SIZE", 8192);
    check_fi("fi_cq_open", fi_cq_open(domain_, &cq_attr, &cq_, nullptr));

    fi_av_attr av_attr{};
    av_attr.type = FI_AV_TABLE;
    check_fi("fi_av_open", fi_av_open(domain_, &av_attr, &av_, nullptr));

    // All lanes share the one CQ and AV. Only lane 0 posts; the other lanes
    // are RMA targets for the peer.
    int const num_lanes = cxi_num_lanes();
    for (int lane = 0; lane < num_lanes; ++lane) {
      fid_ep* ep = nullptr;
      check_fi("fi_endpoint", fi_endpoint(domain_, info_, &ep, nullptr));
      eps_.push_back(ep);
      check_fi("fi_ep_bind(cq)",
               fi_ep_bind(ep, &cq_->fid, FI_TRANSMIT | FI_RECV));
      check_fi("fi_ep_bind(av)", fi_ep_bind(ep, &av_->fid, 0));

#ifdef FI_OPT_CUDA_API_PERMITTED
      bool cuda_api_permitted = false;
      check_fi("fi_setopt(FI_OPT_CUDA_API_PERMITTED)",
               fi_setopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_CUDA_API_PERMITTED,
                         &cuda_api_permitted, sizeof(cuda_api_permitted)));
#endif

      check_fi("fi_enable", fi_enable(ep));
    }
    // A post larger than max_msg_size fails with -FI_EMSGSIZE (CXI: 4 GiB
    // - 1 B), so no piece is larger than that, rounded down to a multiple of
    // kCxiPieceAlign.
    max_piece_bytes_ =
        info_->ep_attr->max_msg_size / kCxiPieceAlign * kCxiPieceAlign;
    if (max_piece_bytes_ == 0) {
      throw std::runtime_error("CXI max_msg_size is below 4 KiB");
    }
    UCCL_LOG(INFO) << "CXI lanes: " << num_lanes << ", max piece bytes "
                   << max_piece_bytes_ << ", mr_key_size "
                   << info_->domain_attr->mr_key_size;
  } catch (...) {
    fi_freeinfo(hints);
    throw;
  }

  fi_freeinfo(hints);
}

uint16_t CxiEndpoint::get_p2p_listen_port() {
  return oob_server_ ? oob_server_->get_port() : 0;
}

int CxiEndpoint::get_p2p_listen_fd() {
  return oob_server_ ? oob_server_->get_listen_fd() : -1;
}

std::shared_ptr<EpollClient> CxiEndpoint::get_oob_client() {
  return oob_client_;
}

std::string CxiEndpoint::get_oob_conn_key(uint64_t peer_id) const {
  std::shared_lock<std::shared_mutex> lock(peer_oob_conn_keys_mutex_);
  auto it = peer_oob_conn_keys_.find(peer_id);
  return it == peer_oob_conn_keys_.end() ? "" : it->second;
}

CxiEndpoint::EndpointInfo CxiEndpoint::local_endpoint_info() const {
  if (!fabric_initialized_ || eps_.empty()) {
    throw std::runtime_error("CXI endpoint not initialized");
  }

  EndpointInfo info;
  info.oob_port = oob_server_ ? oob_server_->get_port() : 0;
  info.gpu_index = gpu_index_;
  for (fid_ep* ep : eps_) {
    std::vector<uint8_t> name(512);
    size_t len = name.size();
    check_fi("fi_getname", fi_getname(&ep->fid, name.data(), &len));
    name.resize(len);
    info.ep_names.push_back(std::move(name));
  }
  return info;
}

std::string CxiEndpoint::serialize_endpoint_info(
    EndpointInfo const& info) const {
  CxiEndpointInfoHeader header;
  header.gpu_index = info.gpu_index;
  header.oob_port = info.oob_port;
  header.num_lanes = static_cast<uint16_t>(info.ep_names.size());

  std::string bytes(reinterpret_cast<char const*>(&header), sizeof(header));
  for (auto const& name : info.ep_names) {
    if (name.size() > std::numeric_limits<uint16_t>::max()) {
      throw std::runtime_error("CXI endpoint name too large");
    }
    uint16_t const size = static_cast<uint16_t>(name.size());
    bytes.append(reinterpret_cast<char const*>(&size), sizeof(size));
    bytes.append(reinterpret_cast<char const*>(name.data()), name.size());
  }
  return bytes;
}

CxiEndpoint::EndpointInfo CxiEndpoint::deserialize_endpoint_info(
    std::string const& bytes) const {
  if (bytes.size() < sizeof(CxiEndpointInfoHeader)) {
    throw std::runtime_error("CXI endpoint metadata too short");
  }

  CxiEndpointInfoHeader header{};
  std::memcpy(&header, bytes.data(), sizeof(header));
  // The bytes must be CXI endpoint metadata, both sides must use the same
  // format (version 2 has one endpoint name per lane), and the peer's lane
  // count must fit the per-lane MR keys in the FIFO metadata (at most
  // kCxiMaxLanes).
  if (header.magic != kCxiEndpointInfoMagic) {
    throw std::runtime_error("Invalid CXI endpoint metadata");
  }
  if (header.version != kCxiEndpointInfoVersion) {
    throw std::runtime_error("CXI endpoint metadata version " +
                             std::to_string(header.version) + ", expected " +
                             std::to_string(kCxiEndpointInfoVersion) +
                             ": both sides must run the same UCCL p2p version");
  }
  if (header.num_lanes < 1 || header.num_lanes > kCxiMaxLanes) {
    throw std::runtime_error(
        "CXI peer has " + std::to_string(header.num_lanes) +
        " lanes, expected 1 to " + std::to_string(kCxiMaxLanes));
  }

  EndpointInfo info;
  info.oob_port = header.oob_port;
  info.gpu_index = header.gpu_index;
  size_t pos = sizeof(header);
  for (uint16_t lane = 0; lane < header.num_lanes; ++lane) {
    uint16_t size = 0;
    if (bytes.size() < pos + sizeof(size)) {
      throw std::runtime_error("CXI endpoint metadata length mismatch");
    }
    std::memcpy(&size, bytes.data() + pos, sizeof(size));
    pos += sizeof(size);
    if (size == 0 || bytes.size() < pos + size) {
      throw std::runtime_error("CXI endpoint metadata length mismatch");
    }
    auto const* name = reinterpret_cast<uint8_t const*>(bytes.data() + pos);
    info.ep_names.emplace_back(name, name + size);
    pos += size;
  }
  if (pos != bytes.size()) {
    throw std::runtime_error("CXI endpoint metadata length mismatch");
  }
  return info;
}

std::vector<fi_addr_t> CxiEndpoint::insert_peer(EndpointInfo const& info) {
  if (!av_) throw std::runtime_error("CXI AV missing");
  if (info.ep_names.empty()) {
    throw std::runtime_error("CXI peer endpoint name is empty");
  }

  std::vector<fi_addr_t> addrs;
  for (auto const& name : info.ep_names) {
    fi_addr_t addr = FI_ADDR_UNSPEC;
    int ret = fi_av_insert(av_, name.data(), 1, &addr, 0, nullptr);
    if (ret != 1) check_fi("fi_av_insert", ret < 0 ? ret : -FI_EINVAL);
    addrs.push_back(addr);
  }
  return addrs;
}

ConnID CxiEndpoint::uccl_connect(int remote_gpuidx, std::string remote_ip,
                                 uint16_t remote_port) {
  int32_t peer_id = next_send_peer_id_.fetch_add(1, std::memory_order_relaxed);

  std::string conn_key;
  while (conn_key.empty()) {
    conn_key = oob_client_->connect_to_server(remote_ip, remote_port);
    if (conn_key.empty()) std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  {
    std::unique_lock<std::shared_mutex> lock(peer_oob_conn_keys_mutex_);
    peer_oob_conn_keys_[peer_id] = conn_key;
  }
  auto erase_oob_conn_key = [this, peer_id] {
    std::unique_lock<std::shared_mutex> lock(peer_oob_conn_keys_mutex_);
    peer_oob_conn_keys_.erase(peer_id);
  };

  EndpointInfo local_info = local_endpoint_info();
  std::string payload = serialize_endpoint_info(local_info);

  auto promise = std::make_shared<std::promise<EndpointInfo>>();
  std::future<EndpointInfo> future = promise->get_future();
  bool sent = oob_client_->send_meta(
      conn_key, payload, [this, promise](std::string const& response) {
        // Pass a bad reply (for example a version mismatch) to the caller,
        // so that the connect fails at once with that error.
        try {
          promise->set_value(deserialize_endpoint_info(response));
        } catch (...) {
          promise->set_exception(std::current_exception());
        }
      });
  if (!sent) {
    UCCL_LOG(ERROR) << "Failed to send CXI endpoint metadata";
    erase_oob_conn_key();
    return ConnID{nullptr, -1, -1, UINT64_MAX};
  }

  auto const timeout_ms = std::chrono::milliseconds(10000);
  if (future.wait_for(timeout_ms) == std::future_status::timeout) {
    UCCL_LOG(ERROR) << "Timeout waiting for CXI endpoint handshake";
    erase_oob_conn_key();
    return ConnID{nullptr, -1, -1, UINT64_MAX};
  }

  EndpointInfo remote_info;
  try {
    remote_info = future.get();
  } catch (std::exception const& e) {
    UCCL_LOG(ERROR) << "CXI endpoint handshake failed: " << e.what();
    erase_oob_conn_key();
    return ConnID{nullptr, -1, -1, UINT64_MAX};
  }
  std::vector<fi_addr_t> addrs = insert_peer(remote_info);
  {
    std::unique_lock<std::shared_mutex> lock(peer_mutex_);
    peers_[peer_id] =
        Peer{std::move(addrs), remote_ip, remote_port, remote_gpuidx};
  }

  ConnID conn_id;
  conn_id.context = reinterpret_cast<void*>(static_cast<intptr_t>(peer_id));
  conn_id.sock_fd = 0;
  conn_id.dev = gpu_index_;
  conn_id.peer_id = peer_id;
  return conn_id;
}

ConnID CxiEndpoint::uccl_accept(std::string& remote_ip, int* remote_gpuidx) {
  AcceptedMeta accepted;
  uint64_t peer_id = UINT64_MAX;

  while (!stop_accept_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::shared_mutex> lock(accepted_meta_mutex_);
      if (!accepted_meta_.empty()) {
        auto it = accepted_meta_.begin();
        peer_id = it->first;
        accepted = it->second;
        accepted_meta_.erase(it);
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  if (peer_id == UINT64_MAX) {
    return ConnID{nullptr, -1, -1, UINT64_MAX};
  }

  remote_ip = accepted.ip;
  if (remote_gpuidx) *remote_gpuidx = accepted.gpu_id;

  ConnID conn_id;
  conn_id.context = reinterpret_cast<void*>(static_cast<intptr_t>(peer_id));
  conn_id.sock_fd = 0;
  conn_id.dev = gpu_index_;
  conn_id.peer_id = peer_id;
  return conn_id;
}

void CxiEndpoint::stop_accept() {
  stop_accept_.store(true, std::memory_order_release);
}

void CxiEndpoint::process_meta(std::string const& input, std::string& output,
                               std::string const& client_ip, int client_port) {
  {
    NotifyMsg notify_msg;
    if (deserialize_notify_msg(input, notify_msg)) {
      UCCL_LOG(INFO, UCCL_P2P)
          << "process_meta: received notification from " << notify_msg.name
          << " (" << notify_msg.msg.size() << " bytes)";
      std::lock_guard<std::mutex> lock(notify_mutex);
      notify_list.push_back(std::move(notify_msg));
      output = "";
      return;
    }
  }

  EndpointInfo peer_info = deserialize_endpoint_info(input);
  std::vector<fi_addr_t> addrs = insert_peer(peer_info);
  uint64_t peer_id = next_recv_peer_id_.fetch_add(1, std::memory_order_relaxed);

  {
    std::unique_lock<std::shared_mutex> lock(peer_mutex_);
    peers_[peer_id] =
        Peer{std::move(addrs), client_ip, static_cast<uint16_t>(client_port),
             peer_info.gpu_index};
  }

  {
    std::unique_lock<std::shared_mutex> lock(accepted_meta_mutex_);
    AcceptedMeta accepted;
    accepted.ip = client_ip;
    accepted.port = static_cast<uint16_t>(client_port);
    accepted.gpu_id = peer_info.gpu_index;
    accepted.peer_id = peer_id;
    accepted_meta_[peer_id] = accepted;
  }

  if (peer_info.oob_port > 0) {
    std::string reverse_key =
        oob_client_->connect_to_server(client_ip, peer_info.oob_port);
    if (!reverse_key.empty()) {
      std::unique_lock<std::shared_mutex> lock(peer_oob_conn_keys_mutex_);
      peer_oob_conn_keys_[peer_id] = reverse_key;
    }
  }

  output = serialize_endpoint_info(local_endpoint_info());
}

int CxiEndpoint::uccl_regmr(void* data, size_t len,
                            std::shared_ptr<CxiMemoryRegion>& region) {
  if (!fabric_initialized_) {
    UCCL_LOG(ERROR)
        << "CXI endpoint not initialized before memory registration";
    return -1;
  }
  if (!data || len == 0) {
    UCCL_LOG(ERROR) << "Invalid CXI memory registration";
    return -1;
  }

  auto out = std::make_shared<CxiMemoryRegion>();
  out->addr = data;
  out->len = len;

  int cuda_device = -1;
  bool const is_cuda = is_cuda_pointer(data, cuda_device);
  uint64_t const access =
      FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_WRITE | FI_REMOTE_READ;

  // With FI_MR_ENDPOINT, an MR binds to one endpoint, so register the buffer
  // once per lane. Each lane adds one MR (registration time and one NIC
  // target entry per buffer); UCCL_CXI_NUM_LANES=1 keeps one MR per buffer.
  for (fid_ep* ep : eps_) {
    fid_mr* mr = nullptr;
    if (is_cuda) {
      iovec iov{};
      iov.iov_base = data;
      iov.iov_len = len;

      fi_mr_attr mr_attr{};
      mr_attr.mr_iov = &iov;
      mr_attr.iov_count = 1;
      mr_attr.access = access;
      mr_attr.iface = FI_HMEM_CUDA;
      mr_attr.device.cuda = cuda_device;

      int ret = fi_mr_regattr(domain_, &mr_attr, 0, &mr);
      if (ret != 0) {
        UCCL_LOG(ERROR) << "fi_mr_regattr(cuda) failed: " << fi_strerror(-ret);
        close_mrs(out->mrs);
        return -1;
      }
    } else {
      int ret = fi_mr_reg(domain_, data, len, access, 0, 0, 0, &mr, nullptr);
      if (ret != 0) {
        UCCL_LOG(ERROR) << "fi_mr_reg(host) failed: " << fi_strerror(-ret);
        close_mrs(out->mrs);
        return -1;
      }
    }
    out->mrs.push_back(mr);

    if (info_->domain_attr->mr_mode & FI_MR_ENDPOINT) {
      int ret = fi_mr_bind(mr, &ep->fid, 0);
      if (ret != 0) {
        UCCL_LOG(ERROR) << "fi_mr_bind(ep) failed: " << fi_strerror(-ret);
        close_mrs(out->mrs);
        return -1;
      }
      ret = fi_mr_enable(mr);
      if (ret != 0) {
        UCCL_LOG(ERROR) << "fi_mr_enable failed: " << fi_strerror(-ret);
        close_mrs(out->mrs);
        return -1;
      }
    }
    out->keys.push_back(fi_mr_key(mr));
  }

  region = out;
  return 0;
}

void CxiEndpoint::uccl_deregmr(std::shared_ptr<CxiMemoryRegion> const& region) {
  if (!region) return;
  close_mrs(region->mrs);
  region->keys.clear();
}

int CxiEndpoint::uccl_read_async(
    ConnID const& conn, std::shared_ptr<CxiMemoryRegion> const& local_mr,
    void* dst, size_t size, FifoItem const& fifo_item, UcclRequest* ureq) {
  return post_rma(true, conn, local_mr, dst, size, fifo_item, ureq);
}

int CxiEndpoint::uccl_write_async(
    ConnID const& conn, std::shared_ptr<CxiMemoryRegion> const& local_mr,
    void* src, size_t size, FifoItem const& fifo_item, UcclRequest* ureq) {
  return post_rma(false, conn, local_mr, src, size, fifo_item, ureq);
}

int CxiEndpoint::post_rma(bool is_read, ConnID const& conn,
                          std::shared_ptr<CxiMemoryRegion> const& local_mr,
                          void* local, size_t size, FifoItem const& fifo_item,
                          UcclRequest* ureq) {
  if (!local_mr || local_mr->mrs.empty() || !local || !ureq) return -1;

  CxiFifoMetadata metadata;
  if (!decode_cxi_fifo_metadata(fifo_item, metadata)) {
    UCCL_LOG(ERROR) << "Invalid CXI FIFO metadata";
    return -1;
  }

  uintptr_t const local_addr = reinterpret_cast<uintptr_t>(local);
  uintptr_t const local_base = reinterpret_cast<uintptr_t>(local_mr->addr);
  if (local_addr < local_base ||
      local_addr + size > local_base + local_mr->len) {
    UCCL_LOG(ERROR) << "CXI local range is outside registered MR";
    return -1;
  }
  if (fifo_item.addr < metadata.base ||
      fifo_item.addr + size > metadata.base + metadata.len) {
    UCCL_LOG(ERROR) << "CXI remote range is outside advertised MR";
    return -1;
  }

  Peer peer;
  {
    std::shared_lock<std::shared_mutex> lock(peer_mutex_);
    auto it = peers_.find(conn.peer_id);
    if (it == peers_.end()) {
      UCCL_LOG(ERROR) << "Unknown CXI peer_id " << conn.peer_id;
      return -1;
    }
    peer = it->second;
  }

  size_t const peer_lanes = peer.addrs.size();
  if (peer_lanes == 0) {
    UCCL_LOG(ERROR) << "CXI peer_id " << conn.peer_id << " has no lanes";
    return -1;
  }

  auto op = std::make_unique<OpContext>();
  op->id = static_cast<uint32_t>(
      next_request_id_.fetch_add(1, std::memory_order_relaxed));
  uint64_t const remote_offset = fifo_item.addr - metadata.base;

  // The bandwidth cap is per target endpoint, so a message above 512 KiB is
  // split as on the verbs path (ChunkSplitStrategy: at most 16 pieces of
  // equal size). Each piece is rounded up to kCxiPieceAlign, so only the last
  // piece is shorter, and is at most max_piece_bytes_. The pieces go to the
  // peer lanes in turn, starting at the lane chosen by request id; a message
  // that is not split is one piece on that lane.
  size_t piece = size;
  size_t const count = ChunkSplitStrategy::get_message_chunk_count(size);
  if (peer_lanes > 1 && count > 1) {
    piece = (size + count - 1) / count;
    piece = (piece + kCxiPieceAlign - 1) / kCxiPieceAlign * kCxiPieceAlign;
  }
  piece = std::min(piece, max_piece_bytes_);
  size_t const num_pieces = piece >= size ? 1 : (size + piece - 1) / piece;
  op->ctxs.resize(num_pieces);
  op->lanes.resize(num_pieces);
  op->remaining = static_cast<int>(num_pieces);
  for (size_t i = 0; i < num_pieces; ++i) {
    set_fi_context_owner(op->ctxs[i], op.get());
    op->lanes[i] = static_cast<int>((op->id + i) % peer_lanes);
  }

  void* const desc = fi_mr_desc(local_mr->mrs[0]);
  auto post_piece = [&](size_t i) -> ssize_t {
    size_t const offset = i * piece;
    size_t const len = std::min(piece, size - offset);
    int const lane = op->lanes[i];
    char* const buf = static_cast<char*>(local) + offset;
    if (is_read) {
      return fi_read(eps_[0], buf, len, desc, peer.addrs[lane],
                     remote_offset + offset, metadata.keys[lane], &op->ctxs[i]);
    }
    return fi_write(eps_[0], buf, len, desc, peer.addrs[lane],
                    remote_offset + offset, metadata.keys[lane], &op->ctxs[i]);
  };

  ssize_t rc = 0;
  size_t failed_piece = 0;
  std::string post_error_cq_errors;
  {
    std::lock_guard<std::mutex> fabric_lock(fabric_mutex_);
    for (size_t i = 0; i < num_pieces; ++i) {
      rc = post_piece(i);
      // After the first piece, do not return UCCL_POST_TRANSIENT: the caller
      // would post the whole request again and free this context while
      // earlier pieces are in flight. Drive the CQ and retry the piece.
      while (rc == -FI_EAGAIN && i > 0) {
        poll_cq_locked();
        rc = post_piece(i);
      }
      // FI_EAGAIN means the provider is momentarily out of posting resources;
      // drive the CQ, yield, and let the caller retry. Any other non-zero rc
      // (notably FI_EIO) is a hard failure: fall through and fail closed.
      if (rc == -FI_EAGAIN) {
        poll_cq_locked();
        std::this_thread::yield();
        return UCCL_POST_TRANSIENT;
      }
      if (rc != 0) {
        failed_piece = i;
        break;
      }
    }
    // Drain only on the fatal path (FI_EIO aborts below): draining consumes
    // CQ error entries that may belong to other inflight ops, which would
    // otherwise never be marked failed by poll_cq_locked().
    if (rc == -FI_EIO) {
      post_error_cq_errors = drain_cq_errors(cq_);
    } else if (rc != 0 && failed_piece > 0) {
      // The pieces already posted point to this context: wait until they
      // complete before the context is freed.
      {
        std::lock_guard<std::mutex> lock(op_mutex_);
        op->remaining -= static_cast<int>(num_pieces - failed_piece);
      }
      for (;;) {
        poll_cq_locked();
        std::lock_guard<std::mutex> lock(op_mutex_);
        if (op->remaining <= 0) break;
      }
    }
  }
  if (rc != 0) {
    size_t const offset = failed_piece * piece;
    int const lane = op->lanes[failed_piece];
    std::ostringstream os;
    os << (is_read ? "fi_read" : "fi_write") << " failed: " << fi_strerror(-rc)
       << " rc=" << rc << " peer_id=" << conn.peer_id
       << " peer_gpu=" << peer.gpu_index << " size=" << size
       << " piece=" << failed_piece << "/" << num_pieces << " lane=" << lane
       << " local=0x" << std::hex << local_addr << " local_base=0x"
       << local_base << " local_len=0x" << local_mr->len << " remote=0x"
       << fifo_item.addr << " remote_base=0x" << metadata.base
       << " remote_len=0x" << metadata.len << " remote_offset=0x"
       << remote_offset + offset << " key=0x" << metadata.keys[lane]
       << std::dec;
    if (!post_error_cq_errors.empty()) {
      os << " pending_cq_errors=[" << post_error_cq_errors << "]";
    }
    // FI_EIO indicates a fatal transport error (e.g. an unrecoverable link or
    // memory-registration fault); crash loudly rather than limp on with a
    // half-broken KV path.
    if (rc == -FI_EIO) {
      UCCL_LOG(FATAL) << os.str();
    } else {
      UCCL_LOG(ERROR) << os.str();
    }
    return -1;
  }

  uint32_t const request_id = static_cast<uint32_t>(op->id);
  {
    std::lock_guard<std::mutex> lock(op_mutex_);
    inflight_ops_[request_id] = std::move(op);
  }

  ureq->type = is_read ? ReqType::ReqRead : ReqType::ReqWrite;
  ureq->peer_id = conn.peer_id;
  ureq->engine_idx = request_id;
  return static_cast<int>(request_id);
}

void CxiEndpoint::poll_cq() {
  std::lock_guard<std::mutex> fabric_lock(fabric_mutex_);
  poll_cq_locked();
}

void CxiEndpoint::poll_cq_locked() {
  if (!cq_) return;

  for (;;) {
    fi_cq_entry entries[16]{};
    ssize_t rc = fi_cq_read(cq_, entries, 16);
    if (rc > 0) {
      std::lock_guard<std::mutex> lock(op_mutex_);
      for (ssize_t i = 0; i < rc; ++i) {
        auto* ctx = static_cast<OpContext*>(
            get_fi_context_owner(entries[i].op_context));
        if (ctx && --ctx->remaining <= 0) ctx->done = true;
      }
      continue;
    }
    if (rc == -FI_EAGAIN) return;
    if (rc == -FI_EAVAIL) {
      fi_cq_err_entry err{};
      ssize_t err_rc = fi_cq_readerr(cq_, &err, 0);
      std::string error_message =
          err_rc >= 0
              ? cq_error_string(cq_, err)
              : std::string("fi_cq_readerr failed: ") + fi_strerror(-err_rc);
      {
        std::lock_guard<std::mutex> lock(op_mutex_);
        auto* ctx =
            static_cast<OpContext*>(get_fi_context_owner(err.op_context));
        if (ctx) {
          auto const piece =
              static_cast<fi_context2*>(err.op_context) - ctx->ctxs.data();
          if (piece >= 0 && static_cast<size_t>(piece) < ctx->lanes.size()) {
            error_message += " (lane " + std::to_string(ctx->lanes[piece]) +
                             ", piece " + std::to_string(piece) + ")";
          }
          --ctx->remaining;
          ctx->done = true;
          ctx->failed = true;
          ctx->err = err.err;
          ctx->prov_errno = err.prov_errno;
          ctx->error_message = error_message;
        }
      }
      UCCL_LOG(ERROR) << "CXI CQ error: " << error_message;
      continue;
    }

    UCCL_LOG(ERROR) << "fi_cq_read failed: " << fi_strerror(-rc);
    return;
  }
}

bool CxiEndpoint::check_send_complete_once(uint64_t peer_id,
                                           int64_t request_id) {
  (void)peer_id;
  poll_cq();

  std::lock_guard<std::mutex> lock(op_mutex_);
  auto it = inflight_ops_.find(request_id);
  if (it == inflight_ops_.end()) return true;
  if (!it->second->done) return false;
  if (it->second->failed) {
    UCCL_LOG(ERROR) << "CXI transfer failed: request_id=" << request_id
                    << " peer_id=" << peer_id << " err=" << it->second->err
                    << " prov_errno=" << it->second->prov_errno << " "
                    << it->second->error_message;
    std::abort();
  }
  inflight_ops_.erase(it);
  return true;
}

void CxiEndpoint::send_routine() { poll_cq(); }

void CxiEndpoint::recv_routine() { poll_cq(); }

int CxiEndpoint::send_notification(uint64_t peer_id,
                                   NotifyMsg const& notification) const {
  std::string conn_key = get_oob_conn_key(peer_id);
  if (conn_key.empty() || !oob_client_) return -1;

  std::string payload = serialize_notify_msg(notification);
  return oob_client_->send_meta(conn_key, payload) ? 0 : -1;
}

void encode_cxi_fifo_metadata(CxiMemoryRegion const& region, FifoItem& item) {
  CxiFifoMetadata metadata;
  metadata.base = reinterpret_cast<uint64_t>(region.addr);
  metadata.len = region.len;
  for (size_t i = 0;
       i < region.keys.size() && i < static_cast<size_t>(kCxiMaxLanes); ++i) {
    metadata.keys[i] = region.keys[i];
  }
  std::memset(item.padding, 0, sizeof(item.padding));
  std::memcpy(item.padding, &metadata, sizeof(metadata));
}

bool decode_cxi_fifo_metadata(FifoItem const& item, CxiFifoMetadata& metadata) {
  std::memcpy(&metadata, item.padding, sizeof(metadata));
  return metadata.base != 0 && metadata.len != 0;
}
