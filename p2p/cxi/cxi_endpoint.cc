#include "cxi_endpoint.h"
#include "engine.h"
#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {

constexpr uint32_t kCxiEndpointInfoMagic = 0x43584950;  // "CXIP"
constexpr uint32_t kCxiEndpointInfoVersion = 1;

struct CxiEndpointInfoHeader {
  uint32_t magic = kCxiEndpointInfoMagic;
  uint32_t version = kCxiEndpointInfoVersion;
  int32_t gpu_index = -1;
  uint16_t oob_port = 0;
  uint16_t ep_name_size = 0;
};

void check_fi(char const* what, int ret) {
  if (ret != 0) {
    throw std::runtime_error(std::string(what) + ": " + fi_strerror(-ret));
  }
}

fi_threading threading_hint() {
  char const* value = std::getenv("UCCL_CXI_THREADING");
  if (!value || std::strcmp(value, "endpoint") == 0) return FI_THREAD_ENDPOINT;
  if (std::strcmp(value, "completion") == 0) return FI_THREAD_COMPLETION;
  if (std::strcmp(value, "domain") == 0) return FI_THREAD_DOMAIN;
  if (std::strcmp(value, "fid") == 0) return FI_THREAD_FID;
  if (std::strcmp(value, "safe") == 0) return FI_THREAD_SAFE;
  if (std::strcmp(value, "unspec") == 0) return FI_THREAD_UNSPEC;
  throw std::runtime_error(std::string("Invalid UCCL_CXI_THREADING=") + value);
}

int cxi_device_index_for_gpu(int gpu_index) {
  char const* value = std::getenv("UCCL_CXI_DEVICE_INDEX");
  if (value && value[0] != '\0') return std::atoi(value);
  if (gpu_index < 0 || static_cast<uint32_t>(gpu_index) == INVALID_GPU) {
    return -1;
  }
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

bool is_cuda_pointer(void* ptr, int& cuda_device) {
  cudaPointerAttributes attrs{};
  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  if (attrs.type == cudaMemoryTypeDevice ||
      attrs.type == cudaMemoryTypeManaged) {
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

  if (av_) fi_close(&av_->fid);
  if (cq_) fi_close(&cq_->fid);
  if (ep_) fi_close(&ep_->fid);
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
                 << " on domain " << cxi_domain_name_for_gpu(gpu_index);
  return true;
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
    info_->tx_attr->op_flags |= FI_DELIVERY_COMPLETE;
    UCCL_LOG(INFO) << "CXI FI_DELIVERY_COMPLETE enabled";

    check_fi("fi_fabric", fi_fabric(info_->fabric_attr, &fabric_, nullptr));
    check_fi("fi_domain", fi_domain(fabric_, info_, &domain_, nullptr));
    check_fi("fi_endpoint", fi_endpoint(domain_, info_, &ep_, nullptr));

    fi_cq_attr cq_attr{};
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.size = cxi_size_env("UCCL_CXI_CQ_SIZE", 8192);
    check_fi("fi_cq_open", fi_cq_open(domain_, &cq_attr, &cq_, nullptr));
    check_fi("fi_ep_bind(cq)",
             fi_ep_bind(ep_, &cq_->fid, FI_TRANSMIT | FI_RECV));

    fi_av_attr av_attr{};
    av_attr.type = FI_AV_TABLE;
    check_fi("fi_av_open", fi_av_open(domain_, &av_attr, &av_, nullptr));
    check_fi("fi_ep_bind(av)", fi_ep_bind(ep_, &av_->fid, 0));

#ifdef FI_OPT_CUDA_API_PERMITTED
    bool cuda_api_permitted = false;
    check_fi("fi_setopt(FI_OPT_CUDA_API_PERMITTED)",
             fi_setopt(&ep_->fid, FI_OPT_ENDPOINT, FI_OPT_CUDA_API_PERMITTED,
                       &cuda_api_permitted, sizeof(cuda_api_permitted)));
#endif

    check_fi("fi_enable", fi_enable(ep_));
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
  if (!fabric_initialized_ || !ep_) {
    throw std::runtime_error("CXI endpoint not initialized");
  }

  EndpointInfo info;
  info.oob_port = oob_server_ ? oob_server_->get_port() : 0;
  info.gpu_index = gpu_index_;
  info.ep_name.resize(512);
  size_t len = info.ep_name.size();
  check_fi("fi_getname", fi_getname(&ep_->fid, info.ep_name.data(), &len));
  info.ep_name.resize(len);
  return info;
}

std::string CxiEndpoint::serialize_endpoint_info(
    EndpointInfo const& info) const {
  if (info.ep_name.size() > std::numeric_limits<uint16_t>::max()) {
    throw std::runtime_error("CXI endpoint name too large");
  }

  CxiEndpointInfoHeader header;
  header.gpu_index = info.gpu_index;
  header.oob_port = info.oob_port;
  header.ep_name_size = static_cast<uint16_t>(info.ep_name.size());

  std::string bytes(sizeof(header) + info.ep_name.size(), '\0');
  std::memcpy(bytes.data(), &header, sizeof(header));
  if (!info.ep_name.empty()) {
    std::memcpy(bytes.data() + sizeof(header), info.ep_name.data(),
                info.ep_name.size());
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
  if (header.magic != kCxiEndpointInfoMagic ||
      header.version != kCxiEndpointInfoVersion) {
    throw std::runtime_error("Invalid CXI endpoint metadata");
  }
  if (bytes.size() != sizeof(header) + header.ep_name_size) {
    throw std::runtime_error("CXI endpoint metadata length mismatch");
  }

  EndpointInfo info;
  info.oob_port = header.oob_port;
  info.gpu_index = header.gpu_index;
  info.ep_name.resize(header.ep_name_size);
  if (header.ep_name_size > 0) {
    std::memcpy(info.ep_name.data(), bytes.data() + sizeof(header),
                header.ep_name_size);
  }
  return info;
}

fi_addr_t CxiEndpoint::insert_peer(EndpointInfo const& info) {
  if (!av_) throw std::runtime_error("CXI AV missing");
  if (info.ep_name.empty()) {
    throw std::runtime_error("CXI peer endpoint name is empty");
  }

  fi_addr_t addr = FI_ADDR_UNSPEC;
  int ret = fi_av_insert(av_, info.ep_name.data(), 1, &addr, 0, nullptr);
  if (ret != 1) check_fi("fi_av_insert", ret < 0 ? ret : -FI_EINVAL);
  return addr;
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
        promise->set_value(deserialize_endpoint_info(response));
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

  EndpointInfo remote_info = future.get();
  fi_addr_t addr = insert_peer(remote_info);
  {
    std::unique_lock<std::shared_mutex> lock(peer_mutex_);
    peers_[peer_id] = Peer{addr, remote_ip, remote_port, remote_gpuidx};
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
  if (input.size() >= sizeof(NotifyMsg)) {
    NotifyMsg const* notify_msg =
        reinterpret_cast<NotifyMsg const*>(input.data());
    if (notify_msg->magic == NOTIFY_MSG_MAGIC) {
      std::lock_guard<std::mutex> lock(notify_mutex);
      notify_list.push_back(*notify_msg);
      output = "";
      return;
    }
  }

  EndpointInfo peer_info = deserialize_endpoint_info(input);
  fi_addr_t addr = insert_peer(peer_info);
  uint64_t peer_id = next_recv_peer_id_.fetch_add(1, std::memory_order_relaxed);

  {
    std::unique_lock<std::shared_mutex> lock(peer_mutex_);
    peers_[peer_id] = Peer{addr, client_ip, static_cast<uint16_t>(client_port),
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
  if (is_cuda_pointer(data, cuda_device)) {
    iovec iov{};
    iov.iov_base = data;
    iov.iov_len = len;

    fi_mr_attr mr_attr{};
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_WRITE |
                     FI_REMOTE_READ;
    mr_attr.iface = FI_HMEM_CUDA;
    mr_attr.device.cuda = cuda_device;

    int ret = fi_mr_regattr(domain_, &mr_attr, 0, &out->mr);
    if (ret != 0) {
      UCCL_LOG(ERROR) << "fi_mr_regattr(cuda) failed: " << fi_strerror(-ret);
      return -1;
    }
  } else {
    int ret = fi_mr_reg(domain_, data, len,
                        FI_SEND | FI_RECV | FI_READ | FI_WRITE |
                            FI_REMOTE_WRITE | FI_REMOTE_READ,
                        0, 0, 0, &out->mr, nullptr);
    if (ret != 0) {
      UCCL_LOG(ERROR) << "fi_mr_reg(host) failed: " << fi_strerror(-ret);
      return -1;
    }
  }

  if (info_->domain_attr->mr_mode & FI_MR_ENDPOINT) {
    int ret = fi_mr_bind(out->mr, &ep_->fid, 0);
    if (ret != 0) {
      UCCL_LOG(ERROR) << "fi_mr_bind(ep) failed: " << fi_strerror(-ret);
      fi_close(&out->mr->fid);
      return -1;
    }
    ret = fi_mr_enable(out->mr);
    if (ret != 0) {
      UCCL_LOG(ERROR) << "fi_mr_enable failed: " << fi_strerror(-ret);
      fi_close(&out->mr->fid);
      return -1;
    }
  }

  out->key = fi_mr_key(out->mr);
  region = out;
  return 0;
}

void CxiEndpoint::uccl_deregmr(std::shared_ptr<CxiMemoryRegion> const& region) {
  if (!region || !region->mr) return;
  fid_mr* mr = region->mr;
  int ret = fi_close(&mr->fid);
  if (ret != 0) {
    UCCL_LOG(ERROR) << "fi_close(mr) failed: " << fi_strerror(-ret);
    return;
  }
  region->mr = nullptr;
  region->key = 0;
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
  if (!local_mr || !local_mr->mr || !local || !ureq) return -1;

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

  auto op = std::make_unique<OpContext>();
  op->id = static_cast<uint32_t>(
      next_request_id_.fetch_add(1, std::memory_order_relaxed));
  set_fi_context_owner(op->ctx, op.get());
  uint64_t const remote_offset = fifo_item.addr - metadata.base;

  ssize_t rc;
  {
    std::lock_guard<std::mutex> fabric_lock(fabric_mutex_);
    if (is_read) {
      rc = fi_read(ep_, local, size, fi_mr_desc(local_mr->mr), peer.addr,
                   remote_offset, metadata.key, &op->ctx);
    } else {
      rc = fi_write(ep_, local, size, fi_mr_desc(local_mr->mr), peer.addr,
                    remote_offset, metadata.key, &op->ctx);
    }

    if (rc == -FI_EAGAIN || rc == -FI_EIO) {
      poll_cq_locked();
      std::this_thread::yield();
      return UCCL_POST_TRANSIENT;
    }
  }
  if (rc != 0) {
    std::ostringstream os;
    os << (is_read ? "fi_read" : "fi_write") << " failed: " << fi_strerror(-rc)
       << " rc=" << rc << " peer_id=" << conn.peer_id
       << " peer_gpu=" << peer.gpu_index << " size=" << size << " local=0x"
       << std::hex << local_addr << " local_base=0x" << local_base
       << " local_len=0x" << local_mr->len << " remote=0x" << fifo_item.addr
       << " remote_base=0x" << metadata.base << " remote_len=0x" << metadata.len
       << " remote_offset=0x" << remote_offset << " key=0x" << metadata.key
       << std::dec;
    UCCL_LOG(ERROR) << os.str();
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
        if (ctx) ctx->done = true;
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

  std::string payload(reinterpret_cast<char const*>(&notification),
                      sizeof(NotifyMsg));
  return oob_client_->send_meta(conn_key, payload) ? sizeof(NotifyMsg) : -1;
}

void encode_cxi_fifo_metadata(CxiMemoryRegion const& region, FifoItem& item) {
  CxiFifoMetadata metadata;
  metadata.base = reinterpret_cast<uint64_t>(region.addr);
  metadata.key = region.key;
  metadata.len = region.len;
  std::memset(item.padding, 0, sizeof(item.padding));
  std::memcpy(item.padding, &metadata, sizeof(metadata));
}

bool decode_cxi_fifo_metadata(FifoItem const& item, CxiFifoMetadata& metadata) {
  std::memcpy(&metadata, item.padding, sizeof(metadata));
  return metadata.base != 0 && metadata.len != 0;
}
