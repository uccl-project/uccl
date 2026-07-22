#include "cxi_transport.hpp"
#include "util/gpu_rt.h"

#ifdef USE_LIBFABRIC_CXI
#include <rdma/fi_atomic.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/uio.h>

CxiTransport::~CxiTransport() { destroy(); }

void CxiTransport::unavailable() {
  throw std::runtime_error(
      "CxiTransport requires building with USE_LIBFABRIC_CXI=1");
}

#ifndef USE_LIBFABRIC_CXI

void CxiTransport::init(ProxyCtx&) { unavailable(); }
void CxiTransport::register_main_buffer(void*, size_t, int) { unavailable(); }
void CxiTransport::register_atomic_buffer(void*, size_t, int) { unavailable(); }
LocalConnInfo CxiTransport::local_info() const { unavailable(); }
void CxiTransport::connect_peer(int, RemoteConnInfo const&) { unavailable(); }
void CxiTransport::post_write(int, uint64_t, uint64_t, uint64_t, uint32_t,
                              bool) {
  unavailable();
}
void CxiTransport::post_atomic_add(int, uint64_t, uint64_t, int64_t, bool) {
  unavailable();
}
void CxiTransport::post_barrier_atomic_add(int, uint64_t, size_t, uint64_t) {
  unavailable();
}
uint64_t CxiTransport::load_barrier_word(size_t) const { unavailable(); }
int CxiTransport::poll(TransportCompletion*, int) { unavailable(); }
void CxiTransport::destroy() {}

#else

namespace {

void check_fi(int rc, char const* what) {
  if (rc < 0) {
    throw std::runtime_error(std::string(what) +
                             " failed: " + fi_strerror(-rc));
  }
}

void check_gpu(gpuError_t rc, char const* what) {
  if (rc != gpuSuccess) {
    throw std::runtime_error(std::string(what) +
                             " failed: " + gpuGetErrorString(rc));
  }
}

fid_mr* register_cuda_mr(fid_domain* domain, fid_ep* ep, void* ptr, size_t len,
                         int cuda_device, char const* label) {
  int old_device = 0;
  check_gpu(gpuGetDevice(&old_device), "gpuGetDevice");
  check_gpu(gpuSetDevice(cuda_device), "gpuSetDevice");

  iovec iov = {};
  iov.iov_base = ptr;
  iov.iov_len = len;

  fi_mr_attr attr = {};
  attr.mr_iov = &iov;
  attr.iov_count = 1;
  attr.access =
      FI_SEND | FI_RECV | FI_WRITE | FI_READ | FI_REMOTE_WRITE | FI_REMOTE_READ;
  attr.iface = FI_HMEM_CUDA;
  attr.device.cuda = static_cast<uint64_t>(cuda_device);

  fid_mr* mr = nullptr;
  int rc = fi_mr_regattr(domain, &attr, 0, &mr);
  gpuError_t restore_rc = gpuSetDevice(old_device);
  check_fi(rc, label);
  check_gpu(restore_rc, "gpuSetDevice(restore)");

  check_fi(fi_mr_bind(mr, &ep->fid, 0), "fi_mr_bind");
  check_fi(fi_control(&mr->fid, FI_ENABLE, nullptr), "fi_control(FI_ENABLE)");
  return mr;
}

fid_mr* register_host_mr(fid_domain* domain, fid_ep* ep, void* ptr, size_t len,
                         char const* label) {
  iovec iov = {};
  iov.iov_base = ptr;
  iov.iov_len = len;

  fi_mr_attr attr = {};
  attr.mr_iov = &iov;
  attr.iov_count = 1;
  attr.access =
      FI_SEND | FI_RECV | FI_WRITE | FI_READ | FI_REMOTE_WRITE | FI_REMOTE_READ;

  fid_mr* mr = nullptr;
  check_fi(fi_mr_regattr(domain, &attr, 0, &mr), label);
  check_fi(fi_mr_bind(mr, &ep->fid, 0), "fi_mr_bind");
  check_fi(fi_control(&mr->fid, FI_ENABLE, nullptr), "fi_control(FI_ENABLE)");
  return mr;
}

fi_info* select_cxi_info(fi_info* infos, int local_rank) {
  char target[32];
  std::snprintf(target, sizeof(target), "cxi%d", local_rank);

  for (fi_info* cur = infos; cur; cur = cur->next) {
    char const* name = (cur->domain_attr && cur->domain_attr->name)
                           ? cur->domain_attr->name
                           : "";
    if (std::strcmp(name, target) == 0) {
      std::fprintf(stderr, "Selected CXI domain %s for local_rank %d\n", name,
                   local_rank);
      return cur;
    }
  }

  std::string available;
  for (fi_info* cur = infos; cur; cur = cur->next) {
    char const* name = (cur->domain_attr && cur->domain_attr->name)
                           ? cur->domain_attr->name
                           : "<unknown>";
    if (!available.empty()) available += " ";
    available += name;
  }
  throw std::runtime_error("Could not find required CXI domain " +
                           std::string(target) + " for local_rank " +
                           std::to_string(local_rank) +
                           "; available domains: " + available);
}

}  // namespace

void CxiTransport::init(ProxyCtx& ctx) {
  ctx_ = &ctx;

  fi_info* hints = fi_allocinfo();
  if (!hints) throw std::runtime_error("fi_allocinfo failed");

  hints->caps = FI_RMA | FI_WRITE | FI_ATOMIC | FI_FENCE | FI_REMOTE_WRITE |
                FI_REMOTE_READ | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
  hints->mode = FI_CONTEXT;
  hints->ep_attr->type = FI_EP_RDM;
  hints->fabric_attr->prov_name = strdup("cxi");
  hints->domain_attr->mr_mode =
      FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
  hints->domain_attr->threading = FI_THREAD_DOMAIN;

  fi_info* info = nullptr;
  int rc = fi_getinfo(FI_VERSION(1, 15), nullptr, nullptr, 0, hints, &info);
  fi_freeinfo(hints);
  check_fi(rc, "fi_getinfo(cxi)");

  try {
    fi_info* selected = select_cxi_info(info, ctx.local_rank);

    check_fi(fi_fabric(selected->fabric_attr, &fabric_, nullptr), "fi_fabric");
    check_fi(fi_domain(fabric_, selected, &domain_, nullptr), "fi_domain");

    fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_DATA;
    cq_attr.size = 131072;
    check_fi(fi_cq_open(domain_, &cq_attr, &cq_, nullptr), "fi_cq_open");

    fi_av_attr av_attr = {};
    av_attr.type = FI_AV_MAP;
    check_fi(fi_av_open(domain_, &av_attr, &av_, nullptr), "fi_av_open");

    check_fi(fi_endpoint(domain_, selected, &ep_, nullptr), "fi_endpoint");
    check_fi(fi_ep_bind(ep_, &cq_->fid, FI_TRANSMIT | FI_RECV),
             "fi_ep_bind(cq)");
    check_fi(fi_ep_bind(ep_, &av_->fid, 0), "fi_ep_bind(av)");
    check_fi(fi_enable(ep_), "fi_enable");
  } catch (...) {
    fi_freeinfo(info);
    destroy();
    throw;
  }

  fi_freeinfo(info);

  size_t ep_name_len = sizeof(local_info_.cxi_ep_name);
  check_fi(fi_getname(&ep_->fid, local_info_.cxi_ep_name, &ep_name_len),
           "fi_getname");
  local_info_.cxi_ep_name_len = static_cast<uint32_t>(ep_name_len);

  atomic_operands_.assign(ProxyCtx::kMaxAtomicOps, 0);
  atomic_operand_used_.assign(ProxyCtx::kMaxAtomicOps, 0);
  atomic_operand_mr_ =
      register_host_mr(domain_, ep_, atomic_operands_.data(),
                       atomic_operands_.size() * sizeof(atomic_operands_[0]),
                       "fi_mr_regattr(atomic operands)");
  barrier_words_.assign(8, 0);
  barrier_mr_ =
      register_host_mr(domain_, ep_, barrier_words_.data(),
                       barrier_words_.size() * sizeof(barrier_words_[0]),
                       "fi_mr_regattr(barrier words)");
  local_info_.cxi_barrier_mr_key = fi_mr_key(barrier_mr_);

  std::fprintf(stderr, "[CXI] libfabric/CXI endpoint initialized\n");
}

void CxiTransport::register_main_buffer(void* ptr, size_t len,
                                        int cuda_device) {
  if (!domain_ || !ep_) {
    throw std::runtime_error("CxiTransport::init must run before MR setup");
  }
  main_mr_ = register_cuda_mr(domain_, ep_, ptr, len, cuda_device,
                              "fi_mr_regattr(main)");
  main_buffer_ = ptr;
  main_buffer_len_ = len;
  local_info_.addr = reinterpret_cast<uintptr_t>(ptr);
  local_info_.len = len;
  local_info_.cxi_main_mr_key = fi_mr_key(main_mr_);
  local_info_.rkey = static_cast<uint32_t>(local_info_.cxi_main_mr_key);
}

void CxiTransport::register_atomic_buffer(void* ptr, size_t len,
                                          int cuda_device) {
  if (!domain_ || !ep_) {
    throw std::runtime_error("CxiTransport::init must run before MR setup");
  }
  atomic_mr_ = register_cuda_mr(domain_, ep_, ptr, len, cuda_device,
                                "fi_mr_regattr(atomic)");
  atomic_buffer_ = ptr;
  atomic_buffer_len_ = len;
  local_info_.atomic_buffer_addr = reinterpret_cast<uintptr_t>(ptr);
  local_info_.atomic_buffer_len = len;
  local_info_.cxi_atomic_mr_key = fi_mr_key(atomic_mr_);
  local_info_.atomic_buffer_rkey =
      static_cast<uint32_t>(local_info_.cxi_atomic_mr_key);
}

LocalConnInfo CxiTransport::local_info() const { return local_info_; }

void CxiTransport::connect_peer(int peer, RemoteConnInfo const& remote) {
  if (!av_) throw std::runtime_error("CxiTransport::init must run first");
  if (remote.cxi_ep_name_len == 0 ||
      remote.cxi_ep_name_len > RemoteConnInfo::kMaxCxiEndpointName) {
    throw std::runtime_error("invalid CXI endpoint address in remote info");
  }

  if (peer >= static_cast<int>(peer_addrs_.size())) {
    peer_addrs_.resize(peer + 1, FI_ADDR_UNSPEC);
    peer_infos_.resize(peer + 1);
  }

  fi_addr_t addr = FI_ADDR_UNSPEC;
  int rc = fi_av_insert(av_, const_cast<uint8_t*>(remote.cxi_ep_name), 1, &addr,
                        0, nullptr);
  if (rc != 1) {
    if (rc < 0) check_fi(rc, "fi_av_insert");
    throw std::runtime_error("fi_av_insert inserted no CXI peer address");
  }
  peer_addrs_[peer] = addr;
  peer_infos_[peer] = remote;
}

void CxiTransport::post_write(int dst_rank, uint64_t wr_id,
                              uint64_t local_offset, uint64_t remote_offset,
                              uint32_t bytes, bool) {
  if (dst_rank >= static_cast<int>(peer_addrs_.size()) ||
      peer_addrs_[dst_rank] == FI_ADDR_UNSPEC) {
    throw std::runtime_error("CXI peer is not connected");
  }
  if (!main_mr_ || !main_buffer_) {
    throw std::runtime_error("CXI main buffer MR is not registered");
  }
  if (local_offset + bytes > main_buffer_len_) {
    throw std::runtime_error("CXI write local range exceeds main buffer");
  }

  op_contexts_.push_back({});
  OpContext& op = op_contexts_.back();
  op.wr_id = wr_id;
  op.is_write = true;

  auto* local = static_cast<char*>(main_buffer_) + local_offset;
  ssize_t rc = fi_write(ep_, local, bytes, fi_mr_desc(main_mr_),
                        peer_addrs_[dst_rank], remote_offset,
                        peer_infos_[dst_rank].cxi_main_mr_key, &op.context);
  check_fi(static_cast<int>(rc), "fi_write");
}

void CxiTransport::post_atomic_add(int dst_rank, uint64_t wr_id,
                                   uint64_t remote_atomic_offset, int64_t value,
                                   bool fence) {
  if (dst_rank >= static_cast<int>(peer_addrs_.size()) ||
      peer_addrs_[dst_rank] == FI_ADDR_UNSPEC) {
    throw std::runtime_error("CXI peer is not connected");
  }
  if (!atomic_mr_) {
    throw std::runtime_error("CXI atomic buffer MR is not registered");
  }

  op_contexts_.push_back({});
  OpContext& op = op_contexts_.back();
  op.wr_id = wr_id;
  op.is_atomic = true;

  size_t slot = static_cast<size_t>(-1);
  for (size_t i = 0; i < atomic_operand_used_.size(); ++i) {
    if (!atomic_operand_used_[i]) {
      slot = i;
      atomic_operand_used_[i] = 1;
      break;
    }
  }
  if (slot == static_cast<size_t>(-1)) {
    op_contexts_.pop_back();
    throw std::runtime_error("no free CXI atomic operand slot");
  }
  op.atomic_operand_slot = slot;
  atomic_operands_[slot] = static_cast<uint64_t>(value);

  fi_ioc msg_iov = {};
  msg_iov.addr = &atomic_operands_[slot];
  msg_iov.count = 1;
  void* desc = fi_mr_desc(atomic_operand_mr_);

  fi_rma_ioc rma_iov = {};
  rma_iov.addr = remote_atomic_offset;
  rma_iov.count = 1;
  rma_iov.key = peer_infos_[dst_rank].cxi_atomic_mr_key;

  fi_msg_atomic msg = {};
  msg.msg_iov = &msg_iov;
  msg.desc = &desc;
  msg.iov_count = 1;
  msg.addr = peer_addrs_[dst_rank];
  msg.rma_iov = &rma_iov;
  msg.rma_iov_count = 1;
  msg.datatype = FI_UINT64;
  msg.op = FI_SUM;
  msg.context = &op.context;

  ssize_t rc = fi_atomicmsg(ep_, &msg, fence ? FI_FENCE : 0);
  check_fi(static_cast<int>(rc), "fi_atomicmsg");
}

void CxiTransport::post_barrier_atomic_add(int dst_rank, uint64_t wr_id,
                                           size_t slot, uint64_t value) {
  if (dst_rank >= static_cast<int>(peer_addrs_.size()) ||
      peer_addrs_[dst_rank] == FI_ADDR_UNSPEC) {
    throw std::runtime_error("CXI peer is not connected");
  }
  if (peer_infos_[dst_rank].cxi_barrier_mr_key == 0) {
    throw std::runtime_error("CXI barrier MR is not registered for peer");
  }

  op_contexts_.push_back({});
  OpContext& op = op_contexts_.back();
  op.wr_id = wr_id;
  op.is_atomic = true;

  size_t operand_slot = static_cast<size_t>(-1);
  for (size_t i = 0; i < atomic_operand_used_.size(); ++i) {
    if (!atomic_operand_used_[i]) {
      operand_slot = i;
      atomic_operand_used_[i] = 1;
      break;
    }
  }
  if (operand_slot == static_cast<size_t>(-1)) {
    op_contexts_.pop_back();
    throw std::runtime_error("no free CXI atomic operand slot");
  }
  op.atomic_operand_slot = operand_slot;
  atomic_operands_[operand_slot] = value;

  fi_ioc msg_iov = {};
  msg_iov.addr = &atomic_operands_[operand_slot];
  msg_iov.count = 1;
  void* desc = fi_mr_desc(atomic_operand_mr_);

  fi_rma_ioc rma_iov = {};
  rma_iov.addr = slot * sizeof(uint64_t);
  rma_iov.count = 1;
  rma_iov.key = peer_infos_[dst_rank].cxi_barrier_mr_key;

  fi_msg_atomic msg = {};
  msg.msg_iov = &msg_iov;
  msg.desc = &desc;
  msg.iov_count = 1;
  msg.addr = peer_addrs_[dst_rank];
  msg.rma_iov = &rma_iov;
  msg.rma_iov_count = 1;
  msg.datatype = FI_UINT64;
  msg.op = FI_SUM;
  msg.context = &op.context;

  ssize_t rc = fi_atomicmsg(ep_, &msg, FI_FENCE);
  check_fi(static_cast<int>(rc), "fi_atomicmsg(barrier)");
}

uint64_t CxiTransport::load_barrier_word(size_t slot) const {
  if (slot >= barrier_words_.size()) {
    throw std::runtime_error("CXI barrier slot out of range");
  }
  return __atomic_load_n(&barrier_words_[slot], __ATOMIC_ACQUIRE);
}

int CxiTransport::poll(TransportCompletion* out, int max) {
  if (!cq_ || max <= 0) return 0;

  std::vector<fi_cq_data_entry> entries(max);
  ssize_t rc = fi_cq_read(cq_, entries.data(), max);
  if (rc == -FI_EAGAIN) return 0;
  if (rc < 0) {
    fi_cq_err_entry err = {};
    ssize_t err_rc = fi_cq_readerr(cq_, &err, 0);
    if (err_rc >= 0) {
      throw std::runtime_error(
          std::string("CXI CQ error: ") +
          fi_cq_strerror(cq_, err.prov_errno, err.err_data, nullptr, 0));
    }
    check_fi(static_cast<int>(rc), "fi_cq_read");
  }

  for (ssize_t i = 0; i < rc; ++i) {
    auto* op = static_cast<OpContext*>(entries[i].op_context);
    out[i].wr_id = op ? op->wr_id : 0;
    out[i].data = entries[i].data;
    out[i].is_write = op ? op->is_write : ((entries[i].flags & FI_WRITE) != 0);
    out[i].is_atomic =
        op ? op->is_atomic : ((entries[i].flags & FI_ATOMIC) != 0);
    out[i].is_recv_data = (entries[i].flags & FI_REMOTE_CQ_DATA) != 0;
    out[i].status = 0;

    if (op) {
      if (op->atomic_operand_slot != static_cast<size_t>(-1) &&
          op->atomic_operand_slot < atomic_operand_used_.size()) {
        atomic_operand_used_[op->atomic_operand_slot] = 0;
      }
      for (auto it = op_contexts_.begin(); it != op_contexts_.end(); ++it) {
        if (&*it == op) {
          op_contexts_.erase(it);
          break;
        }
      }
    }
  }
  return static_cast<int>(rc);
}

void CxiTransport::destroy() {
  if (barrier_mr_) {
    fi_close(&barrier_mr_->fid);
    barrier_mr_ = nullptr;
  }
  if (atomic_operand_mr_) {
    fi_close(&atomic_operand_mr_->fid);
    atomic_operand_mr_ = nullptr;
  }
  if (atomic_mr_) {
    fi_close(&atomic_mr_->fid);
    atomic_mr_ = nullptr;
  }
  if (main_mr_) {
    fi_close(&main_mr_->fid);
    main_mr_ = nullptr;
  }
  if (ep_) {
    fi_close(&ep_->fid);
    ep_ = nullptr;
  }
  if (av_) {
    fi_close(&av_->fid);
    av_ = nullptr;
  }
  if (cq_) {
    fi_close(&cq_->fid);
    cq_ = nullptr;
  }
  if (domain_) {
    fi_close(&domain_->fid);
    domain_ = nullptr;
  }
  if (fabric_) {
    fi_close(&fabric_->fid);
    fabric_ = nullptr;
  }
}

#endif
