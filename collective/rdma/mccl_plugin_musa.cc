#include "mccl_net_abi.h"
#include "transport.h"
#include "transport_config.h"
#include "util/debug.h"
#include "util/gpu_rt.h"
#include "util_rdma.h"

#include <arpa/inet.h>
#include <climits>
#include <cstdlib>
#include <limits>
#include <limits.h>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace uccl;

namespace {

constexpr char kPluginName[] = "UCCL-MCCL-RDMA";

class McclRequestPool : public BuffPool {
 public:
  McclRequestPool() : BuffPool(kMaxReq << 2, sizeof(ucclRequest), nullptr) {}
};

struct McclContext {
  std::shared_ptr<RDMAEndpoint> endpoint;
  uint64_t comm_id;
};

struct McclBaseComm {
  McclContext* context;
  int dev;
  ConnID conn_id;
  std::shared_ptr<McclRequestPool> request_pool;
};

enum class ConnectionState : uint8_t {
  kInit = 0,
  kConnecting,
  kConnected,
};

struct McclHandle {
  uint32_t ip_addr_u32;
  uint16_t listen_port;
  int remote_dev;
  int remote_gpuidx;
  ConnectionState state;
  McclBaseComm* pending_state;
};
static_assert(sizeof(McclHandle) <= MCCL_NET_HANDLE_MAXSIZE,
              "MCCL handle exceeds MCCL_NET_HANDLE_MAXSIZE");

struct McclListenComm {
  McclContext* context;
  int dev;
  int listen_fd;
  int gpuidx;
  ConnectionState state = ConnectionState::kInit;
  McclBaseComm accept_base;
  std::string remote_ip;
  int remote_dev = -1;
};

struct McclSendComm {
  McclBaseComm base;
};

struct McclRecvComm {
  McclBaseComm base;
  std::string remote_ip;
  int remote_dev;
};

std::mutex g_endpoint_mu;
std::shared_ptr<RDMAEndpoint> g_endpoint;
std::mutex g_musa_mr_mu;
std::unordered_map<int, bool> g_musa_mr_support;

std::shared_ptr<RDMAEndpoint> endpoint() {
  std::lock_guard<std::mutex> lock(g_endpoint_mu);
  return g_endpoint;
}

mcclResult_t initialize_device(McclContext* context, int* dev) {
  if (context == nullptr || dev == nullptr) return mcclInvalidArgument;

  int gpu = 0;
  if (gpuGetDevice(&gpu) != gpuSuccess) return mcclUnhandledMusaError;
  const int best_dev = context->endpoint->get_best_dev_idx(gpu);
  if (*dev != best_dev) *dev = best_dev;
  context->endpoint->initialize_engine_by_dev(*dev, false);
  return mcclSuccess;
}

bool can_register_musa(RDMAEndpoint* endpoint_ptr, int dev) {
  std::lock_guard<std::mutex> lock(g_musa_mr_mu);
  auto cached = g_musa_mr_support.find(dev);
  if (cached != g_musa_mr_support.end()) return cached->second;

  void* buffer = nullptr;
  bool supported = false;
  if (gpuMalloc(&buffer, 4096) == gpuSuccess) {
    Mhandle* mhandle = nullptr;
    supported = endpoint_ptr->uccl_regmr(dev, buffer, 4096, MCCL_PTR_MUSA,
                                         &mhandle) == 0 &&
                mhandle != nullptr;
    if (mhandle != nullptr) endpoint_ptr->uccl_deregmr(mhandle);
    gpuFree(buffer);
  }
  g_musa_mr_support.emplace(dev, supported);
  return supported;
}

mcclResult_t pci_path(const char* ib_name, char** path) {
  if (ib_name == nullptr || path == nullptr) return mcclInvalidArgument;
  char device_path[PATH_MAX];
  snprintf(device_path, sizeof(device_path), "/sys/class/infiniband/%s/device",
           ib_name);
  *path = realpath(device_path, nullptr);
  return *path == nullptr ? mcclSystemError : mcclSuccess;
}

mcclResult_t register_mr(McclBaseComm* base, void* data, size_t size, int type,
                         void** mhandle) {
  if (base == nullptr || mhandle == nullptr || data == nullptr || size == 0)
    return mcclInvalidArgument;
  if (type != MCCL_PTR_HOST && type != MCCL_PTR_MUSA) return mcclInvalidArgument;
  if (type == MCCL_PTR_MUSA &&
      !can_register_musa(base->context->endpoint.get(), base->dev))
    return mcclUnhandledMusaError;

  Mhandle* registered = nullptr;
  const int ret = base->context->endpoint->uccl_regmr(
      static_cast<UcclFlow*>(base->conn_id.context), data, size, type,
      &registered);
  if (ret != 0 || registered == nullptr) return mcclSystemError;
  *mhandle = registered;
  return mcclSuccess;
}

mcclResult_t allocate_request(McclBaseComm* base, ucclRequest** request) {
  if (base == nullptr || request == nullptr) return mcclInvalidArgument;
  uint64_t address = 0;
  if (base->request_pool->alloc_buff(&address) != 0) {
    *request = nullptr;
    return mcclSuccess;
  }
  *request = reinterpret_cast<ucclRequest*>(address);
  (*request)->req_pool = base->request_pool.get();
  return mcclSuccess;
}

void release_request(ucclRequest* request) {
  if (request == nullptr || request->req_pool == nullptr) return;
  auto* pool = reinterpret_cast<McclRequestPool*>(request->req_pool);
  pool->free_buff(reinterpret_cast<uint64_t>(request));
}

mcclResult_t close_base(McclBaseComm* base) {
  if (base == nullptr) return mcclInvalidArgument;
  base->request_pool.reset();
  return mcclSuccess;
}

}  // namespace

extern "C" mcclResult_t mcclPluginInit(
    void** ctx, uint64_t comm_id, mcclNetCommConfig_v11_t* /*config*/,
    mcclDebugLogger_t /*log_function*/,
    mcclProfilerCallback_t /*prof_function*/) {
  if (ctx == nullptr) return mcclInvalidArgument;
  try {
    {
      std::lock_guard<std::mutex> lock(g_endpoint_mu);
      if (!g_endpoint)
        g_endpoint = std::make_shared<RDMAEndpoint>(ucclParamNUM_ENGINES());
    }
    *ctx = new McclContext{endpoint(), comm_id};
    return mcclSuccess;
  } catch (...) {
    *ctx = nullptr;
    return mcclSystemError;
  }
}

extern "C" mcclResult_t mcclPluginDevices(int* ndev) {
  if (ndev == nullptr) return mcclInvalidArgument;
  auto ep = endpoint();
  if (!ep) return mcclSystemError;
  *ndev = static_cast<int>(ep->get_num_devices());
  return *ndev > 0 ? mcclSuccess : mcclSystemError;
}

extern "C" mcclResult_t mcclPluginGetProperties(
    int dev, mcclNetProperties_v11_t* props) {
  if (props == nullptr) return mcclInvalidArgument;
  auto ep = endpoint();
  if (!ep || dev < 0 || dev >= static_cast<int>(ep->get_num_devices()))
    return mcclInvalidArgument;

  auto* factory_dev = RDMAFactory::get_factory_dev(dev);
  memset(props, 0, sizeof(*props));
  props->name = const_cast<char*>(kPluginName);
  props->guid = factory_dev->dev_attr.sys_image_guid;
  props->speed = static_cast<int>(factory_dev->link_bw * 8 / 1e6);
  props->port = factory_dev->ib_port_num;
  props->latency = 0;
  props->maxComms = 1024 * 1024;
  props->maxRecvs = 1;
  props->netDeviceType = MCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = MCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = MCCL_MAX_NET_SIZE_BYTES;
  props->maxMultiRequestSize = 1;
  props->ptrSupport = MCCL_PTR_HOST;
  props->regIsGlobal = 0;

  if (pci_path(factory_dev->ib_name, &props->pciPath) != mcclSuccess)
    props->pciPath = nullptr;
  if (can_register_musa(ep.get(), dev)) {
    props->ptrSupport |= MCCL_PTR_MUSA;
    props->forceFlush = 1;
  }
  if (factory_dev->dma_buf_support) props->ptrSupport |= MCCL_PTR_DMABUF;
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginListen(void* ctx, int dev, void* opaque_handle,
                                           void** listen_comm) {
  if (ctx == nullptr || opaque_handle == nullptr || listen_comm == nullptr)
    return mcclInvalidArgument;
  auto* context = static_cast<McclContext*>(ctx);
  auto* handle = static_cast<McclHandle*>(opaque_handle);
  memset(handle, 0, MCCL_NET_HANDLE_MAXSIZE);
  if (initialize_device(context, &dev) != mcclSuccess) return mcclSystemError;

  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return mcclSystemError;
  int reuse = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
    close(fd);
    return mcclSystemError;
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
      listen(fd, 1) != 0) {
    close(fd);
    return mcclSystemError;
  }
  socklen_t address_len = sizeof(address);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&address), &address_len) !=
      0) {
    close(fd);
    return mcclSystemError;
  }

  int gpu = 0;
  if (gpuGetDevice(&gpu) != gpuSuccess) {
    close(fd);
    return mcclUnhandledMusaError;
  }
  auto* factory_dev = RDMAFactory::get_factory_dev(dev);
  handle->ip_addr_u32 = str_to_ip(factory_dev->local_ip_str);
  handle->listen_port = ntohs(address.sin_port);
  handle->remote_dev = dev;
  handle->remote_gpuidx = gpu;

  auto* comm = new McclListenComm{};
  comm->context = context;
  comm->dev = dev;
  comm->listen_fd = fd;
  comm->gpuidx = gpu;
  *listen_comm = comm;
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginConnect(
    void* ctx, int dev, void* opaque_handle, void** send_comm,
    mcclNetDeviceHandle_v11_t** send_dev_comm) {
  if (ctx == nullptr || opaque_handle == nullptr || send_comm == nullptr)
    return mcclInvalidArgument;
  if (send_dev_comm != nullptr) *send_dev_comm = nullptr;
  auto* context = static_cast<McclContext*>(ctx);
  auto* handle = static_cast<McclHandle*>(opaque_handle);
  if (initialize_device(context, &dev) != mcclSuccess) return mcclSystemError;

  int gpu = 0;
  if (gpuGetDevice(&gpu) != gpuSuccess) return mcclUnhandledMusaError;
  if (handle->state == ConnectionState::kInit) {
    handle->state = ConnectionState::kConnecting;
    std::thread([context, dev, gpu, handle] {
      auto* state = new McclBaseComm{};
      state->context = context;
      state->dev = dev;
      state->conn_id = context->endpoint->uccl_connect(
          dev, gpu, handle->remote_dev, handle->remote_gpuidx,
          ip_to_str(handle->ip_addr_u32), handle->listen_port);
      state->request_pool = std::make_shared<McclRequestPool>();
      handle->state = ConnectionState::kConnected;
      handle->remote_dev = state->dev;
      handle->pending_state = state;
    }).detach();
    *send_comm = nullptr;
    return mcclSuccess;
  }
  if (handle->state == ConnectionState::kConnecting) {
    *send_comm = nullptr;
    return mcclSuccess;
  }

  McclBaseComm* state = handle->pending_state;
  handle->pending_state = nullptr;
  if (state == nullptr) return mcclSystemError;
  auto* comm = new McclSendComm{};
  comm->base = *state;
  delete state;
  *send_comm = comm;
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginAccept(
    void* opaque_listen_comm, void** recv_comm,
    mcclNetDeviceHandle_v11_t** recv_dev_comm) {
  if (opaque_listen_comm == nullptr || recv_comm == nullptr)
    return mcclInvalidArgument;
  if (recv_dev_comm != nullptr) *recv_dev_comm = nullptr;
  auto* listen_comm = static_cast<McclListenComm*>(opaque_listen_comm);
  if (listen_comm->state == ConnectionState::kInit) {
    listen_comm->state = ConnectionState::kConnecting;
    std::thread([listen_comm] {
      auto* state = &listen_comm->accept_base;
      int remote_gpu = -1;
      state->context = listen_comm->context;
      state->dev = listen_comm->dev;
      state->conn_id = listen_comm->context->endpoint->uccl_accept(
          listen_comm->dev, listen_comm->listen_fd, listen_comm->gpuidx,
          listen_comm->remote_ip, &listen_comm->remote_dev, &remote_gpu);
      state->request_pool = std::make_shared<McclRequestPool>();
      listen_comm->state = ConnectionState::kConnected;
    }).detach();
    *recv_comm = nullptr;
    return mcclSuccess;
  }
  if (listen_comm->state == ConnectionState::kConnecting) {
    *recv_comm = nullptr;
    return mcclSuccess;
  }

  auto* comm = new McclRecvComm{};
  comm->base = listen_comm->accept_base;
  comm->remote_ip = listen_comm->remote_ip;
  comm->remote_dev = listen_comm->remote_dev;
  *recv_comm = comm;
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginRegMr(void* comm, void* data, size_t size,
                                          int type, void** mhandle) {
  return register_mr(static_cast<McclBaseComm*>(comm), data, size, type,
                     mhandle);
}

extern "C" mcclResult_t mcclPluginRegMrDmaBuf(
    void* comm, void* data, size_t size, int type, uint64_t offset, int fd,
    void** mhandle) {
  auto* base = static_cast<McclBaseComm*>(comm);
  if (base == nullptr || mhandle == nullptr || offset > INT_MAX || fd < 0)
    return mcclInvalidArgument;
  if (type != MCCL_PTR_HOST && type != MCCL_PTR_MUSA) return mcclInvalidArgument;
  if (type == MCCL_PTR_MUSA &&
      !can_register_musa(base->context->endpoint.get(), base->dev))
    return mcclUnhandledMusaError;
  Mhandle* registered = nullptr;
  const int ret = base->context->endpoint->uccl_regmr_dmabuf(
      static_cast<UcclFlow*>(base->conn_id.context), data, size, type,
      static_cast<int>(offset), fd, &registered);
  if (ret != 0 || registered == nullptr || registered->mr == nullptr)
    return mcclSystemError;
  *mhandle = registered;
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginDeregMr(void* comm, void* mhandle) {
  auto* base = static_cast<McclBaseComm*>(comm);
  if (base == nullptr || mhandle == nullptr) return mcclInvalidArgument;
  base->context->endpoint->uccl_deregmr(static_cast<Mhandle*>(mhandle));
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginIsend(
    void* send_comm, void* data, size_t size, int /*tag*/, void* mhandle,
    void* /*p_handle*/, void** request) {
  auto* comm = static_cast<McclSendComm*>(send_comm);
  if (comm == nullptr || request == nullptr || size > INT_MAX)
    return mcclInvalidArgument;
  ucclRequest* req = nullptr;
  allocate_request(&comm->base, &req);
  if (req == nullptr) return mcclSuccess;
  if (comm->base.context->endpoint->uccl_send_async(
          static_cast<UcclFlow*>(comm->base.conn_id.context),
          static_cast<Mhandle*>(mhandle), data, size, req) != 0) {
    release_request(req);
    *request = nullptr;
    return mcclSuccess;
  }
  *request = req;
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginIrecv(
    void* recv_comm, int n, void** data, size_t* sizes, int* /*tags*/,
    void** mhandles, void** /*p_handles*/, void** request) {
  auto* comm = static_cast<McclRecvComm*>(recv_comm);
  if (comm == nullptr || request == nullptr || n <= 0 || data == nullptr ||
      sizes == nullptr || mhandles == nullptr)
    return mcclInvalidArgument;
  std::vector<int> recv_sizes(n);
  for (int i = 0; i < n; ++i) {
    if (sizes[i] > INT_MAX) return mcclInvalidArgument;
    recv_sizes[i] = static_cast<int>(sizes[i]);
  }
  ucclRequest* req = nullptr;
  allocate_request(&comm->base, &req);
  if (req == nullptr) return mcclSuccess;
  if (comm->base.context->endpoint->uccl_recv_async(
          static_cast<UcclFlow*>(comm->base.conn_id.context),
          reinterpret_cast<Mhandle**>(mhandles), data, recv_sizes.data(), n,
          req) != 0) {
    release_request(req);
    *request = nullptr;
    return mcclSuccess;
  }
  *request = req;
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginIflush(
    void* recv_comm, int n, void** data, int* sizes, void** mhandles,
    void** request) {
  auto* comm = static_cast<McclRecvComm*>(recv_comm);
  if (comm == nullptr || request == nullptr || n <= 0 || data == nullptr ||
      sizes == nullptr || mhandles == nullptr)
    return mcclInvalidArgument;
  ucclRequest* req = nullptr;
  allocate_request(&comm->base, &req);
  if (req == nullptr) return mcclSuccess;
  if (comm->base.context->endpoint->uccl_flush(
          static_cast<UcclFlow*>(comm->base.conn_id.context),
          reinterpret_cast<Mhandle**>(mhandles), data, sizes, n, req) != 0) {
    release_request(req);
    *request = nullptr;
    return mcclSuccess;
  }
  *request = req;
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginTest(void* opaque_request, int* done,
                                         int* sizes) {
  if (opaque_request == nullptr || done == nullptr) return mcclInvalidArgument;
  auto* request = static_cast<ucclRequest*>(opaque_request);
  auto ep = endpoint();
  if (!ep || request->context == nullptr) return mcclInternalError;
  if (!ep->uccl_poll_ureq_once(request)) {
    *done = 0;
    return mcclSuccess;
  }
  *done = 1;
  if (sizes != nullptr) {
    if (request->type == ReqTx || request->type == ReqTxRC) {
      sizes[0] = request->send.data_len;
    } else if (request->type == ReqRx || request->type == ReqRxRC) {
      for (int i = 0; i < request->n; ++i)
        sizes[i] = request->recv.data_len[i];
    }
  }
  release_request(request);
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginCloseSend(void* comm) {
  if (comm == nullptr) return mcclInvalidArgument;
  auto* send = static_cast<McclSendComm*>(comm);
  close_base(&send->base);
  delete send;
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginCloseRecv(void* comm) {
  if (comm == nullptr) return mcclInvalidArgument;
  auto* recv = static_cast<McclRecvComm*>(comm);
  close_base(&recv->base);
  delete recv;
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginCloseListen(void* comm) {
  if (comm == nullptr) return mcclInvalidArgument;
  auto* listen = static_cast<McclListenComm*>(comm);
  close(listen->listen_fd);
  delete listen;
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginFinalize(void* opaque_context) {
  delete static_cast<McclContext*>(opaque_context);
  return mcclSuccess;
}

extern "C" mcclResult_t mcclPluginGetDeviceMr(void*, void*, void**) {
  return mcclInternalError;
}

extern "C" mcclResult_t mcclPluginIrecvConsumed(void*, int, void*) {
  return mcclInternalError;
}

extern "C" mcclResult_t mcclPluginMakeVDevice(int*,
                                                mcclNetVDeviceProps_v11_t*) {
  return mcclInternalError;
}

extern "C" mcclResult_t mcclPluginSetNetAttr(void*, mcclNetAttr_v11_t*) {
  return mcclSuccess;
}

extern "C" const mcclNet_v11_t mcclNetPlugin_v11 = {
    kPluginName,
    mcclPluginInit,
    mcclPluginDevices,
    mcclPluginGetProperties,
    mcclPluginListen,
    mcclPluginConnect,
    mcclPluginAccept,
    mcclPluginRegMr,
    mcclPluginRegMrDmaBuf,
    mcclPluginDeregMr,
    mcclPluginIsend,
    mcclPluginIrecv,
    mcclPluginIflush,
    mcclPluginTest,
    mcclPluginCloseSend,
    mcclPluginCloseRecv,
    mcclPluginCloseListen,
    mcclPluginGetDeviceMr,
    mcclPluginIrecvConsumed,
    mcclPluginMakeVDevice,
    mcclPluginFinalize,
    mcclPluginSetNetAttr,
};
