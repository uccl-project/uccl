
#include <glog/logging.h>
#include <signal.h>

#include <atomic>
#include <thread>

#include "nccl_net.h"
#include "transport.h"

using namespace uccl;

const char* PLUGIN_NAME = "AFXDP_Plugin";

volatile bool quit = false;

void interrupt_handler(int signal) {
    (void)signal;
    quit = true;
    AFXDPFactory::shutdown();
}

Endpoint* ep;

struct ConnectAcceptHandler {
    uint32_t ip_addr_u32;
    uint32_t conn_idx;

    bool operator==(const ConnectAcceptHandler& other) const {
        return ip_addr_u32 == other.ip_addr_u32 && conn_idx == other.conn_idx;
    }
};

namespace std {
template <>
struct hash<ConnectAcceptHandler> {
    std::size_t operator()(const ConnectAcceptHandler& key) const {
        return std::hash<uint64_t>{}(*(uint64_t*)&key);
    }
};
}  // namespace std

struct CommHandler {
    ConnID conn_id;
};

struct AsyncConnectState {
    CommHandler comm_handler;
    std::atomic<bool> done = false;
    std::thread connect_th;
};
// Mapping from remote_ip to AsyncConnectState
std::mutex connect_tracker_mu;
std::unordered_map<ConnectAcceptHandler, AsyncConnectState*> connect_tracker;

struct UcclRequest {
    bool send = false;
    size_t data_len = 0;
    size_t recv_len = 0;
    PollCtx* poll_ctx = nullptr;
    bool done = false;

    void clear() { memset(this, 0, sizeof(UcclRequest)); }
};

static constexpr size_t kMaxInflightMsg = 1024 * 256;
static char* uccl_req_pool_buf = nullptr;
SharedPool<UcclRequest*, true> uccl_req_pool(kMaxInflightMsg);

static std::atomic<size_t> inflight_send = 0;
static std::atomic<size_t> inflight_recv = 0;

ncclResult_t pluginInit(ncclDebugLogger_t logFunction) {
    google::InitGoogleLogging("nccl_plugin");
    google::InstallFailureSignalHandler();

    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, interrupt_handler);
    signal(SIGHUP, interrupt_handler);

    ep = new Endpoint(DEV_DEFAULT, NUM_QUEUES, NUM_FRAMES, ENGINE_CPU_START);
    // pin_thread_to_cpu(ENGINE_CPU_START + 1);

    uccl_req_pool_buf = (char*)malloc(sizeof(UcclRequest) * kMaxInflightMsg);
    for (int i = 0; i < kMaxInflightMsg; i++) {
        auto* req = (UcclRequest*)uccl_req_pool_buf + i;
        req->clear();
        uccl_req_pool.push(req);
    }

    LOG(INFO) << "NCCL Plugin initialized";
    return ncclSuccess;
}
ncclResult_t pluginDevices(int* ndev) {
    *ndev = 1;
    LOG(INFO) << "pluginDevices 1";
    return ncclSuccess;
}

ncclResult_t pluginPciPath(int dev, char** path) { return ncclSuccess; }
ncclResult_t pluginPtrSupport(int dev, int* supportedTypes) {
    return ncclSuccess;
}
ncclResult_t pluginGetProperties(int dev, ncclNetProperties_v8_t* props) {
    // Below are default values, if unsure don't change.

    props->name = (char*)DEV_DEFAULT;
    // Fill for proper topology detection, e.g.
    // /sys/devices/pci0000:00/0000:00:10.0/0000:0b:00.0
    props->pciPath = NULL;
    // Only used to detect NICs with multiple PCI attachments.
    props->guid = 0;
    // Add NCCL_PTR_CUDA if GPU Direct RDMA is supported and regMr can take CUDA
    // pointers.
    props->ptrSupport = NCCL_PTR_HOST;
    // If you regMr has a fast registration cache, set to 1. If set to 0, user
    // buffer registration may be disabled.
    props->regIsGlobal = 0;
    // Speed in *Mbps*. 100000 means 100G
    props->speed = 100000;
    // Port number, used in conjunction with guid
    props->port = 0;
    // Custom latency (used to help tuning if latency is high. If set to 0, use
    // default NCCL values.
    props->latency = 0;
    // Maximum number of comm objects we can create.
    props->maxComms = 1024 * 1024;
    // Maximum number of receive operations taken by irecv().
    props->maxRecvs = 1;
    // Coupling with NCCL network device-side code.
    props->netDeviceType = (ncclNetDeviceType)0;
    props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
    return ncclSuccess;
}

// To create a connection, NCCL will start by calling listen on the receiver
// side. This function takes a device number as input argument, and should
// return a local listenComm object, and a handle to pass to the other side, so
// that the sender side can connect to the receiver. The handle is a buffer of
// size NCCL_NET_HANDLE_MAXSIZE and is provided by NCCL. This call should never
// block, but contrary to connect and accept, listenComm should never be NULL if
// the call succeeds.
ncclResult_t pluginListen(int dev, void* handle, void** listenComm) {
    DCHECK(dev == 0);

    struct ConnectAcceptHandler* ctx =
        static_cast<struct ConnectAcceptHandler*>(handle);
    static_assert(sizeof(struct ConnectAcceptHandler) < NCCL_NET_HANDLE_MAXSIZE,
                  "ncclSocketHandle size too large");
    static uint32_t global_conn_idx = 0;

    std::string local_ip_str = get_dev_ip(DEV_DEFAULT);
    ctx->ip_addr_u32 = str_to_ip(local_ip_str);
    ctx->conn_idx = global_conn_idx++;
    LOG(INFO) << "pluginListen: " << local_ip_str;

    // Listen is alreday done by Endpoint init.
    *listenComm = ctx;
    return ncclSuccess;
}

// NCCL will use its bootstrap infrastructure to provide the handle to the
// sender side, then call connect on the sender side on a given device index
// dev, providing the handle. connect should not block either, and instead set
// sendComm to NULL and return ncclSuccess. In that case, NCCL will call accept
// again until it succeeds.
ncclResult_t pluginConnect(int dev, void* handle, void** sendComm,
                           ncclNetDeviceHandle_v8_t** sendDevComm) {
    DCHECK(dev == 0);

    // This handle data from pluginListen is transferred from remote via MPI
    struct ConnectAcceptHandler* ctrl_ctx =
        static_cast<struct ConnectAcceptHandler*>(handle);
    auto remote_ip = ctrl_ctx->ip_addr_u32;

    std::lock_guard<std::mutex> lock(connect_tracker_mu);
    auto it = connect_tracker.find(*ctrl_ctx);
    if (it == connect_tracker.end()) {
        auto* connect_state = new AsyncConnectState();
        connect_state->connect_th = std::thread([connect_state, remote_ip] {
            std::string remote_ip_str = ip_to_str(remote_ip);
            connect_state->comm_handler.conn_id =
                ep->uccl_connect(remote_ip_str);
            LOG(INFO) << "pluginConnect: connected to " << remote_ip_str;
            std::atomic_thread_fence(std::memory_order_release);
            std::atomic_store_explicit(&connect_state->done, true,
                                       std::memory_order_relaxed);
        });
        connect_tracker[*ctrl_ctx] = connect_state;
        *sendComm = nullptr;
    } else {
        auto* connect_state = it->second;
        auto done = std::atomic_load_explicit(&connect_state->done,
                                              std::memory_order_relaxed);
        if (done) {
            std::atomic_thread_fence(std::memory_order_acquire);
            auto* comm_handler = new CommHandler();
            *comm_handler = connect_state->comm_handler;
            connect_state->connect_th.join();
            delete connect_state;
            *sendComm = comm_handler;
        } else {
            *sendComm = nullptr;
        }
    }
    return ncclSuccess;
}

// To finalize the connection, the receiver side will call accept on the
// listenComm returned by the listen call previously. If the sender did not
// connect yet, accept should not block. It should return ncclSuccess, setting
// recvComm to NULL. NCCL will call accept again until it succeeds.
ncclResult_t pluginAccept(void* listenComm, void** recvComm,
                          ncclNetDeviceHandle_v8_t** recvDevComm) {
    struct ConnectAcceptHandler* ctrl_ctx =
        static_cast<struct ConnectAcceptHandler*>(listenComm);
    auto* comm_handler = new CommHandler{};

    std::string remote_ip_str;
    comm_handler->conn_id = ep->uccl_accept(remote_ip_str);
    LOG(INFO) << "pluginAccept: accepted connection from " << remote_ip_str;

    *recvComm = comm_handler;
    return ncclSuccess;
}

ncclResult_t pluginRegMr(void* collComm, void* data, size_t size, int type,
                         void** mhandle) {
    return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}

ncclResult_t pluginRegMrDmaBuf(void* collComm, void* data, size_t size,
                               int type, uint64_t offset, int fd,
                               void** mhandle) {
    return ncclInternalError;
}

ncclResult_t pluginDeregMr(void* collComm, void* mhandle) {
    return ncclSuccess;
}

ncclResult_t pluginIsend(void* sendComm, void* data, int size, int tag,
                         void* mhandle, void** request) {
    inflight_send += size;
    struct CommHandler* comm_handler =
        static_cast<struct CommHandler*>(sendComm);
    auto conn_id = comm_handler->conn_id;

    auto* req = uccl_req_pool.pop();
    req->send = true;
    req->data_len = size;
    req->poll_ctx = ep->uccl_send_async(conn_id, data, size);

    // LOG(INFO) << "pluginIsend " << size << " " << inflight_send;

    *request = req;
    return ncclSuccess;
}

ncclResult_t pluginIrecv(void* recvComm, int n, void** data, int* sizes,
                         int* tags, void** mhandles, void** request) {
    if (n != 1) return ncclInternalError;
    inflight_recv += sizes[0];
    struct CommHandler* comm_handler =
        static_cast<struct CommHandler*>(recvComm);
    auto conn_id = comm_handler->conn_id;

    auto* req = uccl_req_pool.pop();
    req->send = false;
    req->data_len = sizes[0];
    req->poll_ctx = ep->uccl_recv_async(conn_id, data[0], &req->recv_len);

    // LOG(INFO) << "pluginIrecv " << sizes[0];

    *request = req;
    return ncclSuccess;
}

ncclResult_t pluginIflush(void* recvComm, int n, void** data, int* sizes,
                          void** mhandles, void** request) {
    // We don't support CUDA pointers, so we don't need a flush operation
    return ncclInternalError;
}

ncclResult_t pluginTest(void* request, int* done, int* size) {
    *done = 0;
    struct UcclRequest* req = static_cast<struct UcclRequest*>(request);

    if (ep->uccl_poll_once(req->poll_ctx)) {
        if (req->send) {
            inflight_send -= req->data_len;
            VLOG(4) << "pluginTest send " << req->data_len << " "
                    << inflight_send;
        } else {
            inflight_recv -= req->data_len;
            VLOG(4) << "pluginTest recv " << req->data_len << " "
                    << inflight_recv;
        }
        *done = 1;
        *size = req->data_len;
        req->clear();
        uccl_req_pool.push(req);
    }
    return ncclSuccess;
}

ncclResult_t pluginCloseSend(void* sendComm) { return ncclSuccess; }
ncclResult_t pluginCloseRecv(void* recvComm) { return ncclSuccess; }
ncclResult_t pluginCloseListen(void* listenComm) { return ncclSuccess; }
ncclResult_t pluginIrecvConsumed(void* recvComm, int n, void* request) {
    return ncclSuccess;
}
ncclResult_t pluginGetDeviceMr(void* comm, void* mhandle, void** dptr_mhandle) {
    return ncclSuccess;
}

volatile ncclNet_v8_t ncclNetPlugin_v8 = {
    .name = PLUGIN_NAME,
    .init = pluginInit,
    .devices = pluginDevices,
    .getProperties = pluginGetProperties,
    .listen = pluginListen,
    .connect = pluginConnect,
    .accept = pluginAccept,
    .regMr = pluginRegMr,
    .regMrDmaBuf = pluginRegMrDmaBuf,
    .deregMr = pluginDeregMr,
    .isend = pluginIsend,
    .irecv = pluginIrecv,
    .iflush = pluginIflush,
    .test = pluginTest,
    .closeSend = pluginCloseSend,
    .closeRecv = pluginCloseRecv,
    .closeListen = pluginCloseListen,
    .getDeviceMr = pluginGetDeviceMr,
    .irecvConsumed = pluginIrecvConsumed,
};
