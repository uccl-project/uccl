
#include <glog/logging.h>
#include <signal.h>

#include <atomic>
#include <thread>

#include "nccl_net.h"
#include "transport.h"
#include "transport_config.h"

using namespace uccl;

const char* PLUGIN_NAME = "AFXDP_Plugin";

volatile bool quit = false;

void interrupt_handler(int signal) {
    (void)signal;
    quit = true;
    AFXDPFactory::shutdown();
}

Endpoint* ep;

ncclResult_t pluginInit(ncclDebugLogger_t logFunction) {
    google::InitGoogleLogging("nccl_plugin");
    google::InstallFailureSignalHandler();

    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, interrupt_handler);
    signal(SIGHUP, interrupt_handler);

    ep = new Endpoint(DEV_DEFAULT, NUM_QUEUES, NUM_FRAMES, ENGINE_CPU_START);
    // pin_thread_to_cpu(ENGINE_CPU_START + 1);

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

struct SharedCtlCtx {
    uint32_t ip_addr_u32;
};

// To create a connection, NCCL will start by calling listen on the receiver
// side. This function takes a device number as input argument, and should
// return a local listenComm object, and a handle to pass to the other side, so
// that the sender side can connect to the receiver. The handle is a buffer of
// size NCCL_NET_HANDLE_MAXSIZE and is provided by NCCL. This call should never
// block, but contrary to connect and accept, listenComm should never be NULL if
// the call succeeds.
ncclResult_t pluginListen(int dev, void* handle, void** listenComm) {
    DCHECK(dev == 0);

    struct SharedCtlCtx* ctx = static_cast<struct SharedCtlCtx*>(handle);
    static_assert(sizeof(struct SharedCtlCtx) < NCCL_NET_HANDLE_MAXSIZE,
                  "ncclSocketHandle size too large");

    std::string local_ip_str = get_dev_ip(DEV_DEFAULT);
    ctx->ip_addr_u32 = str_to_ip(local_ip_str);
    LOG(INFO) << "pluginListen: " << local_ip_str;

    // Listen is alreday done by Endpoint init.
    *listenComm = ctx;
    return ncclSuccess;
}

struct SharedDataCtx {
    ConnID conn_id[NUM_QUEUES];
};

struct AsyncConnectCtx {
    SharedDataCtx data_ctx;
    std::atomic<bool> done = false;
    std::thread connect_th;
};
// Mapping from remote_ip to AsyncConnectCtx
std::mutex async_connect_ctx_map_mu;
std::unordered_map<uint32_t, AsyncConnectCtx*> async_connect_ctx_map;

// NCCL will use its bootstrap infrastructure to provide the handle to the
// sender side, then call connect on the sender side on a given device index
// dev, providing the handle. connect should not block either, and instead set
// sendComm to NULL and return ncclSuccess. In that case, NCCL will call accept
// again until it succeeds.
ncclResult_t pluginConnect(int dev, void* handle, void** sendComm,
                           ncclNetDeviceHandle_v8_t** sendDevComm) {
    DCHECK(dev == 0);

    // This handle data from pluginListen is transferred from remote via MPI
    struct SharedCtlCtx* ctrl_ctx = static_cast<struct SharedCtlCtx*>(handle);
    auto remote_ip = ctrl_ctx->ip_addr_u32;

    std::lock_guard<std::mutex> lock(async_connect_ctx_map_mu);
    auto it = async_connect_ctx_map.find(remote_ip);
    if (it == async_connect_ctx_map.end()) {
        auto* async_ctx = new AsyncConnectCtx();
        async_ctx->connect_th = std::thread([async_ctx, remote_ip] {
            std::string remote_ip_str = ip_to_str(remote_ip);
            for (int i = 0; i < NUM_QUEUES; i++) {
                async_ctx->data_ctx.conn_id[i] =
                    ep->uccl_connect(remote_ip_str);
            }
            LOG(INFO) << "pluginConnect: connected to " << remote_ip_str;
            std::atomic_thread_fence(std::memory_order_release);
            std::atomic_store_explicit(&async_ctx->done, true,
                                       std::memory_order_relaxed);
        });
        async_connect_ctx_map[remote_ip] = async_ctx;
        *sendComm = nullptr;
    } else {
        auto* async_ctx = it->second;
        auto done = std::atomic_load_explicit(&async_ctx->done,
                                              std::memory_order_relaxed);
        if (done) {
            std::atomic_thread_fence(std::memory_order_acquire);
            auto* data_ctx = new SharedDataCtx();
            *data_ctx = async_ctx->data_ctx;
            async_ctx->connect_th.join();
            delete async_ctx;
            *sendComm = data_ctx;
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
    struct SharedCtlCtx* ctrl_ctx =
        static_cast<struct SharedCtlCtx*>(listenComm);
    auto* data_ctx = new SharedDataCtx{};

    std::string remote_ip_str;
    for (int i = 0; i < NUM_QUEUES; i++) {
        std::tie(data_ctx->conn_id[i], remote_ip_str) = ep->uccl_accept();
    }
    LOG(INFO) << "pluginAccept: accepted connection from " << remote_ip_str;

    *recvComm = data_ctx;
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

struct UcclRequest {
    bool send;
    size_t data_len = 0;
    size_t recv_len[NUM_QUEUES];
    PollCtx* poll_ctxs[NUM_QUEUES];
    bool done[NUM_QUEUES] = {false};
};

static std::atomic<size_t> inflight_send = 0;
static std::atomic<size_t> inflight_recv = 0;

ncclResult_t pluginIsend(void* sendComm, void* data, int size, int tag,
                         void* mhandle, void** request) {
    inflight_send += size;
    struct SharedDataCtx* data_ctx =
        static_cast<struct SharedDataCtx*>(sendComm);
    auto* conn_id_vec = data_ctx->conn_id;

    auto req = new UcclRequest();
    req->send = true;
    req->data_len = size;

    size_t step_size = size / NUM_QUEUES;
    if (size % NUM_QUEUES != 0) step_size += 1;

    for (int i = 0; i < NUM_QUEUES; i++) {
        auto iter_len = std::min(step_size, size - i * step_size);
        auto* iter_data = data + i * step_size;
        req->poll_ctxs[i] =
            ep->uccl_send_async(conn_id_vec[i], iter_data, iter_len);
    }

    VLOG(4) << "pluginIsend " << size << " " << inflight_send;

    *request = req;
    return ncclSuccess;
}

ncclResult_t pluginIrecv(void* recvComm, int n, void** data, int* sizes,
                         int* tags, void** mhandles, void** request) {
    if (n != 1) return ncclInternalError;
    inflight_recv += sizes[0];
    struct SharedDataCtx* data_ctx =
        static_cast<struct SharedDataCtx*>(recvComm);
    auto* conn_id_vec = data_ctx->conn_id;

    auto req = new UcclRequest();
    req->send = false;
    req->data_len = sizes[0];

    size_t step_size = sizes[0] / NUM_QUEUES;
    if (sizes[0] % NUM_QUEUES != 0) step_size += 1;

    for (int i = 0; i < NUM_QUEUES; i++) {
        auto iter_len = std::min(step_size, sizes[0] - i * step_size);
        auto* iter_data = data[0] + i * step_size;
        req->poll_ctxs[i] =
            ep->uccl_recv_async(conn_id_vec[i], iter_data, &req->recv_len[i]);
    }

    VLOG(4) << "pluginIrecv " << sizes[0] << " " << inflight_recv;

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

    bool all_done = true;
    for (int i = 0; i < NUM_QUEUES; i++) {
        if (!req->done[i]) {
            req->done[i] = ep->uccl_poll_once(req->poll_ctxs[i]);
        }
        all_done = all_done && req->done[i];
    }

    if (all_done) {
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
        delete req;
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
