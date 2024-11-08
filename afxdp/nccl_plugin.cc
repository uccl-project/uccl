
#include <glog/logging.h>
#include <signal.h>

#include <atomic>
#include <thread>

#include "nccl_net.h"
#include "transport.h"
#include "transport_config.h"

using namespace uccl;

const char* PLUGIN_NAME = "AFXDP_Plugin";
const size_t NUM_FRAMES = 4096 * 64;  // 1GB frame pool
const size_t QUEUE_ID = 0;

volatile bool quit = false;

void interrupt_handler(int signal) {
    (void)signal;
    quit = true;
    AFXDPFactory::shutdown();
}

Channel* channel;
UcclEngine* engine;
Endpoint* ep;
std::thread* engine_th;

ncclResult_t pluginInit(ncclDebugLogger_t logFunction) {
    google::InitGoogleLogging("nccl_plugin");
    google::InstallFailureSignalHandler();

    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, interrupt_handler);
    signal(SIGHUP, interrupt_handler);

    channel = new Channel();

    AFXDPFactory::init(interface_name,
                       "/home/ubuntu/uccl/afxdp/ebpf_transport.o",
                       "ebpf_transport");

    std::string local_ip_str = get_dev_ip(interface_name);
    DCHECK(local_ip_str != "");
    std::string local_mac_str = get_dev_mac(interface_name);
    DCHECK(local_mac_str != "");

    bool is_this_client =
        (local_ip_str == client_ip_str && local_mac_str == client_mac_str);
    bool is_this_server =
        (local_ip_str == server_ip_str && local_mac_str == server_mac_str);

    if (is_this_client) {
        LOG(INFO) << "pluginListen: This is the client machine";
        engine = new UcclEngine(QUEUE_ID, NUM_FRAMES, channel, client_ip_str,
                                client_port, server_ip_str, server_port,
                                client_mac_str, server_mac_str);
    } else if (is_this_server) {
        LOG(INFO) << "pluginListen: This is the server machine";
        engine = new UcclEngine(QUEUE_ID, NUM_FRAMES, channel, server_ip_str,
                                server_port, client_ip_str, client_port,
                                server_mac_str, client_mac_str);
    } else {
        DCHECK(false) << "This machine is neither client nor server";
    }

    engine_th = new std::thread([]() {
        pin_thread_to_cpu(2);
        engine->run();
    });

    // pin_thread_to_cpu(3);

    ep = new Endpoint(channel);
    if (is_this_client) {
        ep->uccl_connect(server_ip_str);
        LOG(INFO) << "Connected to server " << server_ip_str;
    } else if (is_this_server) {
        auto [conn_id, client_ip_str] = ep->uccl_accept();
        LOG(INFO) << "Accepted connection from " << client_ip_str;
    } else {
        DCHECK(false) << "This machine is neither client nor server";
    }

    LOG(INFO) << "NCCL Plugin initialized";
    return ncclSuccess;
}
ncclResult_t pluginDevices(int* ndev) {
    LOG(INFO) << "pluginDevices";
    *ndev = 1;
    return ncclSuccess;
}

ncclResult_t pluginPciPath(int dev, char** path) { return ncclSuccess; }
ncclResult_t pluginPtrSupport(int dev, int* supportedTypes) {
    return ncclSuccess;
}
ncclResult_t pluginGetProperties(int dev, ncclNetProperties_v8_t* props) {
    // Below are default values, if unsure don't change.

    props->name = (char*)"Example";
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

struct afxdp_context {
    ConnectionID conn_id;
};

ncclResult_t pluginListen(int dev, void* handle, void** listenComm) {
    struct afxdp_context* ctx = static_cast<struct afxdp_context*>(handle);
    static_assert(sizeof(struct afxdp_context) < NCCL_NET_HANDLE_MAXSIZE,
                  "ncclSocketHandle size too large");

    *listenComm = ctx;
    return ncclSuccess;
}
ncclResult_t pluginConnect(int dev, void* handle, void** sendComm,
                           ncclNetDeviceHandle_v8_t** sendDevComm) {
    // This handle data is transferred from remote via MPI
    struct afxdp_context* ctx = static_cast<struct afxdp_context*>(handle);

    *sendComm = ctx;
    return ncclSuccess;
}
ncclResult_t pluginAccept(void* listenComm, void** recvComm,
                          ncclNetDeviceHandle_v8_t** recvDevComm) {
    struct afxdp_context* ctx = static_cast<struct afxdp_context*>(listenComm);

    *recvComm = ctx;
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

#define MAX_RECV_CHUNKS 32
struct afxdp_request {
    struct afxdp_context* ctx;
    bool send;
    size_t data_len = 0;
};

static std::atomic<size_t> inflight_send = 0;
static std::atomic<size_t> inflight_recv = 0;

ncclResult_t pluginIsend(void* sendComm, void* data, int size, int tag,
                         void* mhandle, void** request) {
    inflight_send += size;
    // LOG(INFO) << "pluginIsend " << size << " " << inflight_send;
    struct afxdp_context* ctx = static_cast<struct afxdp_context*>(sendComm);
    auto req = new afxdp_request();
    req->ctx = ctx;
    req->send = true;
    req->data_len = size;
    DCHECK(ep->uccl_send_async(ctx->conn_id, data, req->data_len));

    *request = req;
    return ncclSuccess;
}
ncclResult_t pluginIrecv(void* recvComm, int n, void** data, int* sizes,
                         int* tags, void** mhandles, void** request) {
    if (n != 1) return ncclInternalError;
    inflight_recv += sizes[0];
    // LOG(INFO) << "pluginIrecv " << sizes[0] << " " << inflight_recv;
    struct afxdp_context* ctx = static_cast<struct afxdp_context*>(recvComm);
    auto req = new afxdp_request();
    req->ctx = ctx;
    req->send = false;
    req->data_len = sizes[0];
    DCHECK(ep->uccl_recv_async(ctx->conn_id, data[0], &req->data_len));

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
    struct afxdp_request* req = static_cast<struct afxdp_request*>(request);
    bool ret = false;
    if (req->send) {
        ret = ep->uccl_send_poll_once();
        if (ret) {
            inflight_send -= req->data_len;
            // LOG(INFO) << "pluginTest send " << req->data_len << " " <<
            // inflight_send;
        }
    } else {
        ret = ep->uccl_recv_poll_once();
        if (ret) {
            inflight_recv -= req->data_len;
            // LOG(INFO) << "pluginTest recv " << req->data_len << " " <<
            // inflight_recv;
        }
    }
    if (ret) {
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
