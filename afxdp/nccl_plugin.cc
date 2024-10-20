
#include <glog/logging.h>
#include <signal.h>

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

ncclResult_t pluginInit(ncclDebugLogger_t logFunction) {
    google::InitGoogleLogging("nccl_plugin");
    google::InstallFailureSignalHandler();

    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, interrupt_handler);
    signal(SIGHUP, interrupt_handler);
    // signal(SIGALRM, interrupt_handler);
    // alarm(10);

    // TODO(yang): separate into separate root process and pass xsk sock back.
    AFXDPFactory::init("ens6", "/home/ubuntu/uccl/afxdp/ebpf_transport.o",
                       "ebpf_transport");
    LOG(INFO) << "NCCL Plugin initialized";
    return ncclSuccess;
}
ncclResult_t pluginDevices(int* ndev) {
    LOG(INFO) << "Plugin Devices";
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
    Channel* channel;
    UcclEngine* engine;
    Endpoint* ep;
    std::thread* engine_th;
    ConnectionID conn_id;
};

ncclResult_t pluginListen(int dev, void* handle, void** listenComm) {
    LOG(INFO) << "Plugin Listen";
    struct afxdp_context* ctx = static_cast<struct afxdp_context*>(handle);
    static_assert(sizeof(struct afxdp_context) < NCCL_NET_HANDLE_MAXSIZE,
                  "ncclSocketHandle size too large");
    ctx->channel = new Channel();

    std::string local_ip_str = get_dev_ip("ens6");
    DCHECK(local_ip_str != "");
    std::string local_mac_str = get_dev_mac("ens6");
    DCHECK(local_mac_str != "");

    std::string client_mac_str = mac_to_str(client_mac_char);
    std::string server_mac_str = mac_to_str(server_mac_char);
    DCHECK(server_mac_str != "" && client_mac_str != "");

    if (local_ip_str == client_ip_str && local_mac_str == client_mac_str) {
        LOG(INFO) << "This is the client machine";
        ctx->engine = new UcclEngine(
            QUEUE_ID, NUM_FRAMES, ctx->channel, client_ip_str, client_port,
            server_ip_str, server_port, client_mac_char, server_mac_char);

    } else if (local_ip_str == server_ip_str &&
               local_mac_str == server_mac_str) {
        LOG(INFO) << "This is the server machine";
        ctx->engine = new UcclEngine(
            QUEUE_ID, NUM_FRAMES, ctx->channel, server_ip_str, server_port,
            client_ip_str, client_port, server_mac_char, client_mac_char);
    } else {
        DCHECK(false) << "This machine is neither client nor server";
    }

    ctx->ep = new Endpoint(ctx->channel);
    ctx->engine_th = new std::thread([&ctx]() {
        pin_thread_to_cpu(2);
        ctx->engine->run();
    });

    pin_thread_to_cpu(3);

    *listenComm = ctx;
    return ncclSuccess;
}
ncclResult_t pluginConnect(int dev, void* handle, void** sendComm,
                           ncclNetDeviceHandle_v8_t** sendDevComm) {
    struct afxdp_context* ctx = static_cast<struct afxdp_context*>(handle);
    ctx->conn_id = ctx->ep->uccl_connect(server_ip_str);

    *sendComm = ctx;
    return ncclSuccess;
}
ncclResult_t pluginAccept(void* listenComm, void** recvComm,
                          ncclNetDeviceHandle_v8_t** recvDevComm) {
    // Nothing to do

    *recvComm = listenComm;
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

struct afxdp_request {
    struct afxdp_context* ctx;
    bool send;
    size_t data_len = 0;
};

ncclResult_t pluginIsend(void* sendComm, void* data, int size, int tag,
                         void* mhandle, void** request) {
    struct afxdp_context* ctx = static_cast<struct afxdp_context*>(sendComm);
    auto req = new afxdp_request();
    req->ctx = ctx;
    req->send = true;
    req->data_len = size;
    // TODO(yang): fixing EndPoint Channel Msg to accept size_t of data_len, not
    // ptr.
    DCHECK(ctx->ep->uccl_send_async(ctx->conn_id, data, req->data_len));

    *request = ctx;
    return ncclSuccess;
}
static size_t recv_data_len = 0;
ncclResult_t pluginIrecv(void* recvComm, int n, void** data, int* sizes,
                         int* tags, void** mhandles, void** request) {
    if (n != 1) return ncclInternalError;
    struct afxdp_context* ctx = static_cast<struct afxdp_context*>(recvComm);
    auto req = new afxdp_request();
    req->ctx = ctx;
    req->send = false;
    req->data_len = sizes[0];
    DCHECK(ctx->ep->uccl_recv_async(ctx->conn_id, data[0], &req->data_len));

    *request = ctx;
    return ncclSuccess;
}
ncclResult_t pluginIflush(void* recvComm, int n, void** data, int* sizes,
                          void** mhandles, void** request) {
    // We don't support CUDA pointers, so we don't need a flush operation
    return ncclInternalError;
}
ncclResult_t pluginTest(void* request, int* done, int* size) {
    struct afxdp_request* req = static_cast<struct afxdp_request*>(request);
    if (req->send) {
        DCHECK(req->ctx->ep->uccl_send_poll());
    } else {
        DCHECK(req->ctx->ep->uccl_recv_poll());
    }
    *done = 1;
    *size = req->data_len;
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
