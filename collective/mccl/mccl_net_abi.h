#ifndef UCCL_MCCL_NET_ABI_H_
#define UCCL_MCCL_NET_ABI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MCCL_NET_HANDLE_MAXSIZE 128
#define MCCL_MAX_NET_SIZE_BYTES (1L * 1024 * 1024 * 1024 * 1024)
#define MCCL_NET_MAX_DEVS_PER_NIC 4

#define MCCL_PTR_HOST 0x1
#define MCCL_PTR_MUSA 0x2
#define MCCL_PTR_DMABUF 0x4

#define MCCL_NET_DEVICE_INVALID_VERSION 0x0

typedef enum {
  mcclSuccess = 0,
  mcclUnhandledMusaError = 1,
  mcclSystemError = 2,
  mcclInternalError = 3,
  mcclInvalidArgument = 4,
  mcclInvalidUsage = 5,
  mcclRemoteError = 6,
} mcclResult_t;

typedef enum {
  MCCL_LOG_NONE = 0,
  MCCL_LOG_VERSION = 1,
  MCCL_LOG_WARN = 2,
  MCCL_LOG_INFO = 3,
  MCCL_LOG_ABORT = 4,
  MCCL_LOG_TRACE = 5,
} mcclDebugLogLevel;

typedef void (*mcclDebugLogger_t)(mcclDebugLogLevel level, unsigned long flags,
                                  char const* file, int line, char const* fmt,
                                  ...);

typedef mcclResult_t (*mcclProfilerCallback_t)(void** eHandle, int type,
                                               void* pHandle, int64_t pluginId,
                                               void* extData);

typedef enum {
  MCCL_NET_DEVICE_HOST = 0,
  MCCL_NET_DEVICE_UNPACK = 1,
  MCCL_NET_DEVICE_GIN_PROXY = 2,
  MCCL_NET_DEVICE_GIN_GDAKI = 3,
} mcclNetDeviceType;

typedef struct {
  mcclNetDeviceType netDeviceType;
  int netDeviceVersion;
  void* handle;
  size_t size;
  int needsProxyProgress;
} mcclNetDeviceHandle_v11_t;

typedef struct {
  int ndevs;
  int devs[MCCL_NET_MAX_DEVS_PER_NIC];
} mcclNetVDeviceProps_v11_t;

typedef struct {
  int trafficClass;
} mcclNetCommConfig_v11_t;

typedef struct {
  char* name;
  char* pciPath;
  uint64_t guid;
  int ptrSupport;
  int regIsGlobal;
  int forceFlush;
  int speed;
  int port;
  float latency;
  int maxComms;
  int maxRecvs;
  mcclNetDeviceType netDeviceType;
  int netDeviceVersion;
  mcclNetVDeviceProps_v11_t vProps;
  size_t maxP2pBytes;
  size_t maxCollBytes;
  int maxMultiRequestSize;
} mcclNetProperties_v11_t;

typedef struct {
  int32_t maxConcurrentPeers;
  int32_t minConcurrentPeers;
  int32_t maxFlowsPerPeer;
  int32_t minFlowsPerPeer;
} mcclNetCommAttr_v11_t;

typedef struct {
  mcclNetCommAttr_v11_t sendCommAttr;
  mcclNetCommAttr_v11_t recvCommAttr;
  uint32_t op;
  uint32_t algo;
  uint32_t proto;
} mcclNetAttr_v11_t;

typedef struct {
  char const* name;
  mcclResult_t (*init)(void** ctx, uint64_t commId,
                       mcclNetCommConfig_v11_t* config,
                       mcclDebugLogger_t logFunction,
                       mcclProfilerCallback_t profFunction);
  mcclResult_t (*devices)(int* ndev);
  mcclResult_t (*getProperties)(int dev, mcclNetProperties_v11_t* props);
  mcclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  mcclResult_t (*connect)(void* ctx, int dev, void* handle, void** sendComm,
                          mcclNetDeviceHandle_v11_t** sendDevComm);
  mcclResult_t (*accept)(void* listenComm, void** recvComm,
                         mcclNetDeviceHandle_v11_t** recvDevComm);
  mcclResult_t (*regMr)(void* comm, void* data, size_t size, int type,
                        void** mhandle);
  mcclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type,
                              uint64_t offset, int fd, void** mhandle);
  mcclResult_t (*deregMr)(void* comm, void* mhandle);
  mcclResult_t (*isend)(void* sendComm, void* data, size_t size, int tag,
                        void* mhandle, void* pHandle, void** request);
  mcclResult_t (*irecv)(void* recvComm, int n, void** data, size_t* sizes,
                        int* tags, void** mhandles, void** pHandles,
                        void** request);
  mcclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes,
                         void** mhandles, void** request);
  mcclResult_t (*test)(void* request, int* done, int* sizes);
  mcclResult_t (*closeSend)(void* sendComm);
  mcclResult_t (*closeRecv)(void* recvComm);
  mcclResult_t (*closeListen)(void* listenComm);
  mcclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptrMhandle);
  mcclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);
  mcclResult_t (*makeVDevice)(int* d, mcclNetVDeviceProps_v11_t* props);
  mcclResult_t (*finalize)(void* ctx);
  mcclResult_t (*setNetAttr)(void* ctx, mcclNetAttr_v11_t* netAttr);
} mcclNet_v11_t;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(mcclNet_v11_t) == 22 * sizeof(void*),
              "MCCL v11 net ABI field count changed");
#endif

#endif  // UCCL_MCCL_NET_ABI_H_
