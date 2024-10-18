// Copyright 2021 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef THIRD_PARTY_GPUS_NCCL_FASTSOCKET_PLUGIN_COMPAT_H_
#define THIRD_PARTY_GPUS_NCCL_FASTSOCKET_PLUGIN_COMPAT_H_

#include "nccl_net.h"

// ncclNet_v2_t: defined on earliest supported version, removed on 2.6.
#if (NCCL_MAJOR == 2 && NCCL_MINOR >= 6)

typedef struct {
  const char* name;
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*pci_path)(int dev, char** path);
  ncclResult_t (*ptr_support)(int dev, int* supportedTypes);
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  ncclResult_t (*reg_mr)(void* comm, void* data, int size, int type,
                         void** mhandle);
  ncclResult_t (*dereg_mr)(void* comm, void* mhandle);
  ncclResult_t (*isend)(void* sendComm, void* data, int size, void* mhandle,
                        void** request);
  ncclResult_t (*irecv)(void* recvComm, void* data, int size, void* mhandle,
                        void** request);
  ncclResult_t (*flush)(void* recvComm, void* data, int size, void* mhandle);
  ncclResult_t (*test)(void* request, int* done, int* size);
  ncclResult_t (*close_send)(void* sendComm);
  ncclResult_t (*close_recv)(void* recvComm);
  ncclResult_t (*close_listen)(void* listenComm);
} ncclNet_v2_t;

#endif

// ncclNet_v3_t: defined on 2.6, removed on 2.8.
#if (NCCL_MAJOR == 2 && NCCL_MINOR < 6) || (NCCL_MAJOR == 2 && NCCL_MINOR >= 8)

typedef ncclNetProperties_v6_t ncclNetProperties_v3_t;
typedef struct {
  const char* name;
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*get_properties)(int dev, ncclNetProperties_v3_t* props);
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  ncclResult_t (*reg_mr)(void* comm, void* data, int size, int type,
                         void** mhandle);
  ncclResult_t (*dereg_mr)(void* comm, void* mhandle);
  ncclResult_t (*isend)(void* sendComm, void* data, int size, void* mhandle,
                        void** request);
  ncclResult_t (*irecv)(void* recvComm, void* data, int size, void* mhandle,
                        void** request);
  ncclResult_t (*flush)(void* recvComm, void* data, int size, void* mhandle);
  ncclResult_t (*test)(void* request, int* done, int* size);
  ncclResult_t (*close_send)(void* sendComm);
  ncclResult_t (*close_recv)(void* recvComm);
  ncclResult_t (*close_listen)(void* listenComm);
} ncclNet_v3_t;

#endif

// ncclNet_v4_t: defined on 2.8, removed on 2.19.
#if (NCCL_MAJOR == 2 && NCCL_MINOR < 8) || (NCCL_MAJOR == 2 && NCCL_MINOR >= 19)

typedef ncclNetProperties_v6_t ncclNetProperties_v4_t;
typedef struct {
  const char* name;
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*get_properties)(int dev, ncclNetProperties_v4_t* props);
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  ncclResult_t (*reg_mr)(void* comm, void* data, int size, int type,
                         void** mhandle);
  ncclResult_t (*dereg_mr)(void* comm, void* mhandle);
  ncclResult_t (*isend)(void* sendComm, void* data, int size, void* mhandle,
                        void** request);
  ncclResult_t (*irecv)(void* recvComm, void* data, int size, void* mhandle,
                        void** request);
  ncclResult_t (*iflush)(void* recvComm, void* data, int size, void* mhandle,
                         void** request);
  ncclResult_t (*test)(void* request, int* done, int* size);
  ncclResult_t (*close_send)(void* sendComm);
  ncclResult_t (*close_recv)(void* recvComm);
  ncclResult_t (*close_listen)(void* listenComm);
  ncclResult_t (*calloc)(int dev, int size, int type, void** data,
                         void** mhandle);
  ncclResult_t (*free)(void* mhandle);
} ncclNet_v4_t;

#endif

// ncclNet_v5_t: defined on 2.12, live until now.
#if NCCL_MAJOR == 2 && NCCL_MINOR < 12

typedef struct {
  char* name;      // Used mostly for logging.
  char* pciPath;   // Path to the PCI device in /sys.
  uint64_t guid;   // Unique identifier for the NIC chip. Important for
                   // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport;  // NCCL_PTR_HOST or NCCL_PTR_HOST|NCCL_PTR_CUDA
  int speed;       // Port speed in Mbps.
  int port;        // Port number.
  float latency;   // Network latency
  int maxComms;    // Maximum number of comms we can create
  int maxRecvs;    // Maximum number of grouped receives.
} ncclNetProperties_v5_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*get_properties)(int dev, ncclNetProperties_v5_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*reg_mr)(void* comm, void* data, int size, int type,
                         void** mhandle);
  ncclResult_t (*dereg_mr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag,
                        void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes,
                        int* tags, void** mhandles, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes,
                         void** mhandles, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* sizes);
  ncclResult_t (*close_send)(void* sendComm);
  ncclResult_t (*close_recv)(void* recvComm);
  ncclResult_t (*close_listen)(void* listenComm);
} ncclNet_v5_t;

#endif

// ncclNet_v6_t: defined on 2.13, live until now.
#if NCCL_MAJOR == 2 && NCCL_MINOR < 13

typedef ncclNetProperties_v5_t ncclNetProperties_v6_t;
typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* comm, void* data, int size, int type,
                        void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type,
                              uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag,
                        void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes,
                        int* tags, void** mhandles, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes,
                         void** mhandles, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* sizes);
  // Close and free send/recv comm objects
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
  // Allocates a zero-initialized host buffer to be used by the net subsystem,
  // either allocated or registered through CUDA for host memory - For more
  // details, refer to cudaHostRegister(). The plugin can keep track of
  // allocation-related information by via mhandle. If this is is being used,
  // regMR and deregMr must be set to NULL.
  ncclResult_t (*calloc)(int dev, int size, int type, void** data,
                         void** mhandle);
  // Frees the buffer identified by mhandle. The buffer must have been allocated
  // by a previous alloc() call. It is legal to pass a nullptr to this function,
  // in which case it should return success and do nothing.
  ncclResult_t (*free)(void* mhandle);
} ncclNet_v6_t;

#endif

#endif  // THIRD_PARTY_GPUS_NCCL_FASTSOCKET_PLUGIN_COMPAT_H_
