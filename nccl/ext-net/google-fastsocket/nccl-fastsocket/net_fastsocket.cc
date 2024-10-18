// Copyright 2021 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE for license information
 ************************************************************************/

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/errqueue.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>

#include "nccl_net.h"
#include "compat.h"
#include "utilities.h"

#define MAX_INLINE_THRESHOLD 2048
#define MAX_SOCKETS 32
#define MAX_THREADS 16
#define MAX_REQUESTS 16
#define MAX_QUEUE_LEN MAX_REQUESTS
#define MAX_TASKS 6

#define MAX_FLOW_ENGINES 16
#define MAX_CONNECT_RETRY 1000

#define BUFFERED_CTRL
#define TX_ZCOPY

#ifndef SO_BUSY_POLL
#define SO_BUSY_POLL 46
#endif

#ifndef TCP_NOTSENT_LOWAT
#define TCP_NOTSENT_LOWAT 25
#endif

#ifndef SO_ZEROCOPY
#define SO_ZEROCOPY 60
#endif

#ifndef MSG_ZEROCOPY
#define MSG_ZEROCOPY 0x4000000
#endif

#ifndef SO_EE_ORIGIN_ZEROCOPY
#define SO_EE_ORIGIN_ZEROCOPY 5
#endif

#ifndef SO_EE_CODE_ZEROCOPY_COPIED
#define SO_EE_CODE_ZEROCOPY_COPIED 1
#endif

#ifndef SO_INCOMING_CPU
#define SO_INCOMING_CPU 49
#endif

#define NCCL_SOCKET_SEND 0
#define NCCL_SOCKET_RECV 1

#define HINT_BOTTLENECK

// Global variables
static int kNcclNetIfs = -1;
struct ncclSocketDev {
  union socketAddress addr;
  char dev_name[MAX_IF_NAME_SIZE];
  char* pci_path;
};
static struct ncclSocketDev kNcclSocketDevs[MAX_IFS];
pthread_mutex_t kNcclFastSocketLock = PTHREAD_MUTEX_INITIALIZER;
static int kEnableSpin = 0;
static int kInlineThreshold = 0;
static int kSockBusyPoll = 0;
static int kSockNotsentLowat = 0;
static int kSockSentbuf = 0;
static int kMinZcopySize = 0;
static int kDynamicChunkSize = 128 * 1024;
static int kEnableFlowPlacement = 0;
static int kNumFlowEngine = 4;

static int kTxCPUStart = -1;
static int kRxCPUStart = -1;
static int kQueueSkip = 0;

// Whether to enable the plugin. Default is enabled.
NCCL_PARAM(EnableFastSocket, "FAST_SOCKET_ENABLE", 1);

// Maximum chunk size in bytes for dynamic loading balancing.
// Default is 128 KB
NCCL_PARAM(DynamicChunkSize, "DYNAMIC_CHUNK_SIZE", 0);

// Whether to spin the helper thread. Default is disabled.
NCCL_PARAM(EnableThreadSpin, "THREAD_SPIN_ENABLE", 0);

// Maximum size of data to inline with a control message.
// 0 means disable inlining.
NCCL_PARAM(InlineThreshold, "INLINE_THRESHOLD", 0);

// Whether to busy poll the control socket. Default is disabled.
NCCL_PARAM(SockBusyPoll, "SOCK_BUSY_POLL", 0);

// Limit of unsent bytes in sockets: https://lwn.net/Articles/560082/
// The backpressure mechanism shifts the load to userspace, helping
// the load balancing algorithm better detect the load of each socket.
// 0 means disable backpressure.
NCCL_PARAM(SockNotsentLowat, "SOCK_NOTSENT_LOWAT", 0);

// Size of socket send buffer in bytes. 0 means the kernel default value.
NCCL_PARAM(SockSendBuf, "SOCK_SEND_BUF", 0);

// Minimum data size to use zero-copy. 0 means disabled.
NCCL_PARAM(MinZcopySize, "MIN_ZCOPY_SIZE", 0);

NCCL_PARAM(NsocksPerThread, "NSOCKS_PERTHREAD", -2);
NCCL_PARAM(NThreads, "SOCKET_NTHREADS", -2);

NCCL_PARAM(EnableFlowPlacement, "FLOW_PLACEMENT_ENABLE", 1);
NCCL_PARAM(NumFlowEngine, "NUM_FLOW_ENGINE", 4);

NCCL_PARAM(TxCPUStart, "TX_CPU_START", -2);
NCCL_PARAM(RxCPUStart, "RX_CPU_START", -2);
NCCL_PARAM(QueueSkip, "QUEUE_SKIP", 0);

static ncclResult_t socketSpin(int op, int fd, void* ptr, int size,
                               int* offset) {
  while (*offset < size)
    NCCLCHECK(socketProgressOpt(op, fd, ptr, size, offset, 0));
  return ncclSuccess;
}

// Data Structures
template <typename IndexType, typename IndexUnderType, typename ItemType,
          int MAX_ITEMS, int NSTATES>
struct ncclItemQueue {
  // 0: next dequeue slot, NSTATES - 1: next enqueue slot
  IndexType idx[NSTATES];
  ItemType items[MAX_ITEMS];
  ncclItemQueue() {
    for (int i = 0; i < NSTATES; ++i) idx[i] = 0;
  }

  bool empty() { return idx[0] == idx[NSTATES - 1]; }
  bool has_free() { return idx[NSTATES - 1] - idx[0] < MAX_ITEMS; }

  template <int STATE>
  bool has() {
    if (STATE == 0) return has_free();
    return idx[STATE] > idx[STATE - 1];
  }

  template <int STATE>
  ItemType* first() {
    if (STATE == 0) return items + idx[NSTATES - 1] % MAX_ITEMS;
    return items + idx[STATE - 1] % MAX_ITEMS;
  }

  template <int STATE>
  void advance() {
    ++idx[STATE - 1];
  }

  void enqueue() { ++idx[NSTATES - 1]; }
  void dequeue() { ++idx[0]; }

  // For cases when we need to iterate through all items in a state (other than
  // 0).
  template <int STATE>
  IndexUnderType get_iterator() {
    return idx[STATE - 1];
  }
  IndexUnderType next(IndexUnderType it) { return it + 1; }
  template <int STATE>
  bool is(IndexUnderType it) {
    return it < idx[STATE];
  }
  ItemType* to_item(IndexUnderType it) { return items + it % MAX_ITEMS; }
};

struct ncclCtrl {
  uint16_t type;
  uint16_t index;
  uint32_t size;
  uint32_t offset;
  uint32_t total;
} __attribute__((__packed__));

struct ncclSocketHandle {
  union socketAddress connect_addr;
  int num_socks;
  int num_threads;
};

struct ncclSocketRequest {
  struct ncclFastSocketComm* comm;
  void* data;
  int op;
  int next_sock_id;
  int next_size;
  int offset;
  int size;
  int size_pending;
};

struct ncclSocketTask {
  int op;
  int size;
  int offset;
  void* data;
  struct ncclSocketRequest* r;
#ifdef TX_ZCOPY
  uint32_t tx_count;
  uint32_t tx_bound;
#endif
  ncclResult_t result;
};

enum ThreadState { start, stop };
enum CtrlType {
  CTRL_NORMAL = 0,
  CTRL_INLINE = 1,
};

struct ncclSocketThreadResources {
  int id;  // thread index
  std::atomic_uint next;
  enum ThreadState state;
  struct ncclFastSocketComm* comm;
  pthread_mutex_t thread_lock;
  pthread_cond_t thread_cond;
};

// Must be identical to ncclSocketListenComm in net_socket.cc
struct ncclSocketListenComm {
  int fd;
  int num_socks;
  int num_threads;
};

// Request state transistion:
// FREE->ACTIVE->INACTIVE->FREE
enum {
  REQUEST_FREE = 0,
  REQUEST_INACTIVE = 1,
  REQUEST_ACTIVE = 2,
  REQUEST_MAX_STATES = 3,
};

struct ncclSocketRequestQueue
    : ncclItemQueue<uint32_t, uint32_t, struct ncclSocketRequest, MAX_REQUESTS,
                    REQUEST_MAX_STATES> {
  using Base = ncclItemQueue<uint32_t, uint32_t, struct ncclSocketRequest,
                             MAX_REQUESTS, REQUEST_MAX_STATES>;
  ncclSocketRequestQueue() : Base() {}
  bool has_active() { return has<REQUEST_ACTIVE>(); }
  bool has_inactive() { return has<REQUEST_INACTIVE>(); }
  struct ncclSocketRequest* next_free() { return first<REQUEST_FREE>(); }
  struct ncclSocketRequest* next_active() { return first<REQUEST_ACTIVE>(); }
  struct ncclSocketRequest* next_inactive() {
    return first<REQUEST_INACTIVE>();
  }
  void mark_inactive() { advance<REQUEST_ACTIVE>(); }
};

// Task state transistion:
// FREE->ACTIVE->INACTIVE->FREE
enum {
  TASK_FREE = 0,
  TASK_INACTIVE = 1,
  TASK_COMPLETING = 2,
  TASK_ACTIVE = 3,
  TASK_MAX_STATES = 4,
};

struct ncclSocketTaskQueue
    : ncclItemQueue<std::atomic_uint, unsigned, ncclSocketTask, MAX_TASKS,
                    TASK_MAX_STATES> {
  using Base = ncclItemQueue<std::atomic_uint, unsigned, ncclSocketTask,
                             MAX_TASKS, TASK_MAX_STATES>;
  ncclSocketTaskQueue() : Base() {}
  bool has_active() { return has<TASK_ACTIVE>(); }
  bool has_inactive() { return has<TASK_INACTIVE>(); }
  bool has_completing() { return has<TASK_COMPLETING>(); }
  ncclSocketTask* next_free() { return first<TASK_FREE>(); }
  ncclSocketTask* next_active() { return first<TASK_ACTIVE>(); }
  ncclSocketTask* next_completing() { return first<TASK_COMPLETING>(); }
  ncclSocketTask* next_inactive() { return first<TASK_INACTIVE>(); }
  void finish_active() { advance<TASK_ACTIVE>(); }
  void finish_completing() { advance<TASK_COMPLETING>(); }
};

template <unsigned BUF_SIZE>
struct ncclBufferedSendSocket {
  ncclBufferedSendSocket() : fd(-1), cur(0) {}
  void setFd(int fileFd) { fd = fileFd; }
  ncclResult_t sync() {
    if (cur == 0) return ncclSuccess;
    int off = 0;
    NCCLCHECK(socketSpin(NCCL_SOCKET_SEND, fd, buf, cur, &off));
    cur = 0;
    return ncclSuccess;
  }
  ncclResult_t send(void* ptr, unsigned s) {
    if (s > BUF_SIZE) return ncclInternalError;
    if (cur + s > BUF_SIZE) NCCLCHECK(sync());
    memcpy(buf + cur, ptr, s);
    cur += s;
    return ncclSuccess;
  }

  int fd;
  int cur;
  char buf[BUF_SIZE];
};

template <unsigned BUF_SIZE>
struct ncclBufferedRecvSocket {
  ncclBufferedRecvSocket() : fd(-1), cur(0), end(0) {}
  void setFd(int fileFd) { fd = fileFd; }
  bool empty() { return cur == end; }
  ncclResult_t refill() {
    if (!empty()) return ncclSuccess;
    cur = end = 0;
    return socketProgress(NCCL_SOCKET_RECV, fd, buf, BUF_SIZE, &end);
  }
  ncclResult_t recv(void* ptr, int s) {
    while (s) {
      refill();
      int len = std::min(s, end - cur);
      memcpy(ptr, buf + cur, len);
      cur += len;
      ptr = reinterpret_cast<char*>(ptr) + len;
      s -= len;
    }
    return ncclSuccess;
  }
  int brecv(void* ptr, int s) {
    int sz = std::min(s, end - cur);
    memcpy(ptr, buf + cur, sz);
    cur += sz;
    return sz;
  }

  int fd;
  int cur;
  int end;
  char buf[BUF_SIZE];
};

struct ncclFdData {
  int fd;
#ifdef TX_ZCOPY
  uint32_t tx_upper;
  uint32_t tx_lower;
#endif
  bool used;
  uint64_t stat;
  ncclSocketTaskQueue tasks;
};

struct ncclFastSocketComm {
  int ctrl_fd;  // control socket fd
  bool passive;
  std::atomic<bool> connected;
  struct ncclFdData
      fd_data[MAX_SOCKETS];   // data socket fd and its auxiliary data
  int num_socks;              // total number of socket fds per comm
  int num_threads;            // number of helper threads per comm
  int last_fd;                // the last enqueued fd idx
  ncclSocketRequestQueue rq;  // requests queue

#ifdef BUFFERED_CTRL
#define CTRL_BUFFER_SIZE (sizeof(ncclCtrl) * 8)
  ncclBufferedSendSocket<CTRL_BUFFER_SIZE> ctrl_send;
  ncclBufferedRecvSocket<CTRL_BUFFER_SIZE> ctrl_recv;
#endif

#ifdef HINT_BOTTLENECK
  struct timeval start_time;
#endif

  // helper threads
  pthread_t helper_thread[MAX_THREADS];
  pthread_t connect_thread;
  // auxiliary data with helper threads
  struct ncclSocketThreadResources thread_resource[MAX_THREADS];
  union socketAddress connect_addr;
};

// Control Path Functions
static inline void setSockBusyPoll(int fd) {
  if (kSockBusyPoll) {
    if (setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &kSockBusyPoll,
                   sizeof kSockBusyPoll) < 0) {
      WARN("Cannot enable socket busy poll");
    }
  }
}

static inline void setSockNotsentLowat(int fd) {
  if (kSockNotsentLowat) {
    if (setsockopt(fd, SOL_TCP, TCP_NOTSENT_LOWAT, &kSockNotsentLowat,
                   sizeof kSockNotsentLowat) < 0) {
      WARN("Cannot set socket TCP_NOTSENT_LOWAT");
    }
  }
}

static inline void setSockSendBuf(int fd) {
  if (kSockSentbuf) {
    if (setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &kSockSentbuf,
                   sizeof kSockSentbuf) < 0) {
      WARN("Cannot set socket SO_SNDBUF");
    }
  }
}

static inline void setSockZcopy(int fd) {
  if (kMinZcopySize > 0) {
    int one = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_ZEROCOPY, &one, sizeof one) < 0) {
      WARN("Cannot set socket to SO_ZEROCOPY");
      kMinZcopySize = 0;
    }
  }
}

static ncclResult_t ncclFastSocketGetPciPath(char* devName, char** pciPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/net/%s/device", devName);
  // May return NULL if the file doesn't exist.
  *pciPath = realpath(devicePath, nullptr);
  return ncclSuccess;
}

ncclResult_t ncclFastSocketPciPath(int dev, char** pciPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/net/%s/device",
           kNcclSocketDevs[dev].dev_name);
  // May return NULL if the file doesn't exist.
  *pciPath = realpath(devicePath, nullptr);
  if (*pciPath == nullptr) {
    INFO(NCCL_NET | NCCL_INIT, "Could not find real path of %s", devicePath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclFastSocketInit(ncclDebugLogger_t logFunction) {
  int enable = ncclParamEnableFastSocket();
  nccl_log_func = logFunction;
#ifdef CHECK_COLLNET_ENABLE
  char* collnet_enable = getenv("NCCL_COLLNET_ENABLE");
  if (!collnet_enable || strcmp(collnet_enable, "0") == 0) {
    enable = 0;
  }
#endif
  if (!enable) {
    INFO(NCCL_NET | NCCL_INIT, "NET/FastSocket disabled");
    return ncclInternalError;
  }

  int dcs = ncclParamDynamicChunkSize();
  if (dcs > 0) kDynamicChunkSize = dcs;

  kInlineThreshold = ncclParamInlineThreshold();
  if (kInlineThreshold < 0) kInlineThreshold = 0;
  if (kInlineThreshold > MAX_INLINE_THRESHOLD)
    kInlineThreshold = MAX_INLINE_THRESHOLD;

  kEnableSpin = ncclParamEnableThreadSpin();

  kSockBusyPoll = ncclParamSockBusyPoll();
  if (kSockBusyPoll < 0) kSockBusyPoll = 0;

  int snl = ncclParamSockNotsentLowat();
  if (snl > 0) kSockNotsentLowat = snl;

  kSockSentbuf = ncclParamSockSendBuf();
  if (kSockSentbuf < 0) kSockSentbuf = 0;
  kMinZcopySize = ncclParamMinZcopySize();

  kTxCPUStart = ncclParamTxCPUStart();
  kRxCPUStart = ncclParamRxCPUStart();
  INFO(NCCL_INIT | NCCL_NET, "NET/FastSocket : Tx CPU start: %d", kTxCPUStart);
  INFO(NCCL_INIT | NCCL_NET, "NET/FastSocket : Rx CPU start: %d", kRxCPUStart);

  kEnableFlowPlacement = ncclParamEnableFlowPlacement();
  if (kEnableFlowPlacement) {
    INFO(NCCL_INIT | NCCL_NET, "NET/FastSocket : Flow placement enabled.");
    kNumFlowEngine = ncclParamNumFlowEngine();
    if (kNumFlowEngine < 1) kNumFlowEngine = 1;
    if (kNumFlowEngine > MAX_FLOW_ENGINES) kNumFlowEngine = MAX_FLOW_ENGINES;
  }

  kQueueSkip = ncclParamQueueSkip();
  INFO(NCCL_INIT | NCCL_NET, "NET/FastSocket : queue skip: %d", kQueueSkip);

  if (kNcclNetIfs == -1) {
    pthread_mutex_lock(&kNcclFastSocketLock);
    if (kNcclNetIfs == -1) {
      char names[MAX_IF_NAME_SIZE * MAX_IFS];
      union socketAddress addrs[MAX_IFS];
      kNcclNetIfs = findInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (kNcclNetIfs <= 0) {
        WARN("NET/FastSocket : no interface found");
        pthread_mutex_unlock(&kNcclFastSocketLock);
        return ncclInternalError;
      } else {
        char line[2048];
        char addrline[2048];
        line[0] = '\0';
        for (int i = 0; i < kNcclNetIfs; i++) {
          strncpy(kNcclSocketDevs[i].dev_name, names + i * MAX_IF_NAME_SIZE,
                  MAX_IF_NAME_SIZE);
          memcpy(&kNcclSocketDevs[i].addr, addrs + i,
                 sizeof(union socketAddress));
          NCCLCHECK(ncclFastSocketGetPciPath(kNcclSocketDevs[i].dev_name,
                                             &kNcclSocketDevs[i].pci_path));
          snprintf(line + strlen(line), 2047 - strlen(line), " [%d]%s:%s", i,
                   names + i * MAX_IF_NAME_SIZE,
                   socketToString(&addrs[i].sa, addrline));
        }
        line[2047] = '\0';
        INFO(NCCL_INIT | NCCL_NET, "NET/FastSocket : Using%s", line);
      }
    }
    pthread_mutex_unlock(&kNcclFastSocketLock);
  }
  INFO(NCCL_INIT | NCCL_NET, "NET/FastSocket plugin initialized");
  return ncclSuccess;
}

ncclResult_t ncclFastSocketDevices(int* ndev) {
  *ndev = kNcclNetIfs;
  return ncclSuccess;
}

static ncclResult_t ncclFlowPlacementGetNsockNthread(int* ns, int* nt) {
  *ns = kNumFlowEngine;
  *nt = *ns;
  INFO(NCCL_NET, "Flow placement forcing parameters: nthreads %d nsocks %d",
       *nt, *ns);
  return ncclSuccess;
}

static ncclResult_t ncclFastSocketGetNsockNthread(int dev, int* ns, int* nt) {
  if (kEnableFlowPlacement) {
    return ncclFlowPlacementGetNsockNthread(ns, nt);
  }
  int nSocksPerThread = ncclParamNsocksPerThread();
  int nThreads = ncclParamNThreads();
  if (nThreads > MAX_THREADS) {
    WARN(
        "NET/Socket : NCCL_SOCKET_NTHREADS is greater than the maximum "
        "allowed, setting to %d",
        MAX_THREADS);
    nThreads = MAX_THREADS;
  }
  if (nThreads == -2 || nSocksPerThread == -2) {
    // Auto-detection
    int autoNt = 1, autoNs = 1;
    char vendorPath[PATH_MAX];
    snprintf(vendorPath, PATH_MAX, "/sys/class/net/%s/device/vendor",
             kNcclSocketDevs[dev].dev_name);
    char* rPath = realpath(vendorPath, nullptr);
    int fd = open(rPath, O_RDONLY);
    free(rPath);
    if (fd == -1) {
      // Could not find device vendor. This is handled silently so
      // we don't want to print an INFO error.
      INFO(NCCL_NET, "Open of %s failed : %s\n", vendorPath, strerror(errno));
      goto end;
    }
    char vendor[7];
    strncpy(vendor, "0x0000", 7);
    int len;
    SYSCHECKVAL(read(fd, vendor, 6), "read", len);
    SYSCHECK(close(fd), "close");
    if (strcmp(vendor, "0x1d0f") == 0) {  // AWS
      autoNt = 2;
      autoNs = 8;
    } else if (strcmp(vendor, "0x1ae0") == 0) {  // GCP
      autoNt = 6;
      autoNs = 1;
    }
  end:
    if (nThreads == -2) nThreads = autoNt;
    if (nSocksPerThread == -2) nSocksPerThread = autoNs;
  }
  int nSocks = nSocksPerThread * nThreads;
  if (nSocks > MAX_SOCKETS) {
    nSocksPerThread = MAX_SOCKETS / nThreads;
    WARN(
        "NET/Socket : the total number of sockets is greater than the maximum "
        "allowed, setting NCCL_NSOCKS_PERTHREAD to %d",
        nSocksPerThread);
    nSocks = nSocksPerThread * nThreads;
  }
  *ns = nSocks;
  *nt = nThreads;
  INFO(NCCL_INIT, "NET/Socket: Using %d threads and %d sockets per thread",
       nThreads, nSocksPerThread);
  return ncclSuccess;
}

ncclResult_t ncclFastSocketNewComm(struct ncclFastSocketComm** comm) {
  NCCLCHECK(ncclCalloc(comm, 1));
  (*comm)->ctrl_fd = -1;
  (*comm)->last_fd = 0;
  for (int i = 0; i < MAX_SOCKETS; i++) {
    (*comm)->fd_data[i].fd = -1;
    (*comm)->fd_data[i].used = false;
    (*comm)->fd_data[i].stat = 0;
#ifdef TX_ZCOPY
    (*comm)->fd_data[i].tx_upper = 0;
    (*comm)->fd_data[i].tx_lower = 0;
#endif
  }
  gettimeofday(&(*comm)->start_time, nullptr);
  return ncclSuccess;
}

static ncclResult_t ncclSocketNewListenComm(
    struct ncclSocketListenComm** comm) {
  NCCLCHECK(ncclCalloc(comm, 1));
  (*comm)->fd = -1;
  return ncclSuccess;
}

static ncclResult_t GetSocketAddr(int dev, union socketAddress* addr) {
  if (dev >= kNcclNetIfs) return ncclInternalError;
  memcpy(addr, &kNcclSocketDevs[dev].addr, sizeof(*addr));
  return ncclSuccess;
}

ncclResult_t ncclFastSocketListen(int dev, void* opaqueHandle,
                                  void** listenComm) {
  if (dev < 0) {  // data transfer socket is based on specified dev
    return ncclInternalError;
  }
  struct ncclSocketHandle* handle =
      static_cast<struct ncclSocketHandle*>(opaqueHandle);
  static_assert(sizeof(struct ncclSocketHandle) < NCCL_NET_HANDLE_MAXSIZE,
                "ncclSocketHandle size too large");
  struct ncclSocketListenComm* comm;
  NCCLCHECK(ncclSocketNewListenComm(&comm));
  NCCLCHECK(GetSocketAddr(dev, &handle->connect_addr));
  NCCLCHECK(createListenSocket(&comm->fd, &handle->connect_addr));
  NCCLCHECK(
      ncclFastSocketGetNsockNthread(dev, &comm->num_socks, &comm->num_threads));
  handle->num_socks = comm->num_socks;
  handle->num_threads = comm->num_threads;
  *listenComm = comm;
  return ncclSuccess;
}

static void initCtrlFd(struct ncclFastSocketComm* comm, int fd) {
  comm->ctrl_fd = fd;
#ifdef BUFFERED_CTRL
  comm->ctrl_send.setFd(fd);
  comm->ctrl_recv.setFd(fd);
#endif
}

void waitConnect(struct ncclFastSocketComm* comm) {
  while (!comm->connected) {
    pthread_yield();
  }
}

ncclResult_t ncclSocketAsyncConnectV2(struct ncclFastSocketComm* comm) {
  int i = 0;
  int retry = 0;
  while (i < comm->num_socks + 1) {
    int tmpFd, offset;
    NCCLCHECK(connectAddress(&tmpFd, &comm->connect_addr));
    if (i == comm->num_socks) {
      int ii = 0;
      offset = 0;
      NCCLCHECK(socketWait(NCCL_SOCKET_RECV, tmpFd, &ii, sizeof(int), &offset));
      initCtrlFd(comm, tmpFd);
    } else {
      int qid, dqid;
      int rqid = 0;
      int cpu = 0;
      socklen_t opt_len = sizeof cpu;

      if (retry < MAX_CONNECT_RETRY) {
        if (getsockopt(tmpFd, SOL_SOCKET, SO_INCOMING_CPU, &cpu, &opt_len) <
            0) {
          WARN("Cannot get incoming CPU.");
        }

        qid = cpu % kNumFlowEngine;
        dqid = cpu % (kNumFlowEngine * 2);
        if (cpu < kQueueSkip || dqid >= kNumFlowEngine ||
            comm->fd_data[qid].used) {
          qid = -1;
        }
      } else {
        int j = 0;
        while (j < comm->num_socks) {
          if (!comm->fd_data[j].used) break;
          ++j;
        }
        if (j == comm->num_socks) {
          WARN("Cannot find empty socket for %d.", i);
          return ncclInternalError;
        }
        dqid = j;
        qid = j;
        if (retry == MAX_CONNECT_RETRY) {
          WARN("Maximum retry reached for connect %d.", i);
        }
      }

      offset = 0;
      NCCLCHECK(
          socketWait(NCCL_SOCKET_RECV, tmpFd, &rqid, sizeof(int), &offset));
      offset = 0;
      NCCLCHECK(
          socketWait(NCCL_SOCKET_SEND, tmpFd, &qid, sizeof(int), &offset));
      if (qid < 0 || rqid < 0) {
        close(tmpFd);
        ++retry;
        continue;
      }

      INFO(NCCL_INIT | NCCL_NET, "connect incoming cpu: %u", cpu);
      INFO(NCCL_INIT | NCCL_NET, "connect qid: %d, rqid: %d", qid, rqid);

      setSockNotsentLowat(tmpFd);
      setSockSendBuf(tmpFd);
      setSockZcopy(tmpFd);
      comm->fd_data[rqid].fd = tmpFd;
      comm->fd_data[qid].used = true;  // qid, not rqid
      INFO(NCCL_INIT | NCCL_NET, "NET/FastSocket : Connected after %d retries.",
           retry);
      INFO(NCCL_INIT | NCCL_NET, "NET/FastSocket : Connected data socket %d",
           i);
      retry = 0;
    }
    ++i;
  }
  setSockBusyPoll(comm->ctrl_fd);
  INFO(NCCL_INIT | NCCL_NET, "NET/FastSocket : Async connect done");
  comm->connected = true;
  return ncclSuccess;
}

void* asyncConnect(void* opaque) {
  ncclSocketAsyncConnectV2(static_cast<struct ncclFastSocketComm*>(opaque));
  return nullptr;
}

ncclResult_t ncclSocketConnectV2(int dev, void* opaqueHandle, void** sendComm) {
  if (dev < 0) {  // data transfer socket is based on specified dev
    return ncclInternalError;
  }
  struct ncclFastSocketComm* comm;
  NCCLCHECK(ncclFastSocketNewComm(&comm));
  struct ncclSocketHandle* handle =
      static_cast<struct ncclSocketHandle*>(opaqueHandle);
  comm->num_socks = handle->num_socks;
  comm->num_threads = handle->num_threads;
  comm->connect_addr = handle->connect_addr;
  comm->passive = false;
  comm->connected = false;

  pthread_create(&comm->connect_thread, nullptr, asyncConnect,
                 reinterpret_cast<void*>(comm));
  pthread_detach(comm->connect_thread);
  *sendComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclSocketAcceptV2(void* listenComm, void** recvComm) {
  struct ncclSocketListenComm* lComm =
      static_cast<struct ncclSocketListenComm*>(listenComm);
  struct ncclFastSocketComm* rComm;
  NCCLCHECK(ncclFastSocketNewComm(&rComm));
  rComm->num_socks = lComm->num_socks;
  rComm->num_threads = lComm->num_threads;
  rComm->passive = true;
  int i = 0;
  int retry = 0;
  while (i < rComm->num_socks + 1) {
    int tmpFd, offset;
    struct sockaddr_in sockaddr;
    socklen_t socklen = sizeof(struct sockaddr_in);
    SYSCHECKVAL(accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen),
                "accept", tmpFd);
    if (i == rComm->num_socks) {
      offset = 0;
      NCCLCHECK(socketWait(NCCL_SOCKET_SEND, tmpFd, &i, sizeof(int), &offset));
      initCtrlFd(rComm, tmpFd);
    } else {
      unsigned cpu = 0;
      int qid, dqid;
      int rqid;
      socklen_t opt_len = sizeof cpu;

      if (retry < MAX_CONNECT_RETRY) {
        if (getsockopt(tmpFd, SOL_SOCKET, SO_INCOMING_CPU, &cpu, &opt_len) <
            0) {
          WARN("Cannot get incoming CPU.");
        }
        qid = static_cast<int>(cpu) % kNumFlowEngine;
        dqid = static_cast<int>(cpu) % (kNumFlowEngine * 2);
        if (dqid < kNumFlowEngine || rComm->fd_data[qid].used) {
          qid = -1;
        }
      } else {
        int j = 0;
        while (j < rComm->num_socks) {
          if (!rComm->fd_data[j].used) break;
          ++j;
        }
        if (j == rComm->num_socks) {
          WARN("Cannot find empty socket for %d.", i);
          return ncclInternalError;
        }
        qid = j;
        dqid = j + kNumFlowEngine;
        if (retry == MAX_CONNECT_RETRY) {
          WARN("Maximum retry reached for accept %d.", i);
        }
      }

      offset = 0;
      NCCLCHECK(
          socketWait(NCCL_SOCKET_SEND, tmpFd, &qid, sizeof(int), &offset));
      rqid = 0;
      offset = 0;
      NCCLCHECK(
          socketWait(NCCL_SOCKET_RECV, tmpFd, &rqid, sizeof(int), &offset));
      if (qid < 0 || rqid < 0) {
        close(tmpFd);
        ++retry;
        continue;
      }

      INFO(NCCL_INIT | NCCL_NET, "accept qid: %d, rqid: %d", qid, rqid);
      INFO(NCCL_INIT | NCCL_NET, "accept incoming cpu: %u", cpu);

      setSockNotsentLowat(tmpFd);
      setSockSendBuf(tmpFd);
      setSockZcopy(tmpFd);
      rComm->fd_data[qid].fd = tmpFd;
      rComm->fd_data[qid].used = true;
      INFO(NCCL_INIT | NCCL_NET, "NET/FastSocket : Connected after %d retries.",
           retry);
      INFO(NCCL_INIT | NCCL_NET, "NET/FastSocket : Accepted data socket %d", i);
      retry = 0;
    }
    ++i;
  }
  setSockBusyPoll(rComm->ctrl_fd);
  *recvComm = rComm;
  rComm->connected = true;
  return ncclSuccess;
}

ncclResult_t ncclFastSocketConnect(int dev, void* opaqueHandle,
                                   void** sendComm) {
  if (dev < 0) {  // data transfer socket is based on specified dev
    return ncclInternalError;
  }
  if (kEnableFlowPlacement) {
    return ncclSocketConnectV2(dev, opaqueHandle, sendComm);
  }
  struct ncclFastSocketComm* comm;
  NCCLCHECK(ncclFastSocketNewComm(&comm));
  struct ncclSocketHandle* handle =
      static_cast<struct ncclSocketHandle*>(opaqueHandle);
  comm->num_socks = handle->num_socks;
  comm->num_threads = handle->num_threads;
  for (int i = 0; i < comm->num_socks + 1; i++) {
    int tmpFd, offset = 0;
    NCCLCHECK(connectAddress(&tmpFd, &handle->connect_addr));
    NCCLCHECK(socketWait(NCCL_SOCKET_SEND, tmpFd, &i, sizeof(int), &offset));
    if (i == comm->num_socks) {
      initCtrlFd(comm, tmpFd);
    } else {
      setSockNotsentLowat(tmpFd);
      setSockSendBuf(tmpFd);
      setSockZcopy(tmpFd);
      comm->fd_data[i].fd = tmpFd;
    }
  }
  setSockBusyPoll(comm->ctrl_fd);
  *sendComm = comm;
  comm->passive = false;
  comm->connected = true;
  return ncclSuccess;
}

ncclResult_t ncclFastSocketAccept(void* listenComm, void** recvComm) {
  if (kEnableFlowPlacement) {
    return ncclSocketAcceptV2(listenComm, recvComm);
  }
  struct ncclSocketListenComm* lComm =
      static_cast<struct ncclSocketListenComm*>(listenComm);
  struct ncclFastSocketComm* rComm;
  NCCLCHECK(ncclFastSocketNewComm(&rComm));
  rComm->num_socks = lComm->num_socks;
  rComm->num_threads = lComm->num_threads;
  for (int i = 0; i < rComm->num_socks + 1; i++) {
    int tmpFd, sendSockIdx, offset = 0;
    struct sockaddr_in sockaddr;
    socklen_t socklen = sizeof(struct sockaddr_in);
    SYSCHECKVAL(accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen),
                "accept", tmpFd);
    NCCLCHECK(socketWait(NCCL_SOCKET_RECV, tmpFd, &sendSockIdx, sizeof(int),
                         &offset));
    if (sendSockIdx == rComm->num_socks) {
      initCtrlFd(rComm, tmpFd);
    } else {
      setSockNotsentLowat(tmpFd);
      setSockSendBuf(tmpFd);
      setSockZcopy(tmpFd);
      rComm->fd_data[sendSockIdx].fd = tmpFd;
    }
  }
  setSockBusyPoll(rComm->ctrl_fd);
  *recvComm = rComm;
  rComm->passive = true;
  rComm->connected = true;
  return ncclSuccess;
}

ncclResult_t ncclFastSocketClose(void* opaqueComm) {
  struct ncclFastSocketComm* comm =
      static_cast<struct ncclFastSocketComm*>(opaqueComm);
  if (comm) {
    for (int i = 0; i < comm->num_threads; i++) {
      struct ncclSocketThreadResources* res = comm->thread_resource + i;
      if (comm->helper_thread[i]) {
        pthread_mutex_lock(&res->thread_lock);
        res->state = stop;
        pthread_cond_signal(&res->thread_cond);
        pthread_mutex_unlock(&res->thread_lock);
        pthread_join(comm->helper_thread[i], nullptr);
      }
    }
    if (comm->ctrl_fd != -1) close(comm->ctrl_fd);
    uint64_t total = 0;
    for (int i = 0; i < comm->num_socks; i++) {
      if (comm->fd_data[i].fd != -1) close(comm->fd_data[i].fd);
      if (comm->fd_data[i].stat) {
        INFO(NCCL_NET, "Socket %i total bytes: %lu, passive = %d", i,
             comm->fd_data[i].stat, (int)comm->passive);
        total += comm->fd_data[i].stat;
      }
    }
    INFO(NCCL_NET, "All bytes: %lu", total);
#ifdef HINT_BOTTLENECK
    struct timeval current_time;
    gettimeofday(&current_time, nullptr);
    timersub(&current_time, &comm->start_time, &current_time);
    double avg_throughput_mb =
        (double)total / (1e6 * current_time.tv_sec + current_time.tv_usec);
    if (avg_throughput_mb > 1000) {
      INFO(NCCL_INIT, "Average throughput: %f MB/s", avg_throughput_mb);
      INFO(NCCL_INIT,
           "This training job might be network bound. Reduction Server boosts "
           "performance of network bound training jobs. "
           "More details at "
           "https://cloud.google.com/blog/products/ai-machine-learning/"
           "faster-distributed-training-with-google-clouds-reduction-server.");
    }
#endif
    free(comm);
  }
  return ncclSuccess;
}

// Data-path functions
#ifdef TX_ZCOPY
static int taskProgress(int fd, struct ncclSocketTask* t) {
  int bytes = 0;
  char* data = reinterpret_cast<char*>(t->data);
  int count = 0;
  do {
    int s = t->size - t->offset;
    int flags = MSG_DONTWAIT;
    int op = t->op;
    if (op == NCCL_SOCKET_SEND && kMinZcopySize > 0 && s >= kMinZcopySize)
      flags |= MSG_ZEROCOPY;
    if (op == NCCL_SOCKET_RECV) bytes = recv(fd, data + t->offset, s, flags);
    if (op == NCCL_SOCKET_SEND) bytes = send(fd, data + t->offset, s, flags);

    if (op == NCCL_SOCKET_RECV && bytes == 0) {
      WARN("Net : Connection closed by remote peer");
      return -1;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        WARN("Call to socket op %d flags %x failed : %s", op, flags,
             strerror(errno));
        if (flags & MSG_ZEROCOPY) {
          WARN("Turning off TX zero copy");
          kMinZcopySize = 0;
          bytes = 0;
        } else {
          return -1;
        }
      } else {
        bytes = 0;
      }
    }
    t->offset += bytes;
    if (bytes && (flags & MSG_ZEROCOPY)) ++count;
  } while (bytes > 0 && t->offset < t->size);
  return count;
}

static int readNotification(struct msghdr* msg, uint32_t* lower,
                            uint32_t* upper) {
  struct sock_extended_err* serr;
  struct cmsghdr* cm;
  cm = CMSG_FIRSTHDR(msg);
  if (cm->cmsg_level != SOL_IP && cm->cmsg_type != IP_RECVERR) {
    WARN("Invalid message level %d or type %d from errorqueue!",
         (int)cm->cmsg_level, (int)cm->cmsg_type);
    return -1;
  }
  serr = reinterpret_cast<struct sock_extended_err*>(CMSG_DATA(cm));
  if (serr->ee_errno != 0 || serr->ee_origin != SO_EE_ORIGIN_ZEROCOPY) {
    WARN("Invalid message errno %d or origin %d from errorqueue!",
         (int)serr->ee_errno, (int)serr->ee_origin);
    return -1;
  }
  *lower = serr->ee_info;
  *upper = serr->ee_data + 1;
  return 0;
}

static int readErrqueue(int fd, uint32_t* lower, uint32_t* upper) {
  char control[100];
  struct msghdr msg = {};
  msg.msg_control = control;
  msg.msg_controllen = sizeof control;
  int ret = recvmsg(fd, &msg, MSG_ERRQUEUE);
  if (ret < 0 && errno == EAGAIN) return 0;
  if (ret < 0) {
    WARN("Read error from errqueue: %d", errno);
    return -errno;
  }
  ret = readNotification(&msg, lower, upper);
  if (ret < 0) return ret;
  return *upper - *lower;
}

void processCompletion(ncclSocketTaskQueue* tasks, uint32_t clower,
                       uint32_t lower, uint32_t upper) {
  auto it = tasks->get_iterator<TASK_COMPLETING>();
  while (lower < upper && tasks->is<TASK_COMPLETING>(it)) {
    ncclSocketTask* r = tasks->to_item(it);
    uint32_t cupper = r->tx_bound;
    uint32_t left = std::max(clower, lower);
    uint32_t right = std::min(cupper, upper);
    if (right > left) {
      r->tx_count += right - left;
    }
    lower = std::max(lower, cupper);
    clower = cupper;
    it = tasks->next(it);
  }
  if (lower < upper && tasks->is<TASK_ACTIVE>(it)) {
    ncclSocketTask* r = tasks->to_item(it);
    r->tx_count += upper - lower;
  }
}
#endif

static void* persistentSocketThread(void* args_) {
  struct ncclSocketThreadResources* resource =
      static_cast<struct ncclSocketThreadResources*>(args_);
  struct ncclFastSocketComm* comm = resource->comm;
  volatile enum ThreadState* state = &resource->state;
  int nSocksPerThread = comm->num_socks / comm->num_threads;
  int tid = resource->id;
  unsigned int mark = 0;
  int core = comm->passive ? kRxCPUStart : kTxCPUStart;

  INFO(NCCL_INIT | NCCL_NET, "Comm %p thread %d started", comm, tid);
  if (core >= 0) {
    cpu_set_t my_set;
    core += tid;
    CPU_ZERO(&my_set);
    CPU_SET(core, &my_set);
    sched_setaffinity(0, sizeof my_set, &my_set);
  }
  INFO(NCCL_INIT | NCCL_NET, "Comm %p thread %d binding to core %d", comm, tid,
       core);
  while (true) {
    int idle = 1;
    // iterate all the sockets associate with the current thread
    for (int i = 0; i < nSocksPerThread; ++i) {
      // int idx = i + tid * nSocksPerThread; // sequential access
      int idx = tid + i * comm->num_threads;  // strided access
#ifdef TX_ZCOPY
      struct ncclFdData* fd_data = comm->fd_data + idx;
      ncclSocketTaskQueue* tasks = &(fd_data->tasks);
      if (tasks->has_active()) {
        struct ncclSocketTask* r = fd_data->tasks.next_active();
        int old_offset = r->offset;
        int cnt = taskProgress(fd_data->fd, r);
        if (cnt < 0) return nullptr;
        fd_data->tx_upper += cnt;
        if (r->op == NCCL_SOCKET_SEND) fd_data->stat += r->offset - old_offset;
        if (r->offset == r->size) {
          r->tx_bound = fd_data->tx_upper;
          tasks->finish_active();
        }
        idle = 0;
      }

      // poll errqueue for send completion
      if (fd_data->tx_upper > fd_data->tx_lower) {
        uint32_t lower, upper;
        while (true) {
          int ret = readErrqueue(fd_data->fd, &lower, &upper);
          if (ret == 0) break;
          if (ret < 0) return nullptr;
          processCompletion(tasks, fd_data->tx_lower, lower, upper);
        }
        idle = 0;
      }

      if (tasks->has_completing()) {
        struct ncclSocketTask* r = tasks->next_completing();
        if (r->tx_count == r->tx_bound - fd_data->tx_lower) {
          fd_data->tx_lower = r->tx_bound;
          tasks->finish_completing();
        }
        idle = 0;
      }
#else
      if (!comm->fdData[idx].tasks.has_active()) continue;
      struct ncclSocketTask* r = comm->fdData[idx].tasks.next_active();
      int fd = comm->fdData[idx].fd;
      if (r->offset < r->size) {
        int old_offset = r->offset;
        r->result = socketProgress(r->op, fd, r->data, r->size, &r->offset);
        if (r->result != ncclSuccess) {
          WARN("NET/Socket : socket progress error");
          return NULL;
        }
        if (r->op == NCCL_SOCKET_SEND)
          comm->fdData[idx].stat += r->offset - old_offset;
        if (r->offset == r->size) {
          comm->fdData[idx].tasks.finish_active();
          comm->fdData[idx].tasks.finish_completing();
        }
        idle = 0;
      }
#endif
    }
    if (kEnableSpin) idle = 0;
    if (idle) {
      pthread_mutex_lock(&resource->thread_lock);
      while (mark == resource->next && *state != stop) {  // no new tasks, wait
        pthread_cond_wait(&resource->thread_cond, &resource->thread_lock);
      }
      mark = resource->next;
      pthread_mutex_unlock(&resource->thread_lock);
    }
    if (*state == stop) return nullptr;
  }
}

static ncclResult_t ncclFastSocketGetRequest(struct ncclFastSocketComm* comm,
                                             int op, void* data, int size,
                                             struct ncclSocketRequest** req) {
  if (!comm->rq.has_free()) {
    WARN("NET/Socket : unable to allocate requests");
    return ncclInternalError;
  }
  struct ncclSocketRequest* r = comm->rq.next_free();
  r->op = op;
  r->next_sock_id = -1;
  r->next_size = 0;
  r->data = data;
  r->offset = 0;
  r->size = size;
  if (op == NCCL_SOCKET_SEND)
    r->size_pending = size;
  else
    r->size_pending = -1;
  r->comm = comm;
  *req = r;
  comm->rq.enqueue();
  return ncclSuccess;
}

#define CTRL_DONE(r) ((r)->next_sock_id >= 0)
#define RESET_CTRL(r) ((r)->next_sock_id = -1)

#define REQUEST_DONE(r) \
  (((r)->size == 0 && CTRL_DONE(r)) || ((r)->size && (r)->size_pending == 0))
#define REQUEST_INACTIVE(r) ((r)->size == (r)->offset)

#ifndef BUFFERED_CTRL
static ncclResult_t ncclProcessCtrl(struct ncclFastSocketComm* comm,
                                    struct ncclSocketRequest* r,
                                    struct ncclCtrl* ctrl) {
  int s = 0;
  NCCLCHECK(socketSpin(r->op, comm->ctrl_fd, ctrl, sizeof *ctrl, &s));
  if (s == 0) return ncclSuccess;
  if (s < sizeof *ctrl) {
    NCCLCHECK(socketSpin(r->op, comm->ctrl_fd, ctrl, sizeof *ctrl, &s));
  }
  if (s) {
    // save control information to request
    r->next_sock_id = ctrl->index;
    r->next_size = ctrl->size;
    if (r->size_pending < 0) {
      r->size_pending = r->size = ctrl->total;
    }
  }
  return ncclSuccess;
}
#endif

static ncclResult_t ncclCtrlRecv(struct ncclFastSocketComm* comm,
                                 struct ncclSocketRequest* r,
                                 struct ncclCtrl* ctrl) {
#ifdef BUFFERED_CTRL
  NCCLCHECK(comm->ctrl_recv.refill());
  if (comm->ctrl_recv.empty()) return ncclSuccess;
  NCCLCHECK(comm->ctrl_recv.recv(ctrl, sizeof *ctrl));
  // save control information to request
  r->next_sock_id = ctrl->index;
  r->next_size = ctrl->size;
  if (r->size_pending < 0) {
    r->size_pending = r->size = ctrl->total;
  }
  return ncclSuccess;
#else
  return ncclProcessCtrl(comm, r, ctrl);
#endif
}

static inline ncclResult_t ncclCtrlSendSync(struct ncclFastSocketComm* comm) {
#ifdef BUFFERED_CTRL
  NCCLCHECK(comm->ctrl_send.sync());
#endif
  return ncclSuccess;
}

static inline ncclResult_t ncclCtrlSend(struct ncclFastSocketComm* comm,
                                        struct ncclSocketRequest* r,
                                        struct ncclCtrl* ctrl) {
#ifdef BUFFERED_CTRL
  NCCLCHECK(comm->ctrl_send.send(ctrl, sizeof *ctrl));
  r->next_sock_id = ctrl->index;
  r->next_size = ctrl->size;
  return ncclSuccess;
#else
  return ncclProcessCtrl(comm, r, ctrl);
#endif
}

static void enqueueTask(struct ncclFastSocketComm* comm,
                        struct ncclSocketRequest* r) {
  int sockId = r->next_sock_id;
  RESET_CTRL(r);
  int sz = r->next_size;
  struct ncclSocketTask* task = comm->fd_data[sockId].tasks.next_free();
  task->op = r->op;
  task->data = reinterpret_cast<char*>(r->data) + r->offset;
  task->r = r;
  task->result = ncclSuccess;
  task->offset = 0;
  task->size = sz;
#ifdef TX_ZCOPY
  task->tx_count = 0;
#endif
  comm->fd_data[sockId].tasks.enqueue();

  r->offset += sz;
  if (REQUEST_INACTIVE(r)) {
    comm->rq.mark_inactive();
  }

  // notify thread
  // int tid = sockId * comm->nThreads / comm->nSocks;
  int tid = sockId % comm->num_threads;
  struct ncclSocketThreadResources* res = comm->thread_resource + tid;
  if (res->comm == nullptr) {
    res->id = tid;
    res->next = 0;
    res->comm = comm;
    res->state = start;
    waitConnect(comm);
    pthread_mutex_init(&res->thread_lock, nullptr);
    pthread_cond_init(&res->thread_cond, nullptr);
    pthread_create(comm->helper_thread + tid, nullptr, persistentSocketThread,
                   res);
  } else {
    if (kEnableSpin) {
      ++res->next;
    } else {
      pthread_mutex_lock(&res->thread_lock);
      ++res->next;
      pthread_cond_signal(&res->thread_cond);
      pthread_mutex_unlock(&res->thread_lock);
    }
  }
}

static ncclResult_t ncclCommProgress(struct ncclFastSocketComm* comm) {
  int empty_tasks[MAX_SOCKETS];
  int num_empty = 0;

  // no more requests
  if (comm->rq.empty()) return ncclSuccess;

  for (int i = 0; i < comm->num_socks; ++i) {
    int idx = comm->last_fd - i;
    if (idx < 0) idx += comm->num_socks;
    ncclSocketTaskQueue* tasks = &(comm->fd_data[idx].tasks);
    if (tasks->has_inactive()) {
      ncclSocketTask* task = tasks->next_inactive();
      task->r->size_pending -= task->size;
      tasks->dequeue();  // inactive -> free
    }
    if (tasks->has_free()) {
      // socket fd_idx has room for more tasks
      empty_tasks[num_empty++] = idx;
    }
  }

  // no active requests or no socket has room for new tasks
  if (!comm->rq.has_active() || num_empty == 0) return ncclSuccess;

  ncclSocketRequest* ar = comm->rq.next_active();
  if (ar->op == NCCL_SOCKET_SEND) {
    // small enough to send via control socket
    if (ar->size <= kInlineThreshold) {
      ncclCtrl ctrl = {CTRL_INLINE, 0, static_cast<uint32_t>(ar->size), 0,
                       static_cast<uint32_t>(ar->size)};
      NCCLCHECK(ncclCtrlSend(comm, ar, &ctrl));
      NCCLCHECK(ncclCtrlSendSync(comm));
      if (CTRL_DONE(ar)) {
        if (ar->size > 0) {
          int off = 0;
          // send data through control socket
          NCCLCHECK(socketSpin(NCCL_SOCKET_SEND, comm->ctrl_fd, ar->data,
                               ar->size, &off));
          ar->offset = ar->size;
          ar->size_pending = 0;
        }
        comm->rq.mark_inactive();
      }

      return ncclSuccess;
    }
    // there are pending requests and we have available sockets
    while (ar->offset < ar->size && num_empty) {
      --num_empty;
      uint32_t send_size = std::min(kDynamicChunkSize, ar->size - ar->offset);
      ncclCtrl ctrl = {
          CTRL_NORMAL, static_cast<uint16_t>(empty_tasks[num_empty]), send_size,
          static_cast<uint32_t>(ar->offset), static_cast<uint32_t>(ar->size)};
      NCCLCHECK(ncclCtrlSend(comm, ar, &ctrl));
      if (!CTRL_DONE(ar)) {
        break;
      }
      enqueueTask(comm, ar);
      comm->last_fd = empty_tasks[num_empty];
    }
    NCCLCHECK(ncclCtrlSendSync(comm));
  } else {
    do {
      ncclCtrl ctrl;
      if (!CTRL_DONE(ar)) {
        NCCLCHECK(ncclCtrlRecv(comm, ar, &ctrl));
        if (!CTRL_DONE(ar)) break;
        if (ctrl.type == CTRL_INLINE) {
          if (ar->size) {
#ifdef BUFFERED_CTRL
            ar->offset = comm->ctrl_recv.brecv(ar->data, ar->size);
#endif
            NCCLCHECK(socketSpin(NCCL_SOCKET_RECV, comm->ctrl_fd, ar->data,
                                 ar->size, &ar->offset));
            ar->size_pending = 0;
          }
          comm->rq.mark_inactive();
          break;
        }
      }
      if (!comm->fd_data[ar->next_sock_id].tasks.has_free()) {
        break;
        WARN("No free space for recv task");
      }
      // uint32_t recv_size = std::min(dynamic_chunk_size, ar->size -
      // ar->offset);
      enqueueTask(comm, ar);
    } while (ar->offset < ar->size);
  }

  return ncclSuccess;
}

// Called by netSendProxy and netRecvProxy from the proxy thread
ncclResult_t ncclFastSocketTest(void* request, int* done, int* size) {
  *done = 0;
  struct ncclSocketRequest* r = static_cast<struct ncclSocketRequest*>(request);
  if (r == nullptr) {
    WARN("NET/FastSocket : test called with NULL request");
    return ncclInternalError;
  }
  NCCLCHECK(ncclCommProgress(r->comm));
  if (r->comm->rq.has_inactive()) {
    if (r != r->comm->rq.next_inactive()) {
      WARN("NET/FastSocket : test called with invalid request");
      return ncclInternalError;
    }
    if (REQUEST_DONE(r)) {
      r->comm->rq.dequeue();
      *done = 1;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclSocketGetSpeed(char* devName, int* speed) {
  *speed = 0;
  char speedPath[PATH_MAX];
  snprintf(speedPath, PATH_MAX, "/sys/class/net/%s/speed", devName);
  int fd = open(speedPath, O_RDONLY);
  if (fd != -1) {
    char speedStr[] = "        ";
    if (read(fd, speedStr, sizeof(speedStr) - 1) > 0) {
      *speed = strtol(speedStr, nullptr, 0);
    }
    close(fd);
  }
  if (*speed <= 0) {
    INFO(NCCL_NET, "Could not get speed from %s. Defaulting to 10 Gbps.",
         speedPath);
    *speed = 10000;
  }
  return ncclSuccess;
}

template <typename T>
ncclResult_t ncclFastSocketGetProperties(int dev, T* props) {
  props->name = kNcclSocketDevs[dev].dev_name;
  props->pciPath = kNcclSocketDevs[dev].pci_path;
  props->guid = dev;
  props->ptrSupport = NCCL_PTR_HOST;
  NCCLCHECK(ncclSocketGetSpeed(props->name, &props->speed));
  props->port = 0;
  props->maxComms = 65536;
  if constexpr (std::is_same<T, ncclNetProperties_v6_t>::value) {
    props->latency = 0;
    props->maxRecvs = 1;
  }
  return ncclSuccess;
}

ncclResult_t ncclFastSocketRegMr(void* comm, void* data, int size, int type,
                                 void** mhandle) {
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}

ncclResult_t ncclFastSocketDeregMr(void* comm, void* mhandle) {
  return ncclSuccess;
}

ncclResult_t ncclFastSocketFlush_v2(void* recvComm, void* data, int size,
                                    void* mhandle) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}

ncclResult_t ncclFastSocketIsend_v2(void* sendComm, void* data, int size,
                                    void* mhandle, void** request) {
  struct ncclFastSocketComm* comm =
      static_cast<struct ncclFastSocketComm*>(sendComm);
  NCCLCHECK(ncclFastSocketGetRequest(comm, NCCL_SOCKET_SEND, data, size,
                                     (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclFastSocketIrecv_v2(void* recvComm, void* data, int size,
                                    void* mhandle, void** request) {
  struct ncclFastSocketComm* comm =
      static_cast<struct ncclFastSocketComm*>(recvComm);
  NCCLCHECK(ncclFastSocketGetRequest(comm, NCCL_SOCKET_RECV, data, size,
                                     (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclFastSocketIflush_v4(void* recvComm, void* data, int size,
                                     void* mhandle, void** request) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}

ncclResult_t ncclFastSocketIsend_v5(void* sendComm, void* data, int size,
                                    int tag, void* mhandle, void** request) {
  struct ncclFastSocketComm* comm =
      static_cast<struct ncclFastSocketComm*>(sendComm);
  NCCLCHECK(ncclFastSocketGetRequest(comm, NCCL_SOCKET_SEND, data, size,
                                     (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclFastSocketIrecv_v5(void* recvComm, int n, void** data,
                                    int* sizes, int* tags, void** mhandles,
                                    void** request) {
  struct ncclFastSocketComm* comm =
      static_cast<struct ncclFastSocketComm*>(recvComm);
  if (n != 1) return ncclInternalError;
  NCCLCHECK(ncclFastSocketGetRequest(comm, NCCL_SOCKET_RECV, data[0], sizes[0],
                                     (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclFastSocketIflush_v5(void* recvComm, int n, void** data,
                                     int* sizes, void** mhandle,
                                     void** request) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}

ncclResult_t ncclFastSocketCloseListen(void* opaqueComm) {
  struct ncclSocketListenComm* comm = (struct ncclSocketListenComm*)opaqueComm;
  if (comm) {
    if (comm->fd != -1) close(comm->fd);
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclFastSocketPtrSupport(int dev, int* supportedTypes) {
  *supportedTypes = NCCL_PTR_HOST;
  return ncclSuccess;
}

ncclResult_t ncclFastSocketFlush(void* recvComm, void* data, int size,
                                 void* mhandle) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}


volatile ncclNet_v2_t ncclNetPlugin_v2 = {
    "FastSocket",          ncclFastSocketInit,       ncclFastSocketDevices,
    ncclFastSocketPciPath, ncclFastSocketPtrSupport, ncclFastSocketListen,
    ncclFastSocketConnect, ncclFastSocketAccept,     ncclFastSocketRegMr,
    ncclFastSocketDeregMr, ncclFastSocketIsend_v2,   ncclFastSocketIrecv_v2,
    ncclFastSocketFlush,   ncclFastSocketTest,       ncclFastSocketClose,
    ncclFastSocketClose,   ncclFastSocketCloseListen};

volatile ncclNet_v3_t ncclNetPlugin_v3 = {
    "FastSocket",           ncclFastSocketInit,
    ncclFastSocketDevices,  ncclFastSocketGetProperties<ncclNetProperties_v3_t>,
    ncclFastSocketListen,   ncclFastSocketConnect,
    ncclFastSocketAccept,   ncclFastSocketRegMr,
    ncclFastSocketDeregMr,  ncclFastSocketIsend_v2,
    ncclFastSocketIrecv_v2, ncclFastSocketFlush,
    ncclFastSocketTest,     ncclFastSocketClose,
    ncclFastSocketClose,    ncclFastSocketCloseListen};

volatile ncclNet_v4_t ncclNetPlugin_v4 = {
    "FastSocket",           ncclFastSocketInit,
    ncclFastSocketDevices,  ncclFastSocketGetProperties<ncclNetProperties_v4_t>,
    ncclFastSocketListen,   ncclFastSocketConnect,
    ncclFastSocketAccept,   ncclFastSocketRegMr,
    ncclFastSocketDeregMr,  ncclFastSocketIsend_v2,
    ncclFastSocketIrecv_v2, ncclFastSocketIflush_v4,
    ncclFastSocketTest,     ncclFastSocketClose,
    ncclFastSocketClose,    ncclFastSocketCloseListen};

volatile ncclNet_v5_t ncclNetPlugin_v5 = {
    "FastSocket",           ncclFastSocketInit,
    ncclFastSocketDevices,  ncclFastSocketGetProperties<ncclNetProperties_v6_t>,
    ncclFastSocketListen,   ncclFastSocketConnect,
    ncclFastSocketAccept,   ncclFastSocketRegMr,
    ncclFastSocketDeregMr,  ncclFastSocketIsend_v5,
    ncclFastSocketIrecv_v5, ncclFastSocketIflush_v5,
    ncclFastSocketTest,     ncclFastSocketClose,
    ncclFastSocketClose,    ncclFastSocketCloseListen};

volatile ncclNet_v6_t ncclNetPlugin_v6 = {
    "FastSocket",
    ncclFastSocketInit,
    ncclFastSocketDevices,
    ncclFastSocketGetProperties<ncclNetProperties_v6_t>,
    ncclFastSocketListen,
    ncclFastSocketConnect,
    ncclFastSocketAccept,
    ncclFastSocketRegMr,
    nullptr,  // No DMA-BUF support
    ncclFastSocketDeregMr,
    ncclFastSocketIsend_v5,
    ncclFastSocketIrecv_v5,
    ncclFastSocketIflush_v5,
    ncclFastSocketTest,
    ncclFastSocketClose,
    ncclFastSocketClose,
    ncclFastSocketCloseListen};
