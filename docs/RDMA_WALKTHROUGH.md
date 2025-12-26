# UCCL RDMA Collective Implementation Walkthrough

This document provides a code walkthrough of the RDMA-based collective communication implementation in UCCL. It covers the key files, data structures, and code flow for both the data path and control path.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Files](#key-files)
3. [Data Structures](#data-structures)
4. [Control Path: Connection Establishment](#control-path-connection-establishment)
5. [Data Path: Send/Receive Operations](#data-path-sendreceive-operations)
6. [Transport Layer Details](#transport-layer-details)
7. [NCCL Plugin Interface](#nccl-plugin-interface)

---

## Architecture Overview

UCCL's RDMA collective implementation provides a drop-in replacement for NCCL using InfiniBand/RoCE RDMA. The key innovation is **software packet spraying** across multiple queue pairs (QPs) to maximize bandwidth utilization.

```
┌─────────────────────────────────────────────────────────┐
│                    PyTorch/Application                   │
└─────────────────────────────┬───────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────┐
│              NCCL/RCCL (torch.distributed)               │
└─────────────────────────────┬───────────────────────────┘
                              │ NCCL_NET_PLUGIN
┌─────────────────────────────▼───────────────────────────┐
│              UCCL NCCL Plugin (nccl_plugin.cc)           │
│  - pluginInit, pluginListen, pluginConnect, pluginSend  │
└─────────────────────────────┬───────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────┐
│              UCCL Transport Layer (transport.h/cc)       │
│  - UcclRDMAEngine: Multi-path packet spraying           │
│  - RDMAContext: Connection management                   │
│  - Flow control & congestion control                    │
└─────────────────────────────┬───────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────┐
│              RDMA I/O Layer (rdma_io.h/cc)               │
│  - QP creation, memory registration                     │
│  - RDMA SEND/RECV operations                           │
└─────────────────────────────┬───────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────┐
│              libibverbs (RDMA verbs API)                 │
└─────────────────────────────────────────────────────────┘
```

---

## Key Files

### Core Implementation

| File | Purpose | Lines |
|------|---------|-------|
| `transport.h/cc` | Core transport layer with packet spraying, congestion control | ~115K |
| `rdma_io.h/cc` | High-level RDMA helper functions (QP creation, MR registration) | ~30K |
| `nccl_plugin.cc` | NCCL plugin interface (ncclNet API implementation) | ~22K |
| `util_rdma.h` | Low-level RDMA utilities (device enumeration, GID handling) | ~5K |

### Supporting Files

| File | Purpose |
|------|---------|
| `timely.h` | TIMELY congestion control algorithm |
| `swift.h` | SWIFT congestion control algorithm |
| `eqds.h/cc` | Efficient Queue Data Structures for flow management |

---

## Data Structures

### Core Classes

#### `UcclRDMAEngine` (transport.h)
The main transport engine that handles packet spraying across multiple paths.

```cpp
class UcclRDMAEngine {
  // RDMA context for connection management
  std::unique_ptr<RDMAContext> rdma_ctx_;

  // Multiple paths for packet spraying
  std::vector<std::unique_ptr<Path>> paths_;

  // Per-connection flow state
  std::unordered_map<ConnID, std::unique_ptr<Flow>> flows_;

  // TX/RX threads
  std::thread tx_thread_;
  std::thread rx_thread_;

  // Work queues
  jring_t* tx_work_queue_;
  jring_t* rx_completion_queue_;
};
```

#### `RDMAContext` (transport.h)
Manages RDMA resources and connections.

```cpp
class RDMAContext {
  // InfiniBand device and protection domain
  struct ibv_context* ib_ctx_;
  struct ibv_pd* pd_;

  // Completion queues
  struct ibv_cq* send_cq_;
  struct ibv_cq* recv_cq_;

  // Memory regions for registered buffers
  std::unordered_map<void*, struct ibv_mr*> mr_cache_;

  // Queue pair pool for packet spraying
  std::vector<QPWrapper> qp_pool_;
};
```

#### `Flow` (transport.h)
Per-connection flow state for reliable transport.

```cpp
struct Flow {
  ConnID conn_id;

  // Sequence numbers for reliability
  uint64_t next_seq_to_send_;
  uint64_t next_seq_expected_;

  // Congestion control state
  double cwnd_;  // Congestion window
  double rtt_;   // Round-trip time estimate

  // Retransmission queue
  std::deque<Packet> unacked_packets_;
};
```

---

## Control Path: Connection Establishment

### 1. Plugin Initialization

When NCCL loads the UCCL plugin, it calls `pluginInit()`:

```cpp
// nccl_plugin.cc
ncclResult_t pluginInit(ncclDebugLogger_t logFunction) {
  // Initialize glog
  google::InitGoogleLogging("UCCL");

  // Create RDMA engines (one per GPU)
  for (int gpu = 0; gpu < num_gpus; gpu++) {
    engines_[gpu] = std::make_unique<UcclRDMAEngine>(gpu);
  }

  return ncclSuccess;
}
```

### 2. Listen for Connections

Server side creates a listening endpoint:

```cpp
// nccl_plugin.cc
ncclResult_t pluginListen(int dev, void* handle, void** listenComm) {
  // Get engine for this device
  auto* engine = engines_[dev].get();

  // Start listening on TCP socket for connection handshake
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  bind(listen_fd, ...);
  listen(listen_fd, SOMAXCONN);

  // Store handle for later accept
  *listenComm = new ListenComm{engine, listen_fd};
  return ncclSuccess;
}
```

### 3. Connect to Remote

Client initiates connection:

```cpp
// nccl_plugin.cc
ncclResult_t pluginConnect(int dev, void* handle, void** sendComm) {
  auto* engine = engines_[dev].get();

  // TCP handshake to exchange RDMA connection info
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  connect(sock, ...);

  // Exchange QP numbers, LIDs, GIDs
  RDMAConnInfo local_info = engine->get_local_conn_info();
  send(sock, &local_info, sizeof(local_info), 0);
  recv(sock, &remote_info, sizeof(remote_info), 0);

  // Create RDMA connection
  ConnID conn_id = engine->connect(remote_info);

  *sendComm = new SendComm{engine, conn_id};
  return ncclSuccess;
}
```

### 4. Accept Connection

Server accepts and establishes RDMA connection:

```cpp
// nccl_plugin.cc
ncclResult_t pluginAccept(void* listenComm, void** recvComm) {
  auto* lc = static_cast<ListenComm*>(listenComm);

  // Accept TCP connection
  int client_fd = accept(lc->listen_fd, ...);

  // Exchange RDMA info
  recv(client_fd, &remote_info, sizeof(remote_info), 0);
  RDMAConnInfo local_info = lc->engine->get_local_conn_info();
  send(client_fd, &local_info, sizeof(local_info), 0);

  // Create RDMA connection
  ConnID conn_id = lc->engine->accept(remote_info);

  *recvComm = new RecvComm{lc->engine, conn_id};
  return ncclSuccess;
}
```

---

## Data Path: Send/Receive Operations

### Send Operation

```cpp
// nccl_plugin.cc
ncclResult_t pluginSend(void* sendComm, void* data, int size,
                        int tag, void** request) {
  auto* sc = static_cast<SendComm*>(sendComm);

  // Create send request
  auto* req = new Request{data, size, tag};

  // Submit to engine's TX queue
  sc->engine->submit_send(sc->conn_id, req);

  *request = req;
  return ncclSuccess;
}
```

### Engine TX Processing

The engine processes sends with packet spraying:

```cpp
// transport.cc
void UcclRDMAEngine::handle_tx_work() {
  while (running_) {
    Request* req = tx_work_queue_->dequeue();
    if (!req) continue;

    Flow* flow = get_flow(req->conn_id);

    // Fragment into chunks for packet spraying
    size_t offset = 0;
    while (offset < req->size) {
      size_t chunk_size = std::min(chunk_size_, req->size - offset);

      // Select path using round-robin for spraying
      int path_idx = flow->next_path_++ % num_paths_;
      Path* path = paths_[path_idx].get();

      // Create packet with sequence number
      Packet pkt{
        .seq = flow->next_seq_to_send_++,
        .data = req->data + offset,
        .size = chunk_size
      };

      // Post RDMA SEND
      path->post_send(pkt);

      // Track for retransmission
      flow->unacked_packets_.push_back(pkt);

      offset += chunk_size;
    }
  }
}
```

### Receive Operation

```cpp
// nccl_plugin.cc
ncclResult_t pluginRecv(void* recvComm, int n, void** data,
                        int* sizes, int* tags, void** request) {
  auto* rc = static_cast<RecvComm*>(recvComm);

  // Create receive request
  auto* req = new Request{data, sizes, tags, n};

  // Submit to engine's RX queue
  rc->engine->submit_recv(rc->conn_id, req);

  *request = req;
  return ncclSuccess;
}
```

### Engine RX Processing

```cpp
// transport.cc
void UcclRDMAEngine::handle_rx_completion() {
  while (running_) {
    // Poll completion queue
    struct ibv_wc wc[BATCH_SIZE];
    int num = ibv_poll_cq(recv_cq_, BATCH_SIZE, wc);

    for (int i = 0; i < num; i++) {
      Packet* pkt = reinterpret_cast<Packet*>(wc[i].wr_id);
      Flow* flow = get_flow(pkt->conn_id);

      // Handle out-of-order packets
      if (pkt->seq == flow->next_seq_expected_) {
        // In-order: deliver immediately
        deliver_to_app(pkt);
        flow->next_seq_expected_++;

        // Check reorder buffer for more in-order packets
        while (flow->has_buffered(flow->next_seq_expected_)) {
          deliver_to_app(flow->get_buffered(flow->next_seq_expected_));
          flow->next_seq_expected_++;
        }
      } else if (pkt->seq > flow->next_seq_expected_) {
        // Out-of-order: buffer for later
        flow->buffer_packet(pkt);
      }
      // else: duplicate, ignore

      // Send ACK
      send_ack(flow, pkt->seq);
    }
  }
}
```

---

## Transport Layer Details

### Packet Spraying

UCCL uses software packet spraying to leverage multiple network paths:

```cpp
// transport.cc
// Configuration via environment variable
int num_paths = env_get_int("UCCL_PORT_ENTROPY", 32);

// Create multiple QPs per connection
for (int i = 0; i < num_paths; i++) {
  paths_.push_back(create_path(remote_info, i));
}

// Round-robin path selection during send
int select_path(Flow* flow) {
  return flow->next_path_++ % paths_.size();
}
```

### Congestion Control

UCCL supports multiple congestion control algorithms:

```cpp
// timely.h - TIMELY algorithm (latency-based)
class TimelyCongestionControl {
  void on_ack(uint64_t rtt_us) {
    if (rtt_us < rtt_low_) {
      // RTT below threshold: increase rate
      cwnd_ += additive_increase_;
    } else if (rtt_us > rtt_high_) {
      // RTT above threshold: decrease rate
      cwnd_ *= (1.0 - beta_);
    }
    // Gradient-based adjustment in between
  }
};

// swift.h - SWIFT algorithm (receiver-driven)
class SwiftCongestionControl {
  void on_ack(AckInfo ack) {
    // Adjust based on receiver feedback
    target_rate_ = ack.suggested_rate;
    cwnd_ = target_rate_ * rtt_;
  }
};
```

### Loss Recovery

Selective repeat for efficient loss recovery:

```cpp
// transport.cc
void UcclRDMAEngine::handle_ack(Flow* flow, uint64_t acked_seq) {
  // Remove from unacked queue
  flow->unacked_packets_.remove_if(
    [acked_seq](const Packet& p) { return p.seq <= acked_seq; });

  // Update RTT estimate
  update_rtt(flow, acked_seq);
}

void UcclRDMAEngine::handle_timeout(Flow* flow) {
  // Retransmit only unacked packets (selective repeat)
  for (const auto& pkt : flow->unacked_packets_) {
    if (now() - pkt.send_time > rto_) {
      retransmit(pkt);
    }
  }
}
```

---

## NCCL Plugin Interface

The plugin implements the `ncclNet_v8_t` interface:

```cpp
// nccl_plugin.cc
ncclNet_v8_t ncclNetPlugin_v8 = {
  .name = "UCCL",
  .init = pluginInit,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties,
  .listen = pluginListen,
  .connect = pluginConnect,
  .accept = pluginAccept,
  .regMr = pluginRegMr,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend,
  .irecv = pluginIrecv,
  .test = pluginTest,
  .closeListen = pluginCloseListen,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
};
```

### Memory Registration

```cpp
// nccl_plugin.cc
ncclResult_t pluginRegMr(void* comm, void* data, size_t size,
                         int type, void** mhandle) {
  auto* sc = static_cast<SendComm*>(comm);

  // Register memory with RDMA device
  struct ibv_mr* mr = ibv_reg_mr(
    sc->engine->get_pd(),
    data, size,
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
  );

  *mhandle = mr;
  return ncclSuccess;
}
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `UCCL_NUM_ENGINES` | Number of RDMA engines per GPU | 4 |
| `UCCL_PORT_ENTROPY` | Number of QPs per connection (paths) | 32 |
| `UCCL_CHUNK_SIZE_KB` | Maximum chunk size per packet | 64 |
| `UCCL_IB_HCA` | IB device names to use | auto |
| `UCCL_IB_GID_INDEX` | GID index for RoCE | -1 |
| `UCCL_RCMODE` | Use RC mode (1) or UC mode (0) | 0 |

---

## Further Reading

- [RDMA I/O Documentation](./RDMA_IO.md) - Detailed rdma_io.h/cc walkthrough
- [Congestion Control](./CONGESTION_CONTROL.md) - TIMELY and SWIFT algorithms
- [NCCL Plugin API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) - NVIDIA NCCL documentation

---

*This documentation is part of the UCCL project. For more information, visit [https://github.com/uccl-project/uccl](https://github.com/uccl-project/uccl)*
