# UCCL Transport Modes: RC, UC, and UD

This document explains the different RDMA transport modes supported by UCCL and how they handle out-of-order packets, reliability, and congestion control.

## Overview

UCCL supports three RDMA transport modes, each with different trade-offs:

| Mode | Full Name | Reliability | Ordering | Use Case |
|------|-----------|-------------|----------|----------|
| **RC** | Reliable Connection | Hardware | Hardware | Default, guaranteed delivery |
| **UC** | Unreliable Connection | Software | Software | Higher throughput potential |
| **UD** | Unreliable Datagram | Software | Software | Multicast, connectionless |

## Configuration

Set the transport mode via environment variable:

```bash
# Use Reliable Connection (default)
export UCCL_RCMODE=1

# Use Unreliable Connection
export UCCL_RCMODE=0
```

---

## Reliable Connection (RC) Mode

### How It Works

RC mode uses the InfiniBand hardware to guarantee reliable, in-order delivery:

```
Sender                              Receiver
  │                                    │
  ├──── Packet 1 (seq=1) ────────────►│
  │                                    │ Hardware ACK
  │◄─────────── ACK 1 ─────────────────┤
  │                                    │
  ├──── Packet 2 (seq=2) ────────────►│
  │         (lost)                     │
  │                                    │ Timeout → Hardware retransmit
  ├──── Packet 2 (seq=2) ────────────►│
  │◄─────────── ACK 2 ─────────────────┤
```

### Characteristics

- **Reliability**: Handled by ConnectX NIC hardware
- **Ordering**: Packets delivered in-order by hardware
- **Retransmission**: Automatic by NIC (go-back-N style)
- **Congestion Control**: Optional (UCCL can add software CC)

### Advantages

1. Simpler software implementation
2. Lower CPU overhead for reliability
3. Guaranteed delivery without software intervention

### Disadvantages

1. Limited scalability (one QP per connection)
2. Head-of-line blocking on packet loss
3. Go-back-N retransmission wastes bandwidth

### Code Path

```cpp
// transport.cc - RC mode initialization
void RDMAContext::create_rc_qp() {
  struct ibv_qp_init_attr qp_attr = {
    .qp_type = IBV_QPT_RC,  // Reliable Connection
    .send_cq = send_cq_,
    .recv_cq = recv_cq_,
    .cap = {
      .max_send_wr = 128,
      .max_recv_wr = 128,
      .max_send_sge = 1,
      .max_recv_sge = 1,
    }
  };
  qp_ = ibv_create_qp(pd_, &qp_attr);
}
```

---

## Unreliable Connection (UC) Mode

### How It Works

UC mode relies on software for reliability while using hardware for basic packet delivery:

```
Sender                              Receiver
  │                                    │
  ├──── Packet 1 (seq=1) ────────────►│
  ├──── Packet 2 (seq=2) ────────────►│
  ├──── Packet 3 (seq=3) ───X         │ (lost)
  ├──── Packet 4 (seq=4) ────────────►│
  │                                    │
  │                                    │ Software detects gap
  │◄───── NACK 3 ──────────────────────┤
  │                                    │
  ├──── Packet 3 (seq=3) ────────────►│ (selective retransmit)
  │◄───── ACK 4 ───────────────────────┤
```

### Characteristics

- **Reliability**: Software-based (UCCL transport layer)
- **Ordering**: Software reordering buffer
- **Retransmission**: Selective repeat (only lost packets)
- **Congestion Control**: TIMELY or SWIFT algorithms

### Out-of-Order Packet Handling

UCCL handles out-of-order packets in software:

```cpp
// transport.cc - UC mode packet handling
void UcclRDMAEngine::handle_uc_packet(Packet* pkt) {
  Flow* flow = get_flow(pkt->conn_id);

  if (pkt->seq == flow->next_seq_expected_) {
    // In-order packet
    deliver_to_app(pkt);
    flow->next_seq_expected_++;

    // Drain reorder buffer
    while (auto* buffered = flow->pop_if_next()) {
      deliver_to_app(buffered);
      flow->next_seq_expected_++;
    }
  } else if (pkt->seq > flow->next_seq_expected_) {
    // Out-of-order: buffer
    flow->reorder_buffer_.insert(pkt->seq, pkt);

    // Send duplicate ACK to trigger fast retransmit
    if (++flow->dup_ack_count_ >= 3) {
      send_nack(flow, flow->next_seq_expected_);
    }
  }
  // Duplicate packets (seq < expected) are dropped
}
```

### Advantages

1. **Selective Repeat**: Only retransmit lost packets
2. **No Head-of-Line Blocking**: Continue receiving while waiting
3. **Software Congestion Control**: Fine-grained rate control
4. **Packet Spraying**: Works well with multi-path

### Disadvantages

1. Higher CPU overhead for software reliability
2. More complex implementation
3. Reorder buffer memory usage

### Code Path

```cpp
// transport.cc - UC mode initialization
void RDMAContext::create_uc_qp() {
  struct ibv_qp_init_attr qp_attr = {
    .qp_type = IBV_QPT_UC,  // Unreliable Connection
    .send_cq = send_cq_,
    .recv_cq = recv_cq_,
    // ... same caps as RC
  };
  qp_ = ibv_create_qp(pd_, &qp_attr);

  // Software reliability state
  flow->reorder_buffer_ = std::map<uint64_t, Packet*>();
  flow->next_seq_expected_ = 0;
}
```

---

## Unreliable Datagram (UD) Mode

### How It Works

UD mode is connectionless - each packet can go to any destination:

```
Sender                    Network                Receivers
  │                          │                      │
  ├──── Packet (dest=A) ────►├─────────────────────►│ A
  ├──── Packet (dest=B) ────►├───────────►│ B       │
  ├──── Packet (dest=C) ────►├────────────┼────────►│ C
```

### Characteristics

- **Connection**: None (address in each packet)
- **Reliability**: Software-based
- **Use Case**: Multicast, discovery, control plane

### Advantages

1. **Scalability**: One QP serves all destinations
2. **Multicast**: Native multicast support
3. **Flexibility**: No connection setup needed

### Disadvantages

1. **MTU Limit**: Packets limited to ~4KB
2. **No RDMA Operations**: Only SEND/RECV, no READ/WRITE
3. **Address Header Overhead**: 40 bytes per packet

### Code Path

```cpp
// transport.cc - UD mode (used for discovery/control)
void RDMAContext::create_ud_qp() {
  struct ibv_qp_init_attr qp_attr = {
    .qp_type = IBV_QPT_UD,  // Unreliable Datagram
    .send_cq = send_cq_,
    .recv_cq = recv_cq_,
    // ...
  };
  qp_ = ibv_create_qp(pd_, &qp_attr);
}

// UD send requires address handle
void send_ud_packet(void* data, size_t size, struct ibv_ah* ah, uint32_t qpn) {
  struct ibv_send_wr wr = {
    .wr_id = (uint64_t)data,
    .sg_list = &sge,
    .num_sge = 1,
    .opcode = IBV_WR_SEND,
    .wr = {
      .ud = {
        .ah = ah,
        .remote_qpn = qpn,
        .remote_qkey = QKEY,
      }
    }
  };
  ibv_post_send(ud_qp_, &wr, &bad_wr);
}
```

---

## Comparison: When to Use Each Mode

### Use RC Mode When:

- You need simplest implementation
- CPU resources are limited
- Network is reliable (low loss rate)
- Latency is less critical than throughput

### Use UC Mode When:

- You want maximum throughput
- You need fine-grained congestion control
- Network has multiple paths (packet spraying)
- You can afford CPU overhead for software reliability

### Use UD Mode When:

- You need multicast
- You have many peers (discovery, control plane)
- Packets are small (< 4KB)
- Connection setup overhead is too high

---

## UCCL's Default: UC with Software Reliability

UCCL defaults to UC mode with software reliability for collective operations because:

1. **Packet Spraying**: UC allows sending packets across multiple QPs/paths
2. **Selective Repeat**: More efficient than RC's go-back-N
3. **Congestion Control**: TIMELY/SWIFT provide better throughput
4. **Multi-Path**: Leverages datacenter network topology

```cpp
// Default configuration
// UCCL_RCMODE=0 (UC mode with software reliability)
// UCCL_PORT_ENTROPY=32 (32 paths for spraying)
// UCCL_CHUNK_SIZE_KB=64 (64KB chunks)
```

---

## Packet Loss Detection

UCCL uses multiple mechanisms to detect packet loss:

### 1. Duplicate ACKs (Fast Retransmit)

```cpp
if (flow->dup_ack_count_ >= 3) {
  // Fast retransmit without waiting for timeout
  retransmit(flow, flow->next_seq_expected_);
}
```

### 2. Timeout-Based Detection

```cpp
void check_timeouts() {
  for (auto& [seq, pkt] : flow->unacked_) {
    if (now() - pkt.send_time > rto_) {
      retransmit(pkt);
    }
  }
}
```

### 3. RACK-TLP (Modern Approach)

UCCL is exploring RACK-TLP for more efficient loss detection:

```cpp
// Recent ACK-based detection (RACK)
void on_ack(uint64_t acked_seq, uint64_t ack_time) {
  // Packets sent before recently-ACKed packet
  // but not yet ACKed are likely lost
  for (auto& pkt : unacked_) {
    if (pkt.send_time < recent_ack_time - reordering_window_) {
      mark_lost(pkt);
    }
  }
}
```

---

## Further Reading

- [UCCL RDMA Walkthrough](./RDMA_WALKTHROUGH.md)
- [InfiniBand Architecture Specification](https://www.infinibandta.org/)
- [RACK-TLP RFC 8985](https://www.rfc-editor.org/rfc/rfc8985)
- [TIMELY Paper](https://dl.acm.org/doi/10.1145/2829988.2787510)

---

*This documentation is part of the UCCL project. For more information, visit [https://github.com/uccl-project/uccl](https://github.com/uccl-project/uccl)*
