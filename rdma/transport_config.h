#pragma once

#include <string>
#include <thread>
#include <unistd.h>
#include <cstdlib>


// #define STATS
// #define LAZY_CREATE_ENGINE

#define MAX_IB_DEVS 32

inline uint32_t getEnvUint32(const char* name, uint32_t default_value) {
    const char* val = std::getenv(name);
    return val ? static_cast<uint32_t>(std::stoul(val)) : default_value;
}   


// Configuration parameters
static const bool ROCE_NET = true;

static const double LINK_BANDWIDTH = 100.0 * 1e9 / 8;
static const uint32_t NUM_ENGINES = getEnvUint32("NUM_ENGINES", 1);
static const uint32_t kPortEntropy = getEnvUint32("PORT_ENTROPY", 256);
static const uint32_t kChunkSize = getEnvUint32("CHUNK_SIZE", 128 << 10);


static const uint32_t MAX_PEER = 256;
// Maximum number of flows (one-way) on each engine.
static const uint32_t MAX_FLOW = 256;

// Traffic class
static const uint8_t kTrafficClass = ROCE_NET ? 3 : 0;
// Service level
static const uint8_t kServiceLevel = ROCE_NET ? 135 : 0;
// GID Index
static const uint8_t GID_IDX = ROCE_NET ? 3 : 0;
/// Interface configuration.

static uint32_t NUM_CPUS = std::thread::hardware_concurrency();
// Each dev use [ENGINE_CPU_START_LIST[dev], ENGINE_CPU_START_LIST[dev] +
// NUM_ENGINES)
static uint32_t ENGINE_CPU_START_LIST[8] = {
    16, 16 + NUM_ENGINES, 16 + 2 * NUM_ENGINES, 16 + 3 * NUM_ENGINES,
    96, 96 + NUM_ENGINES, 96 + 2 * NUM_ENGINES, 96 + 3 * NUM_ENGINES,
};

// Use RC rather than UC.
static const bool kRCMode = false;
// Bypass the pacing stage.
static const bool kBypassPacing = true;

// Limit the per-flow outstanding bytes on each engine.
static const uint32_t kMaxUnAckedBytesPerFlow =
    2 * std::max(kChunkSize, 32768u);
// Limit the outstanding bytes on each engine.
// Low means if a flow exceeds its own budget but doesn't exceed the Low
// threshold, it can send until Low threshold. static constexpr uint32_t
// kMaxUnAckedBytesPerEngineLow = 18 * std::max(kChunkSize, 32768u);
static const uint32_t kMaxUnAckedBytesPerEngineLow =
    8 * std::max(kChunkSize, 32768u);
// High means if all flows reach this threshold, all flows can't send any bytes.
// static constexpr uint32_t kMaxUnAckedBytesPerEngineHigh = 24 *
// std::max(kChunkSize, 32768u);
static const uint32_t kMaxUnAckedBytesPerEngineHigh =
    12 * std::max(kChunkSize, 32768u);

// Congestion control algorithm.
enum SenderCCA {
  SENDER_CCA_NONE,
  // Timely [SIGCOMM'15]
  SENDER_CCA_TIMELY,
  // Swift [SIGCOMM'20]
  SENDER_CCA_SWIFT,
};
enum ReceiverCCA {
  RECEIVER_CCA_NONE,
  // EQDS [NSDI'22]
  RECEIVER_CCA_EQDS,
};
static const enum SenderCCA kSenderCCA = SENDER_CCA_TIMELY;
static const enum ReceiverCCA kReceiverCCA = RECEIVER_CCA_NONE;
static_assert(kSenderCCA != SENDER_CCA_NONE ||
                  kReceiverCCA != RECEIVER_CCA_NONE,
              "At least one of the sender and receiver must have a congestion "
              "control algorithm.");

// Note that load-based policy shoud >= ENGINE_POLICY_LOAD.
enum engine_lb_policy {
  // Bind each flow to one engine.
  ENGINE_POLICY_BIND,
  // Round-robin among engines.
  ENGINE_POLICY_RR,
  // Choose obliviously.
  ENGINE_POLICY_OBLIVIOUS,
  // Load balancing based on the load of each engine.
  ENGINE_POLICY_LOAD,
  // Variant of ENGINE_POLICY_LOAD, which uses power of two.
  ENGINE_POLICY_LOAD_POT,
};
static const enum engine_lb_policy kEngineLBPolicy = ENGINE_POLICY_RR;

static uint32_t const PACER_CPU_START = 3 * NUM_CPUS / 4;

// Replace constexpr with const for kTotalQP
static const int kTotalQP =
    kPortEntropy + 1 /* Credit QP */ + (kRCMode ? 0 : 1) /* Ctrl QP */;
// Recv buffer size smaller than kRCSize will be handled by RC directly.
static const uint32_t kRCSize = 8192;
// fallback to nccl
// static constexpr uint32_t kRCSize = 4000000;
// Minimum post receive size in NCCL.
static const uint32_t NCCL_MIN_POST_RECV = 65536;
// fallback to nccl
// static constexpr uint32_t NCCL_MIN_POST_RECV = 4000000;

// Limit the bytes of consecutive cached QP uses.
static const uint32_t kMAXConsecutiveSameChoiceBytes = 16384;
// Message size threshold for allowing using cached QP.
static const uint32_t kMAXUseCacheQPSize = 8192;
// Message size threshold for bypassing the timing wheel.
static const uint32_t kBypassTimingWheelThres = 9000;

// # of Tx work handled in one loop.
static const uint32_t kMaxTxWork = 4;
// Maximum number of Tx bytes to be transmitted in one loop.
static const uint32_t kMaxTxBytesThres = 32 * std::max(kChunkSize, 32768u);
// # of Rx work handled in one loop.
static const uint32_t kMaxRxWork = 8;
// Completion queue (CQ) size.
static const int kCQSize = 16384;
// Interval for posting a signal WQE.
// static constexpr uint32_t kSignalInterval = kCQSize >> 1;
static const uint32_t kSignalInterval = 1;
// Interval for syncing the clock with NIC.
static const uint32_t kSyncClockIntervalNS = 100000;
// Maximum number of CQEs to retrieve in one loop.
static const uint32_t kMaxBatchCQ = 16;
// CQ moderation count.
static const uint32_t kCQMODCount = 32;
// CQ moderation period in microsecond.
static const uint32_t kCQMODPeriod = 100;
// Maximum size of inline data.
static const uint32_t kMaxInline = 512;
// Maximum number of SGEs in one WQE.
static const uint32_t kMaxSge = 2;
// Maximum number of outstanding receive messages in one recv request.
static const uint32_t kMaxRecv = 1;
// Maximum number of outstanding receive requests in one engine.
static const uint32_t kMaxReq = 128;
// Maximum number of WQEs in SRQ (Shared Receive Queue).
static const uint32_t kMaxSRQ = 16 * kMaxReq;
// Maximum number of chunks can be transmitted from timing wheel in one loop.
static const uint32_t kMaxBurstTW = 8;
// Posting recv WQEs every kPostRQThreshold.
static const uint32_t kPostRQThreshold = kMaxBatchCQ;
// When CQEs from one QP reach kMAXCumWQE, send immediate ack.
// 1 means always send immediate ack.
static const uint32_t kMAXCumWQE = 4;
// When the cumulative bytes reach kMAXCumBytes, send immediate ack.
static const uint32_t kMAXCumBytes = kMAXCumWQE * kChunkSize;
// Before reaching it, the receiver will not consider that it has encountered
// OOO, and thus there is no immediate ack. This is to tolerate the OOO caused
// by the sender's qp scheduling.
static const uint32_t kMAXRXOOO = 8;

// Sack bitmap size in bits.
// Note that kSackBitmapSize must be <= half the maximum value of UINT_CSN.
// E.g., UINT_CSN = 8bit, kSacBitmapSize <= 128.
static const std::size_t kSackBitmapSize = 64 << 1;
// kFastRexmitDupAckThres equals to 1 means all duplicate acks are caused by
// packet loss. This is true for flow-level ECMP, which is the common case. When
// the network supports adaptive routing, duplicate acks may be caused by
// adaptive routing. In this case, kFastRexmitDupAckThres should be set to a
// value greater than 0.
static const std::size_t kFastRexmitDupAckThres = ROCE_NET ? 32 : 65536;

// Maximum number of Retransmission Timeout (RTO) before aborting the flow.
static const uint32_t kRTOAbortThreshold = 50;

static const uint32_t kMAXRTTUS = 10000;
// Constant/Dynamic RTO.
static const bool kConstRTO = true;
// kConstRTO == true: Constant retransmission timeout in microseconds.
static const double kRTOUSec = 1000;
// kConstRTO == false: Minimum retransmission timeout in microseconds.
static const double kMinRTOUsec = 1000;
static const uint32_t kRTORTT = 4;  // RTO = kRTORTT RTTs

// Slow timer (periodic processing) interval in microseconds.
static const size_t kSlowTimerIntervalUs = 1000;

/// Debugging and testing.
// Disable hardware timestamp.
static const bool kTestNoHWTimestamp = false;
// Use constant(maximum) rate for transmission.
static const bool kTestConstantRate = false;
// Test lossy network.
static const bool kTestLoss = false;
static const double kTestLossRate = 0.0;
// Disable RTO.
static const bool kTestNoRTO =
    (ROCE_NET || kTestLoss)
        ? false
        : true;  // Infiniband is lossless, disable RTO even for UC.