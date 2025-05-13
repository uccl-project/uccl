#pragma once
#include <cstdint>
#include <thread>

#include "util.h"

#define PATH_SELECTION
#define REXMIT_SET_PATH
// #define USE_SRD
// #define USE_SRD_FOR_CTRL
// #define EMULATE_RC_ZC
#define SCATTERED_MEMCPY
// #define RTT_STATS
// #define POLLCTX_DEBUG

enum class SenderCCType {
    kNone,
    kTimely,
    kTimelyPP,
    kCubic,
    kCubicPP,
};
enum class ReceiverCCType {
    kNone,
};
static constexpr SenderCCType kSenderCCType = SenderCCType::kCubic;
static constexpr ReceiverCCType kReceiverCCType = ReceiverCCType::kNone;
static_assert(
    kSenderCCType != SenderCCType::kNone ||
        kReceiverCCType != ReceiverCCType::kNone,
    "kSenderCCType and kReceiverCCType can not be kNone at the same time.");

// #define P4D
#define P5EN

static const uint32_t kBundleNIC = 2;
static const uint32_t kNumVdevices = 8;        // # of vEFA/GPUs.
static const uint32_t kNumEnginesPerVdev = kBundleNIC * 2;  // # of engines per vEFA/GPU.
static const uint32_t kNumEngines = kNumVdevices * kNumEnginesPerVdev;
static const bool kSplitSendRecvEngine =
    false;  // Split sender/recevier flows to dedicated engines.

/// Interface configuration.
#ifdef P4D
static const uint8_t NUM_DEVICES = (kNumVdevices + 1) / 2;
static const uint8_t EFA_GID_IDX = 0;
static const std::string EFA_DEVICE_NAME_LIST[] = {
    "rdmap16s27", "rdmap32s27", "rdmap144s27", "rdmap160s27"};
static const std::string ENA_DEVICE_NAME_LIST[] = {"ens32", "ens65", "ens130",
                                                   "ens163"};
static constexpr double kLinkBandwidth = 100.0 * 1e9 / 8;  // 100Gbps
#endif

#ifdef P5EN
static const uint8_t NUM_DEVICES = kNumVdevices * 2;
static const uint8_t EFA_GID_IDX = 0;
static const std::string EFA_DEVICE_NAME_LIST[] = {
    "rdmap85s0",
    "rdmap86s0",
    "rdmap87s0",
    "rdmap88s0",
    "rdmap110s0",
    "rdmap111s0",
    "rdmap112s0",
    "rdmap113s0",
    "rdmap135s0",
    "rdmap136s0",
    "rdmap137s0",
    "rdmap138s0",
    "rdmap160s0",
    "rdmap161s0",
    "rdmap162s0",
    "rdmap163s0",
    };
static const std::string ENA_DEVICE_NAME_LIST[] = {
    "enp71s0",
    "enp72s0",
    "enp73s0",
    "enp74s0",
    "enp96s0",
    "enp97s0",
    "enp98s0",
    "enp99s0",
    "enp122s0",
    "enp123s0",
    "enp124s0",
    "enp125s0",
    "enp146s0",
    "enp147s0",
    "enp148s0",
    "enp149s0",
    };
static constexpr double kLinkBandwidth = 400.0 * 1e9 / 8;  // 400Gbps
#endif


static const uint8_t EFA_PORT_NUM = 1;  // The port of EFA device to use.
static const uint32_t EFA_MTU = 9000;  // Max frame on fabric, includng headers.
static const uint32_t EFA_MAX_PAYLOAD = 8928;  // this excludes EFA_UD_ADDITION.
static const uint32_t EFA_HDR_OVERHEAD = EFA_MTU - EFA_MAX_PAYLOAD;
static const uint32_t EFA_MAX_QPS = 256;         // Max QPs per EFA device.
static const uint32_t EFA_MAX_INLINE_SIZE = 32;  // Max inline data size.
#ifdef USE_SRD
static const uint32_t EFA_UD_ADDITION = 0;  // Auto-added by EFA during recv.
#else
static const uint32_t EFA_UD_ADDITION = 40;  // Auto-added by EFA during recv.
#endif
/// Interface configuration.
static_assert(kNumEngines >= NUM_DEVICES * 2,
              "kNumEngines must be at least twice of NUM_DEVICES, one for send "
              "and one for receive to avoid deadlocks.");

static uint32_t NUM_CPUS = std::thread::hardware_concurrency();
// Using the middle 96 cores to avoid conflicting with nccl proxy service (that
// uses the first 48 and last 48 cores as specified in p5en-48xl-topo.xml). The
// two numbers are for numa 0 and 1 separately. GPU 0-3 + NIC 0-7 are on numa 0,
// and GPU 4-7 + NIC 8-15 are on numa 1.

// NUMA node0 CPU(s):      0-47,96-143
// NUMA node1 CPU(s):      48-95,144-191
static const uint32_t ENGINE_CPU_START[2] = {NUM_CPUS / 2, NUM_CPUS / 4};
static const uint16_t BASE_PORT = 10000;
static const uint64_t NUM_FRAMES = 65536 * 4;  // # of frames.
static const uint32_t RECV_BATCH_SIZE = 32;
static const uint32_t SEND_BATCH_SIZE = 16;
static const uint32_t QKEY = 0x12345;
static const uint32_t SQ_PSN = 0x12345;
static const uint64_t MAX_FLOW_ID = 1000000;

// libibverbs configuration.
static const uint32_t kMaxSendWr = 1024;
static const uint32_t kMaxRecvWr = 256;
static const uint32_t kMaxSendRecvWrForCtrl = 1024;
static const uint32_t kMaxCqeTotal = 16384;
static const uint32_t kMaxPollBatch = 32;
static const uint32_t kMaxRecvWrDeficit = 32;
static const uint32_t kMaxChainedWr = 32;
static const uint32_t kMaxUnconsumedRxMsgbufs = NUM_FRAMES / 16;
static const uint32_t kMaxMultiRecv = 8;

// Path configuration.
// Setting to 20 gives highest bimq perf (191 vs. 186G), but bad for NCCL.
static const uint32_t kMaxDstQP = 16;  // # of paths/QPs for data per src qp.
static const uint32_t kMaxSrcQP = 16;
static const uint32_t kMaxDstQPCtrl = 16;  // # of paths/QPs for control.
static const uint32_t kMaxSrcQPCtrl = 16;
static constexpr uint32_t kMaxSrcDstQP = std::max(kMaxSrcQP, kMaxDstQP);
static constexpr uint32_t kMaxSrcDstQPCtrl =
    std::max(kMaxSrcQPCtrl, kMaxDstQPCtrl);
static const uint32_t kMaxPath = kMaxDstQP * kMaxSrcQP;
static const uint32_t kMaxPathCtrl = kMaxDstQPCtrl * kMaxSrcQPCtrl;
// This check is not enough as the kMaxSendWr/kMaxRecvWr also affects the number of QPs.
// static_assert((kMaxDstQP + kMaxDstQPCtrl) * kNumEnginesPerVdev <= EFA_MAX_QPS);
// static_assert((kMaxSrcQP + kMaxSrcQPCtrl) * kNumEnginesPerVdev <= EFA_MAX_QPS);

// CC parameters.
static const double kMaxUnackedPktsPP = 1.25;
static const uint32_t kMaxUnackedPktsPerEngine = kMaxUnackedPktsPP * kMaxPath;
static const std::size_t kSackBitmapSize = 1024;
static const std::size_t kFastRexmitDupAckThres = 128;
static const double kMaxBwPP = 5.0 * 1e9 / 8;
static const uint32_t kSwitchPathThres = 1u;
static const uint32_t kMaxPktsInTimingWheel = 1024;
static const uint32_t kMaxPathHistoryPerEngine = 4096;

static_assert(kMaxPath <= 4096, "kMaxPath too large");
static_assert(kMaxUnackedPktsPerEngine <= kMaxPathHistoryPerEngine,
              "kMaxUnackedPktsPerEngine too large");
static_assert(is_power_of_two(kMaxPathHistoryPerEngine),
              "kMaxPathHistoryPerEngine must be power of 2");
