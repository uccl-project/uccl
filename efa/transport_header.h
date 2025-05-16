#pragma once

#include <cstdint>

#include "transport_cc.h"
#include "transport_config.h"
#include "util_endian.h"

namespace uccl {

typedef uint16_t PullQuanta;
/**
 * Uccl Packet Header just after UDP header.
 */
struct __attribute__((packed)) UcclPktHdr {
    static constexpr uint16_t kMagic = 0x4e53;
    be16_t magic;  // Magic value tagged after initialization for the flow.
    uint16_t engine_id : 4;  // remote UcclEngine ID to process this packet.
    uint16_t path_id : 12;   // path_id of this dst port.
    enum class UcclFlags : uint8_t {
        kData = 0b0,            // Data packet.
        kAck = 0b10,            // ACK packet.
        kDataRttProbe = 0b100,  // RTT probing packet.
        kAckRttProbe = 0b1000,  // RTT probing packet.
    };
    UcclFlags net_flags;  // Network flags.
    uint8_t msg_flags;    // Field to reflect the `FrameDesc' flags.
    be16_t frame_len;     // Length of the frame.
    be64_t flow_id;       // Flow ID to denote the connection.
    be32_t seqno;  // Sequence number to denote the packet counter in the flow.
    be32_t ackno;  // Sequence number to denote the packet counter in the flow.
    uint64_t timestamp1;  // Filled by sender with calibration for output queue
    uint64_t timestamp2;  // Filled by recver eBPF
    be16_t pullno;    // Receiver-driven congestion control.
    be16_t pad1;      // For GPU memory copy
    be32_t pad2;      // For GPU memory copy
};

struct __attribute__((packed)) UcclPullHdr {
    be16_t magic;  // Magic value tagged after initialization for the flow.
    uint16_t engine_id : 4;  // remote UcclEngine ID to process this packet.
    uint16_t path_id : 12;   // path_id of this dst port.
    UcclPktHdr::UcclFlags net_flags;  // Network flags.
    uint8_t msg_flags;    // Field to reflect the `FrameDesc' flags.
    be16_t frame_len;     // Length of the frame.
    be64_t flow_id;       // Flow ID to denote the connection.
    be16_t pullno;    // Receiver-driven congestion control.
    be16_t pad1;
};

struct __attribute__((packed)) UcclSackHdr {
    uint64_t timestamp3;  // Filled by recer with calibration for output queue
    be64_t sack_bitmap[kSackBitmapSize /
                       swift::Pcb::kSackBitmapBucketSize];  // Bitmap of the
                                                            // SACKs received.
    be16_t sack_bitmap_count;  // Length of the SACK bitmap [0-256].
    be32_t rwnd;        // Receiver window size in terms of packets.
};

static const size_t kUcclPktHdrLen = sizeof(UcclPktHdr);
static const size_t kUcclPullHdrLen = sizeof(UcclPullHdr);
static_assert(kUcclPullHdrLen <= EFA_MAX_INLINE_SIZE, "PullHdr too large for inline");
#if defined(USE_SRD) && defined(SRD_RDMA_WRITE)
static const size_t kUcclPktDataMaxLen = kSRDChunkSize;
#else
static const size_t kUcclPktDataMaxLen = EFA_MAX_PAYLOAD - kUcclPktHdrLen;
#endif
static const size_t kUcclSackHdrLen = sizeof(UcclSackHdr);

inline UcclPktHdr::UcclFlags operator|(UcclPktHdr::UcclFlags lhs,
                                       UcclPktHdr::UcclFlags rhs) {
    using UcclFlagsType = std::underlying_type<UcclPktHdr::UcclFlags>::type;
    return UcclPktHdr::UcclFlags(static_cast<UcclFlagsType>(lhs) |
                                 static_cast<UcclFlagsType>(rhs));
}

inline UcclPktHdr::UcclFlags operator&(UcclPktHdr::UcclFlags lhs,
                                       UcclPktHdr::UcclFlags rhs) {
    using UcclFlagsType = std::underlying_type<UcclPktHdr::UcclFlags>::type;
    return UcclPktHdr::UcclFlags(static_cast<UcclFlagsType>(lhs) &
                                 static_cast<UcclFlagsType>(rhs));
}

}  // namespace uccl
