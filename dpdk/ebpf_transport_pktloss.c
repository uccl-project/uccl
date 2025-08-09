// fmt: off
#include <linux/types.h>
// fmt: on
#include <bpf/bpf_helpers.h>
#include <linux/bpf.h>
#include <linux/if_packet.h>
#include <linux/if_vlan.h>
#include <linux/in.h>

#include "ebpf_util.h"

struct {
    __uint(type, BPF_MAP_TYPE_XSKMAP);
    __type(key, __u32);
    __type(value, __u32);
    __uint(max_entries, 64);
} xsks_map SEC(".maps");

// #define USE_TCP

#ifdef USE_TCP
#define kNetHdrLen \
    (sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct tcphdr))
#else
#define kNetHdrLen \
    (sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct udphdr))
#endif
#define kMagic 0x4e53
#define kUcclHdrLen 40
#define BASE_PORT 10000

SEC("ebpf_transport")
int ebpf_transport_filter(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;
    if (data + kNetHdrLen + kUcclHdrLen > data_end) return XDP_PASS;

    struct ethhdr *eth = data;
    struct iphdr *ip = data + sizeof(struct ethhdr);
    __u16 magic = *(__u16 *)(data + kNetHdrLen);

#ifdef USE_TCP
    if (eth->h_proto != __constant_htons(ETH_P_IP) ||
        ip->protocol != IPPROTO_TCP || magic != __constant_htons(kMagic)) {
        return XDP_PASS;
    }
#else
    if (eth->h_proto != __constant_htons(ETH_P_IP) ||
        ip->protocol != IPPROTO_UDP || magic != __constant_htons(kMagic)) {
        return XDP_PASS;
    }
#endif

    if (bpf_get_prandom_u32() % 1000 == 0) return XDP_DROP;

    __u8 net_flags = *(__u8 *)(data + kNetHdrLen + 4);

    switch (net_flags) {
        case 0b10000: {
            // Receiver receiving RTT probing + data packet.
            void *rtt_probe =
                data + kNetHdrLen + kUcclHdrLen - sizeof(__u64) * 2;

            __u64 *rx_ns_p = (__u64 *)(rtt_probe + sizeof(__u64));
            *rx_ns_p = bpf_ktime_get_ns();

            break;
        }
        case 0b100000: {
            // Sender receiving RTT probing + ACK packet.
            void *rtt_probe = data + kNetHdrLen + kUcclHdrLen;
            if (rtt_probe + 16 > data_end) return XDP_PASS;

            __u64 *rx_ns_p = (__u64 *)(rtt_probe + sizeof(__u64));
            *rx_ns_p = bpf_ktime_get_ns();

            break;
        }
        default:
            break;
    }

    return bpf_redirect_map(&xsks_map, ctx->rx_queue_index, XDP_PASS);
}

char _license[] SEC("license") = "GPL";
