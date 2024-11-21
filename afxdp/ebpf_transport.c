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

// #define USING_TCP

#ifdef USING_TCP
#define kNetHdrLen \
    (sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct tcphdr))
#else
#define kNetHdrLen \
    (sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct udphdr))
#endif
#define kMagic 0x4e53
#define kUcclHdrLen 24
#define BASE_PORT 10000

SEC("ebpf_transport")
int ebpf_transport_filter(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;
    if (data + kNetHdrLen + kUcclHdrLen > data_end) return XDP_PASS;

    struct ethhdr *eth = data;
    struct iphdr *ip = data + sizeof(struct ethhdr);
    __u16 magic = *(__u16 *)(data + kNetHdrLen);

#ifdef USING_TCP
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

    __u8 *net_flags_p = (__u8 *)(data + kNetHdrLen + 4);

    switch (*net_flags_p) {
        case 0b10000: {
            // RTT probing packet.
            void *rtt_probe = data + kNetHdrLen + kUcclHdrLen;
            if (rtt_probe + 10 > data_end) return XDP_PASS;

            *net_flags_p = 0b100000;
            // See craft_rttprobe_packet() in transport.cc
            __u16 *reverse_dst_port = (__u16 *)rtt_probe;

            __u64 rx_ns = bpf_ktime_get_ns();
            __u64 *tx_ns_p = (__u64 *)(data + kNetHdrLen + kUcclHdrLen + 2);
            *tx_ns_p = rx_ns;

            struct udphdr *udp =
                data + sizeof(struct ethhdr) + sizeof(struct iphdr);
            __u16 ori_dst_port = udp->dest;

            reverse_packet_l2_l3_wo_csum(eth, ip);

            udp->source = __constant_htons(BASE_PORT);
            udp->dest = *reverse_dst_port;
            *reverse_dst_port = ori_dst_port;

            ip->check = compute_ip_checksum(ip);
            return XDP_TX;
        }
        case 0b100000: {
            // RTT probing response packet.
            void *rtt_probe = data + kNetHdrLen + kUcclHdrLen;
            if (rtt_probe + 10 > data_end) return XDP_PASS;

            *net_flags_p = 0b1000000;
            __u16 *reverse_dst_port = (__u16 *)rtt_probe;

            struct udphdr *udp =
                data + sizeof(struct ethhdr) + sizeof(struct iphdr);
            __u16 ori_dst_port = udp->dest;

            reverse_packet_l2_l3_wo_csum(eth, ip);

            udp->source = __constant_htons(BASE_PORT);
            udp->dest = *reverse_dst_port;
            *reverse_dst_port = ori_dst_port;

            ip->check = compute_ip_checksum(ip);
            return XDP_TX;
        }
        case 0b1000000: {
            // RTT probing response2 packet.
            void *rtt_probe = data + kNetHdrLen + kUcclHdrLen;
            if (rtt_probe + 10 > data_end) return XDP_PASS;

            __u64 rx_ns = bpf_ktime_get_ns();
            __u64 *tx_ns_p = (__u64 *)(data + kNetHdrLen + kUcclHdrLen + 2);
            __u64 rtt = rx_ns - *tx_ns_p;
            *tx_ns_p = rtt;
            break;
        }
        default:
            break;
    }

    return bpf_redirect_map(&xsks_map, ctx->rx_queue_index, XDP_PASS);
}

char _license[] SEC("license") = "GPL";
