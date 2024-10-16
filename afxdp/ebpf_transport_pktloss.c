// fmt: off
#include <linux/types.h>
// fmt: on
#include <bpf/bpf_helpers.h>
#include <linux/bpf.h>
#include <linux/if_packet.h>
#include <linux/if_vlan.h>
#include <linux/in.h>

#include "ebpf_util.h"
#include "transport_config.h"

struct {
    __uint(type, BPF_MAP_TYPE_XSKMAP);
    __type(key, __u32);
    __type(value, __u32);
    __uint(max_entries, 64);
} xsks_map SEC(".maps");

#ifdef USING_TCP
#define kNetHdrLen \
    (sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct tcphdr))
#else
#define kNetHdrLen \
    (sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct udphdr))
#endif
#define kMagic 0x4e53
#define kUcclHdrLen 46

SEC("ebpf_transport")
int ebpf_transport_filter(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;
    if (data + kNetHdrLen + kUcclHdrLen > data_end) return XDP_PASS;

    struct ethhdr *eth = data;
    struct iphdr *ip = data + sizeof(struct ethhdr);
    __u16 magic = *(__u16 *)(data + kNetHdrLen);

    if (eth->h_proto != __constant_htons(ETH_P_IP) ||
        ip->protocol != IPPROTO_TCP || magic != __constant_htons(kMagic)) {
        return XDP_PASS;
    }

    if (bpf_get_prandom_u32() % 1000 == 0) return XDP_DROP;

    return bpf_redirect_map(&xsks_map, ctx->rx_queue_index, XDP_PASS);
}

char _license[] SEC("license") = "GPL";
