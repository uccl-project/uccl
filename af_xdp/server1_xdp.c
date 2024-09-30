// fmt: off
#include <linux/types.h>
// fmt: on
#include <bpf/bpf_helpers.h>
#include <linux/bpf.h>
#include <linux/if_packet.h>
#include <linux/if_vlan.h>
#include <linux/in.h>

#include "util_xdp.h"

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, int);
    __type(value, __u64);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} received_packets_map SEC(".maps");

SEC("server_xdp")
int server_xdp_filter(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;
    if ((void *)eth + sizeof(struct ethhdr) > data_end) return XDP_PASS;
    if (eth->h_proto != __constant_htons(ETH_P_IP)) return XDP_PASS;

    struct iphdr *ip = data + sizeof(struct ethhdr);
    if ((void *)ip + sizeof(struct iphdr) > data_end) return XDP_PASS;
    if (ip->protocol != IPPROTO_UDP) return XDP_PASS;

    struct udphdr *udp = (void *)ip + sizeof(struct iphdr);
    if ((void *)udp + sizeof(struct udphdr) > data_end) return XDP_PASS;
    if (udp->dest != __constant_htons(40000)) return XDP_PASS;

    void *payload = (void *)udp + sizeof(struct udphdr);
    int payload_bytes = data_end - payload;

    debug_printf("server received %d byte packet", payload_bytes);

    int zero = 0;
    __u64 *packets_received =
        (__u64 *)bpf_map_lookup_elem(&received_packets_map, &zero);

    if (packets_received) {
        __sync_fetch_and_add(packets_received, 1);
    }

    prepare_packet(eth, ip, udp);

    return XDP_TX;
}

char _license[] SEC("license") = "GPL";
