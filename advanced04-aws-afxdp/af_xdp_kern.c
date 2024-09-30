/* SPDX-License-Identifier: GPL-2.0 */

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>

#include <bpf/bpf_helpers.h>

#define SERVER_PORT 8889
#define UDP_PROTO 17
#define MAGIC_NUMBER 0xdeadbeef
#define htons(x) ((__be16)___constant_swab16((x)))

struct {
	__uint(type, BPF_MAP_TYPE_XSKMAP);
	__type(key, __u32);
	__type(value, __u32);
	__uint(max_entries, 64);
} xsks_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, __u32);
	__type(value, __u32);
	__uint(max_entries, 64);
} xdp_stats_map SEC(".maps");

SEC("xdp")
int xdp_sock_prog(struct xdp_md *ctx)
{
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    struct ethhdr *eth = data;
    if (eth + 1 > data_end) return XDP_PASS;

    struct iphdr *ip = data + sizeof(*eth);
    if (ip + 1 > data_end) return XDP_PASS;

    // Let AFXDP process all UDP traffic!
    if (ip->protocol != UDP_PROTO) return XDP_PASS;

    struct udphdr *udp = (struct udphdr *)((char *)ip + (ip->ihl << 2));
    if (udp + 1 > data_end) return XDP_PASS;

    // char* payload = (char*)udp + sizeof(struct udphdr);
    // if (*(__u64*)payload != MAGIC_NUMBER) return XDP_PASS;

    // How to specify PORT in libfabric UDP?
    // if (udp->dest != htons(SERVER_PORT)) return XDP_PASS;

    int index = ctx->rx_queue_index;
    // __u32 *pkt_count;

    // pkt_count = bpf_map_lookup_elem(&xdp_stats_map, &index);
    // if (pkt_count) {

    //     /* We pass every other packet */
    //     if ((*pkt_count)++ & 1)
    //         return XDP_PASS;
    // }

    /* A set entry here means that the correspnding queue_id
     * has an active AF_XDP socket bound to it. */
    if (bpf_map_lookup_elem(&xsks_map, &index))
        return bpf_redirect_map(&xsks_map, index, 0);

    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
