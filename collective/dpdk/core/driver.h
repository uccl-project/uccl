#include <glog/logging.h>
#include <linux/types.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_icmp.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_memory.h>
#include <rte_mempool.h>

#define MAC_ADDRESS "9c:dc:71:56:af:35"

#define RX_RING_SIZE 128
#define TX_RING_SIZE 128
#define NUM_MBUFS ((8192 - 1))
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32

class Driver {
 public:
  Driver() : is_initialized(false) {}
  ~Driver() {
    if (is_initialized) close();
  }

  void start(int argc, char** argv) {
    if (rte_eal_init(argc, argv) < 0)
      rte_exit(EXIT_FAILURE, "Error with EAL init\n");

    LOG(INFO) << "EAL initialized";

    port_id = get_port_id(MAC_ADDRESS);
    LOG(INFO) << "Port ID: " << port_id;

    mbuf_pool =
        rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS, MBUF_CACHE_SIZE, 0,
                                RTE_MBUF_DEFAULT_BUF_SIZE, SOCKET_ID_ANY);

    if (mbuf_pool == NULL) rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

    LOG(INFO) << "Mbuf pool created";

    struct rte_eth_conf port_conf = {};

    // 1 rx queue, 1 tx queue
    if (rte_eth_dev_configure(port_id, 1, 1, &port_conf) < 0)
      rte_exit(EXIT_FAILURE, "Cannot configure device\n");

    LOG(INFO) << "Device configured";

    if (rte_eth_rx_queue_setup(port_id, 0, RX_RING_SIZE, SOCKET_ID_ANY, NULL,
                               mbuf_pool) < 0)
      rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup failed\n");

    LOG(INFO) << "RX queue setup";

    if (rte_eth_tx_queue_setup(port_id, 0, TX_RING_SIZE, SOCKET_ID_ANY, NULL) <
        0)
      rte_exit(EXIT_FAILURE, "TX queue setup failed\n");

    LOG(INFO) << "TX queue setup";

    if (rte_eth_dev_start(port_id) < 0)
      rte_exit(EXIT_FAILURE, "rte_eth_dev_start failed\n");

    LOG(INFO) << "Started listening on port " << port_id;

    is_initialized = true;
  }

  void close() {
    if (is_initialized) {
      rte_eth_dev_stop(port_id);
      rte_eth_dev_close(port_id);
      is_initialized = false;
      LOG(INFO) << "Device closed";
    }
  }

  inline void process_packet(struct rte_mbuf* mbuf) {
    struct rte_ether_hdr* eth;
    struct rte_ipv4_hdr* ip;
    struct rte_icmp_hdr* icmp;

    eth = rte_pktmbuf_mtod(mbuf, struct rte_ether_hdr*);

    if (eth->ether_type != rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)) {
      return;
    }

    ip = (struct rte_ipv4_hdr*)(eth + 1);

    if (ip->next_proto_id != IPPROTO_ICMP) {
      return;
    }

    icmp = (struct rte_icmp_hdr*)((unsigned char*)ip +
                                  sizeof(struct rte_ipv4_hdr));

    if (icmp->icmp_type == RTE_ICMP_TYPE_ECHO_REQUEST) {
      printf("Received ICMP Echo Request from %u.%u.%u.%u\n",
             (ip->src_addr) & 0xff, (ip->src_addr >> 8) & 0xff,
             (ip->src_addr >> 16) & 0xff, (ip->src_addr >> 24) & 0xff);
    }
  }

  inline void send_icmp_reply(struct rte_mbuf* req) {
    struct rte_ether_hdr* eth;
    struct rte_ipv4_hdr* ip;
    struct rte_icmp_hdr* icmp;

    eth = rte_pktmbuf_mtod(req, struct rte_ether_hdr*);
    ip = (struct rte_ipv4_hdr*)(eth + 1);
    icmp = (struct rte_icmp_hdr*)((unsigned char*)ip +
                                  sizeof(struct rte_ipv4_hdr));

    // 只处理 ICMP Echo Request
    if (icmp->icmp_type != RTE_ICMP_TYPE_ECHO_REQUEST) {
      rte_pktmbuf_free(req);
      return;
    }

    // 修改 ICMP 类型为 Echo Reply
    icmp->icmp_type = RTE_ICMP_TYPE_ECHO_REPLY;
    icmp->icmp_cksum = 0;
    icmp->icmp_cksum = rte_ipv4_udptcp_cksum(ip, icmp);

    // 交换源/目的 IP
    uint32_t tmp_ip = ip->src_addr;
    ip->src_addr = ip->dst_addr;
    ip->dst_addr = tmp_ip;

    // 更新 IP 校验和
    ip->hdr_checksum = 0;
    ip->hdr_checksum = rte_ipv4_cksum(ip);

    // 交换源/目的 MAC
    struct rte_ether_addr tmp_mac;
    rte_ether_addr_copy(&eth->src_addr, &tmp_mac);
    rte_ether_addr_copy(&eth->dst_addr, &eth->src_addr);
    rte_ether_addr_copy(&tmp_mac, &eth->dst_addr);

    // 发送
    uint16_t nb_tx = rte_eth_tx_burst(port_id, 0, &req, 1);
    if (nb_tx == 0) {
      rte_pktmbuf_free(req);  // 如果没发出去，记得释放
    }
  }

  void recv() {
    struct rte_mbuf* bufs[BURST_SIZE];
    const uint16_t nb_rx = rte_eth_rx_burst(port_id, 0, bufs, BURST_SIZE);

    for (uint16_t i = 0; i < nb_rx; i++) {
      process_packet(bufs[i]);
      send_icmp_reply(bufs[i]);
    }
  }

 private:
  uint16_t get_port_id(char const* mac_address) {
    uint16_t nb_ports = rte_eth_dev_count_avail();
    struct rte_ether_addr mac;
    char buf[32];

    for (uint16_t pid = 0; pid < nb_ports; pid++) {
      if (rte_eth_macaddr_get(pid, &mac) < 0) continue;

      mac_to_str(&mac, buf, sizeof(buf));

      if (strcasecmp(buf, mac_address) == 0) {
        return pid;
      }
    }
    return -1;  // 没找到
  }

  void mac_to_str(struct rte_ether_addr* mac, char* buf, size_t size) {
    snprintf(buf, size, "%02x:%02x:%02x:%02x:%02x:%02x", mac->addr_bytes[0],
             mac->addr_bytes[1], mac->addr_bytes[2], mac->addr_bytes[3],
             mac->addr_bytes[4], mac->addr_bytes[5]);
  }

 private:
  uint16_t port_id;
  struct rte_mempool* mbuf_pool;
  bool is_initialized;
};