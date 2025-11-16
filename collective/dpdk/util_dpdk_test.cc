#include "dpdk.h"
#include "ether.h"
#include "ipv4.h"
#include "packet.h"
// #include "transport_config.h"
#include "udp.h"
#include "util/util.h"
#include "util_dpdk.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <pthread.h>
#include <signal.h>
#include <vector>

using namespace uccl;

#define BASE_PORT 1000

const uint32_t MY_NUM_QUEUES = 1;

uint32_t server_addr_u32 = 0x0;
uint32_t client_addr_u32 = 0x0;
const uint16_t client_ports[8] = {40000, 40001, 40002, 40003,
                                  40004, 40005, 40006, 40007};

// static char server_mac_str[] = "9c:dc:71:49:a8:81";
// static char client_mac_str[] = "9c:dc:71:5e:2f:21";
// static char server_ip_str[] = "10.10.1.2";
// static char client_ip_str[] = "10.10.1.1";

static char server_mac_str[] = "6c:92:bf:f3:2e:1a";
static char client_mac_str[] = "6c:92:bf:f3:83:1e";
static char server_ip_str[] = "10.10.0.1";
static char client_ip_str[] = "10.10.0.2";

static bool is_client = true;

uint8_t server_mac_char[6] = {};
uint8_t client_mac_char[6] = {};

int const MY_SEND_BATCH_SIZE = 32;
int const MY_RECV_BATCH_SIZE = 32;
// 256 is reserved for xdp_meta, 42 is reserved for eth+ip+udp
// Max payload under AFXDP is 4096-256-42;
int const PAYLOAD_BYTES = 64;
// tune this to change packet rate
int const MAX_INFLIGHT_PKTS = 200000;
// sleep gives unstable rate and latency
int const SEND_INTV_US = 0;
int const RTO_US = 2000;

struct socket_t {
  DPDKSocket *dpdk_socket;
  std::atomic<uint64_t> sent_packets;
  uint64_t last_stall_time;
  uint32_t counter;
  std::vector<uint64_t> rtts;
  std::mutex rtts_lock;
};

struct client_t {
  struct socket_t socket[MY_NUM_QUEUES];
  pthread_t send_thread[MY_NUM_QUEUES];
  pthread_t recv_thread[MY_NUM_QUEUES];
  pthread_t stats_thread;
  uint64_t previous_sent_packets;
};

static struct client_t client;
std::atomic<uint64_t> inflight_pkts{0};
bool volatile quit;

static void *stats_thread(void *arg);
static void *send_thread(void *arg);
static void *recv_thread(void *arg);

int client_init(struct client_t *client, DPDKFactory *dpdk_factory) {
  // per-CPU socket setup

  for (uint32_t i = 0; i < MY_NUM_QUEUES; i++) {
    client->socket[i].dpdk_socket = dpdk_factory->CreateSocket(i);
  }

  int ret;

  // create socket threads
  for (uint32_t i = 0; i < MY_NUM_QUEUES; i++) {
    ret = pthread_create(&client->recv_thread[i], NULL, recv_thread,
                         &client->socket[i]);
    if (ret) {
      printf("\nerror: could not create socket recv thread #%d\n\n", i);
      return 1;
    }

    if (is_client) {
      ret = pthread_create(&client->send_thread[i], NULL, send_thread,
                           &client->socket[i]);
      if (ret) {
        printf("\nerror: could not create socket send thread #%d\n\n", i);
        return 1;
      }
    }
  }

  // create stats thread
  ret = pthread_create(&client->stats_thread, NULL, stats_thread, client);
  if (ret) {
    printf("\nerror: could not create stats thread\n\n");
    return 1;
  }

  return 0;
}

void client_shutdown(struct client_t *client) {
  assert(client);

  for (uint32_t i = 0; i < MY_NUM_QUEUES; i++) {
    pthread_join(client->recv_thread[i], NULL);
    if (is_client) {
      pthread_join(client->send_thread[i], NULL);
    }
  }
  pthread_join(client->stats_thread, NULL);
}

void interrupt_handler(int signal) {
  (void)signal;
  quit = true;
}

static void cleanup() {
  client_shutdown(&client);
  fflush(stdout);
}

int client_generate_packet(Packet *pkt, uint32_t payload_bytes, uint32_t counter) {

  uint32_t packet_len =
      sizeof(Ethernet) + sizeof(Ipv4) + sizeof(Udp) + payload_bytes;

  // generate ethernet header
  Ethernet *eth = pkt->append<Ethernet *>(packet_len);
  eth->dst_addr = server_mac_char;
  eth->src_addr = client_mac_char;
  eth->eth_type = be16_t(Ethernet::kIpv4);

  // generate ip header
  Ipv4 *ipv4 = reinterpret_cast<Ipv4 *>(eth + 1);
  ipv4->version_ihl = 0x45;
  ipv4->type_of_service = 0;
  ipv4->packet_id = be16_t(0x1513);
  ipv4->fragment_offset = be16_t(0);
  ipv4->time_to_live = Ipv4::kDefaultTTL;
  ipv4->total_length = be16_t(packet_len - sizeof(Ethernet));
  ipv4->next_proto_id = Ipv4::Proto::kUdp;
  ipv4->src_addr = client_addr_u32;
  ipv4->dst_addr = server_addr_u32;
  ipv4->hdr_checksum = 0;

  // generate udp header: using different ports to bypass per-flow rate
  // limiting
  Udp *udp = reinterpret_cast<Udp *>(ipv4 + 1);
  udp->src_port =
      client_ports[counter % (sizeof(client_ports) / sizeof(client_ports[0]))];
  udp->dst_port = BASE_PORT;
  udp->len = be16_t(packet_len - sizeof(Ethernet) - sizeof(Ipv4));
  udp->cksum = be16_t(0);

  // generate udp payload
  uint8_t *payload = reinterpret_cast<uint8_t *>(udp + 1);
  auto now = std::chrono::high_resolution_clock::now();
  uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        now.time_since_epoch())
                        .count();
  assert(payload_bytes >= sizeof(uint64_t) + sizeof(uint32_t));
  memcpy(payload, &now_us, sizeof(uint64_t));
  memcpy(payload + sizeof(uint64_t), &counter, sizeof(uint32_t));

  // Offload IPv4 and UDP checksums to hardware.
  pkt->set_l2_len(sizeof(Ethernet));
  pkt->set_l3_len(sizeof(Ipv4));
  pkt->offload_udpv4_csum();

  return packet_len;
}

void server_generate_packet(Packet *pkt) {
  Ethernet *eth = pkt->head_data<Ethernet *>();
  Ipv4 *ipv4 = reinterpret_cast<Ipv4 *>(eth + 1);
  Udp *udp = reinterpret_cast<Udp *>(ipv4 + 1);

  std::swap(eth->src_addr, eth->dst_addr);
  std::swap(ipv4->src_addr, ipv4->dst_addr);
  std::swap(udp->src_port, udp->dst_port);

  // Offload IPv4 and UDP checksums to hardware.
  ipv4->hdr_checksum = 0;
  udp->cksum = be16_t(0);
  pkt->set_l2_len(sizeof(Ethernet));
  pkt->set_l3_len(sizeof(Ipv4));
  pkt->offload_udpv4_csum();
}

void socket_send(struct socket_t *socket, Packet **pkts, int queue_id) {
  if (inflight_pkts >= MAX_INFLIGHT_PKTS) {
    auto now_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    if (socket->last_stall_time == 0) {
      socket->last_stall_time = now_us;
    } else if (now_us - socket->last_stall_time > RTO_US) {
      // These inflight packets get lost, we just ignore them
      LOG(INFO) << "#queue " << queue_id << " tx stall detected, forcing tx...";
      inflight_pkts = 0;
    }
    return;
  }
  socket->last_stall_time = 0;

  for (uint32_t i = 0; i < MY_SEND_BATCH_SIZE; i++) {
    // the 256B before frame_offset is xdp metedata
    pkts[i] = socket->dpdk_socket->pop_packet();
    client_generate_packet(pkts[i], PAYLOAD_BYTES, socket->counter + i);
  }
  inflight_pkts += MY_SEND_BATCH_SIZE;
  uint32_t completed =
      socket->dpdk_socket->send_packets(pkts, MY_SEND_BATCH_SIZE);
  socket->sent_packets += completed;
  socket->counter += completed;
}

void socket_recv(struct socket_t *socket, Packet **pkts, int queue_id) {
  // Check any packet received, in order to drive packet receiving path for
  // other kernel transport.
  uint32_t rcvd = socket->dpdk_socket->recv_packets(pkts, MY_RECV_BATCH_SIZE);

  if (rcvd > 0)
    LOG(INFO) << "rx rcvd = " << rcvd;
  else
    return;

  for (uint32_t i = 0; i < rcvd; i++) {
    auto *pkt = pkts[i];
    if (pkt->length() < sizeof(Ethernet)) {
      LOG(WARNING) << "received non-ethernet packet " << queue_id;
      socket->dpdk_socket->push_packet(pkt);
      continue;
    }

    Ethernet *eth = pkt->head_data<Ethernet *>();
    if (eth->eth_type.value() != Ethernet::kIpv4) [[unlikely]] {
      LOG(WARNING) << "received non-ipv4 packet " << std::hex << eth->eth_type.value();
      socket->dpdk_socket->push_packet(pkt);
      continue;
    }

    Ipv4 *ipv4 = reinterpret_cast<Ipv4 *>(eth + 1);

    if (pkt->length() != sizeof(Ethernet) + ipv4->total_length.value()) {
      LOG(WARNING) << "IPv4 packet length mismatch (expected: "
                   << ipv4->total_length.value()
                   << ", actual: " << pkt->length() << ")";
                   socket->dpdk_socket->push_packet(pkt);
      continue;
    }

    if (ipv4->next_proto_id != Ipv4::Proto::kUdp) [[unlikely]] {
      LOG(WARNING) << "received non-udp packet " << queue_id;
      socket->dpdk_socket->push_packet(pkt);
      continue;
    }

    Udp *udp = reinterpret_cast<Udp *>(ipv4 + 1);
    uint8_t *payload = reinterpret_cast<uint8_t *>(udp + 1);

    uint64_t now_us = *(uint64_t *)payload;
    // uint32_t counter = *(uint32_t *)(payload + sizeof(uint64_t));

    auto now = std::chrono::high_resolution_clock::now();
    uint64_t now_us2 = std::chrono::duration_cast<std::chrono::microseconds>(
                           now.time_since_epoch())
                           .count();
    uint64_t rtt = now_us2 - now_us;
    {
      std::lock_guard<std::mutex> lock(socket->rtts_lock);
      socket->rtts.push_back(rtt);
    }

    if (is_client) {
      socket->dpdk_socket->push_packet(pkts[i]);
    } else {
      server_generate_packet(pkts[i]);
    }
  }

  if (!is_client) {
    socket->dpdk_socket->send_packets(pkts, rcvd);
  }
}

static void *send_thread(void *arg) {
  struct socket_t *socket = (struct socket_t *)arg;

  int queue_id = socket->dpdk_socket->get_queue_id();

  printf("started socket send thread for queue #%d\n", queue_id);

  pin_thread_to_cpu(queue_id);

  Packet **pkts = new Packet *[MY_SEND_BATCH_SIZE];

  while (!quit) {
    socket_send(socket, pkts, queue_id);
    if (SEND_INTV_US)
      usleep(SEND_INTV_US);
  }

  delete[] pkts;

  return NULL;
}

static void *recv_thread(void *arg) {
  struct socket_t *socket = (struct socket_t *)arg;
  int queue_id = socket->dpdk_socket->get_queue_id();

  printf("started socket recv thread for queue #%d\n", queue_id);

  pin_thread_to_cpu(MY_NUM_QUEUES + queue_id);

  Packet **pkts = new Packet *[MY_RECV_BATCH_SIZE];

  while (!quit) {
    socket_recv(socket, pkts, queue_id);
  }

  delete[] pkts;

  LOG(INFO) << "Stopped socket recv thread for queue #" << queue_id;

  return NULL;
}

uint64_t aggregate_sent_packets(struct client_t *client) {
  uint64_t sent_packets = 0;
  for (uint32_t i = 0; i < MY_NUM_QUEUES; i++)
    sent_packets += client->socket[i].sent_packets.load();
  return sent_packets;
}

std::vector<uint64_t> aggregate_rtts(struct client_t *client) {
  std::vector<uint64_t> rtts;
  for (uint32_t i = 0; i < MY_NUM_QUEUES; i++) {
    std::lock_guard<std::mutex> lock(client->socket[i].rtts_lock);
    rtts.insert(rtts.end(), client->socket[i].rtts.begin(),
                client->socket[i].rtts.end());
  }
  return rtts;
}

static void *stats_thread(void *arg) {
  struct client_t *client = (struct client_t *)arg;

  auto start = std::chrono::high_resolution_clock::now();
  auto start_pkts = aggregate_sent_packets(client);
  auto end = start;
  auto end_pkts = start_pkts;
  while (!quit) {
    // Put before usleep to avoid counting it for tput calculation
    end = std::chrono::high_resolution_clock::now();
    end_pkts = aggregate_sent_packets(client);
    usleep(1000000);
    uint64_t sent_packets = aggregate_sent_packets(client);
    auto rtts = aggregate_rtts(client);
    auto med_latency = Percentile(rtts, 50);
    auto tail_latency = Percentile(rtts, 99);
    uint64_t sent_delta = sent_packets - client->previous_sent_packets;
    client->previous_sent_packets = sent_packets;

    printf("send delta: %lu, med rtt: %lu us, tail rtt: %lu us\n", sent_delta,
           med_latency, tail_latency);
  }
  uint64_t duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  auto throughput = (end_pkts - start_pkts) * 1.0 / duration * 1000;
  // 42B: eth+ip+udp, 24B: 4B FCS + 8B frame delimiter + 12B interframe gap
  auto bw_gbps = throughput * (PAYLOAD_BYTES + 42 + 24) * 8.0 / 1024 / 1024;

  auto rtts = aggregate_rtts(client);
  auto med_latency = Percentile(rtts, 50);
  auto tail_latency = Percentile(rtts, 99);

  printf("Throughput: %.2f Kpkts/s, BW: %.2f Gbps, med rtt: %lu us, tail rtt: "
         "%lu us\n",
         throughput, bw_gbps, med_latency, tail_latency);

  return NULL;
}

// 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
// TO RUN THE TEST:
// On server: GLOG_minloglevel=0 sudo ./uccl-util-dpdk-test --server
// On client: GLOG_minloglevel=0 sudo ./uccl-util-dpdk-test --client

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = false;

  if (argc < 2) {
    printf("Usage: %s --server | --client\n", argv[0]);
    return 1;
  }

  signal(SIGINT, interrupt_handler);
  signal(SIGTERM, interrupt_handler);
  signal(SIGHUP, interrupt_handler);
  signal(SIGALRM, interrupt_handler);

  client_addr_u32 = str_to_ip(client_ip_str);
  server_addr_u32 = str_to_ip(server_ip_str);
  DCHECK(str_to_mac(client_mac_str, reinterpret_cast<char*>(client_mac_char)));
  DCHECK(str_to_mac(server_mac_str, reinterpret_cast<char*>(server_mac_char)));

  Dpdk dpdk;
  dpdk.InitDpdk(1, argv);

  uint16_t client_port_id = -1;

  if (strcmp(argv[1], "--server") == 0) {
    is_client = false;
    LOG(INFO) << "[server] " << server_mac_str << " " << server_ip_str;
    client_port_id = dpdk.GetPmdPortIdByMac(server_mac_str);
  } else if (strcmp(argv[1], "--client") == 0) {
    LOG(INFO) << "[client] " << client_mac_str << " " << client_ip_str;
    client_port_id = dpdk.GetPmdPortIdByMac(client_mac_str);
  } else {
    printf("Usage: %s --server | --client\n", argv[0]);
    return 1;
  }

  if (client_port_id == (uint16_t)-1) {
    LOG(INFO) << "Client port not found";
    return 1;
  }

  DPDKFactory dpdk_factory(client_port_id, MY_NUM_QUEUES, MY_NUM_QUEUES);
  dpdk_factory.Init();

  if (client_init(&client, &dpdk_factory) != 0) {
    cleanup();
    return 1;
  }

  LOG(INFO) << "Started running";

  while (!quit) {
    usleep(1000);
  }

  cleanup();

  printf("\n");

  return 0;
}
