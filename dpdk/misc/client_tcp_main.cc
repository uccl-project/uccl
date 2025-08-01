/*
 *  A client timing the roundtrip time of a message sent to a server multiple
 * times. Usage: ./client.out -a <address> -p <port> -b <message_size (bytes)>
 */
#include "util.h"
#include "util_tcp.h"
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>
#include <ctype.h>
#include <fcntl.h>
#include <inttypes.h>
#include <netdb.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

using namespace uccl;

void error(char* msg) {
  perror(msg);
  exit(0);
}

int const MY_SEND_BATCH_SIZE = 1;
int const MY_RECV_BATCH_SIZE = 32;
// tune this to change packet rate
int const MAX_INFLIGHT_PKTS = 1;
int const SEND_INTV_US = 0;

std::vector<uint64_t> rtts;
std::mutex rtts_lock;
std::atomic<uint64_t> sent_packets{0};
std::atomic<uint64_t> inflight_pkts{0};

// TODO(Yang): using multiple TCP connections to saturate the link
static void* send_thread(void* arg) {
  pin_thread_to_cpu(1);

  struct Config* config = (struct Config*)arg;
  int sockfd = config->sockfds[0];
  uint8_t* wbuffer = (uint8_t*)malloc(config->n_bytes);
  while (!quit) {
    if (inflight_pkts >= MAX_INFLIGHT_PKTS) {
      continue;
    }
    for (int i = 0; i < MY_SEND_BATCH_SIZE; i++) {
      auto now = std::chrono::high_resolution_clock::now();
      *(uint64_t*)wbuffer =
          std::chrono::duration_cast<std::chrono::microseconds>(
              now.time_since_epoch())
              .count();
      *(uint32_t*)(wbuffer + sizeof(uint64_t)) = inflight_pkts + i;

      send_message(config->n_bytes, sockfd, wbuffer, &quit);
    }
    inflight_pkts += MY_SEND_BATCH_SIZE;
    sent_packets += MY_SEND_BATCH_SIZE;
    if (SEND_INTV_US) usleep(SEND_INTV_US);
  }
  free(wbuffer);
  return NULL;
}

static void* recv_thread(void* arg) {
  pin_thread_to_cpu(2);

  struct Config* config = (struct Config*)arg;
  int sockfd = config->sockfds[0];
  uint8_t* rbuffer = (uint8_t*)malloc(config->n_bytes);
  while (!quit) {
    for (int i = 0; i < MY_RECV_BATCH_SIZE; i++) {
      receive_message(sizeof(uint64_t), sockfd, rbuffer, &quit);
      inflight_pkts--;

      uint64_t now_us = *(uint64_t*)rbuffer;
      uint32_t counter = *(uint32_t*)(rbuffer + sizeof(uint64_t));

      auto now = std::chrono::high_resolution_clock::now();
      uint64_t now_us2 = std::chrono::duration_cast<std::chrono::microseconds>(
                             now.time_since_epoch())
                             .count();
      uint64_t rtt = now_us2 - now_us;
      {
        std::lock_guard<std::mutex> lock(rtts_lock);
        rtts.push_back(rtt);
      }
    }
  }
  free(rbuffer);
  return NULL;
}

static void* send_recv_thread(void* arg) {
  pin_thread_to_cpu(1);

  struct Config* config = (struct Config*)arg;
  int sockfd = config->sockfds[0];
  // size_t newsize = 10240;
  // setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &newsize, sizeof(newsize));
  // setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &newsize, sizeof(newsize));
  uint8_t* rwbuffer = (uint8_t*)malloc(config->n_bytes);
  while (!quit) {
    for (int i = 0; i < MY_RECV_BATCH_SIZE; i++) {
      sent_packets++;

      auto now = std::chrono::high_resolution_clock::now();
      send_message(config->n_bytes, sockfd, rwbuffer, &quit);
      receive_message(config->n_bytes, sockfd, rwbuffer, &quit);
      auto now2 = std::chrono::high_resolution_clock::now();
      uint64_t rtt =
          std::chrono::duration_cast<std::chrono::microseconds>(now2 - now)
              .count();
      {
        std::lock_guard<std::mutex> lock(rtts_lock);
        rtts.push_back(rtt);
      }
    }
  }
  free(rwbuffer);
  return NULL;
}

static void* stats_thread(void* arg) {
  auto start = std::chrono::high_resolution_clock::now();
  auto start_pkts = sent_packets.load();
  auto end = start;
  auto end_pkts = start_pkts;
  uint64_t previous_sent_packets = sent_packets;
  while (!quit) {
    end = std::chrono::high_resolution_clock::now();
    end_pkts = sent_packets.load();
    usleep(1000000);
    uint64_t med_latency, tail_latency;
    {
      std::lock_guard<std::mutex> lock(rtts_lock);
      med_latency = Percentile(rtts, 50);
      tail_latency = Percentile(rtts, 99);
    }
    uint64_t sent_delta = sent_packets - previous_sent_packets;
    previous_sent_packets = sent_packets;

    printf("send delta: %lu, med rtt: %lu us, tail rtt: %lu us\n", sent_delta,
           med_latency, tail_latency);
  }
  uint64_t duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  auto throughput = (end_pkts - start_pkts) * 1.0 / duration * 1000;
  // 54B: eth+ip+tcp, 24B: 4B FCS + 8B frame delimiter + 12B interframe gap
  // 9000B: MTU
  auto bw_gbps = throughput * (PAYLOAD_BYTES * ((54 + 24 + MTU) * 1.0 / MTU)) *
                 8.0 / 1024 / 1024;

  uint64_t med_latency, tail_latency;
  {
    std::lock_guard<std::mutex> lock(rtts_lock);
    med_latency = Percentile(rtts, 50);
    tail_latency = Percentile(rtts, 99);
  }

  printf(
      "Throughput: %.2f Kpkts/s, BW: %.2f Gbps, med rtt: %lu us, tail rtt: "
      "%lu us\n",
      throughput, bw_gbps, med_latency, tail_latency);

  return NULL;
}

void clean_shutdown_handler(int signal) {
  (void)signal;
  quit = true;
}

int main(int argc, char* argv[]) {
  signal(SIGALRM, clean_shutdown_handler);
  alarm(10);

  int sockfd;
  struct sockaddr_in serv_addr;
  struct hostent* server;

  struct Config config = get_config(argc, argv);

  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    error("ERROR opening socket");
  }
  server = gethostbyname(config.address);
  if (server == NULL) {
    fprintf(stderr, "ERROR, no such host\n");
    exit(0);
  }
  bzero((char*)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  bcopy((char*)server->h_addr, (char*)&serv_addr.sin_addr.s_addr,
        server->h_length);
  serv_addr.sin_port = htons(config.port);

  // Connect and set nonblocking and nodelay
  if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
    error("ERROR connecting");
  }
  fcntl(sockfd, F_SETFL, O_NONBLOCK);
  int flag = 1;
  setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (void*)&flag, sizeof(int));
  config.sockfds[0] = sockfd;

  printf("Connection successful! Starting...\n");
  fflush(stdout);

#ifdef PING_PONG_MSG
  pthread_t send_recv_thread_ctl;
  if (pthread_create(&send_recv_thread_ctl, NULL, send_recv_thread, &config)) {
    error("ERROR creating send_recv thread");
  }
#else
  pthread_t recv_thread_ctl;
  if (pthread_create(&recv_thread_ctl, NULL, recv_thread, &config)) {
    error("ERROR creating recv thread");
  }

  pthread_t send_thread_ctl;
  if (pthread_create(&send_thread_ctl, NULL, send_thread, &config)) {
    error("ERROR creating send thread");
  }
#endif

  pthread_t stats_thread_ctl;
  if (pthread_create(&stats_thread_ctl, NULL, stats_thread, NULL)) {
    printf("\nerror: could not create stats thread\n\n");
    return 1;
  }

  while (!quit) {
    usleep(1000);
  }

#ifdef PING_PONG_MSG
  pthread_join(send_recv_thread_ctl, NULL);
#else
  pthread_join(recv_thread_ctl, NULL);
  pthread_join(send_thread_ctl, NULL);
#endif
  pthread_join(stats_thread_ctl, NULL);

  close(sockfd);

  return 0;
}
