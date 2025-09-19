/**
 * @file rdma_test.cc
 * @brief Test for UCCL RDMA transport
 */

#include "transport.h"
#include <cstdint>
#ifndef __HIP_PLATFORM_AMD__
#include "cuda_runtime.h"
#else
#include "hip/hip_runtime.h"
#endif
#include "transport_config.h"
#include "util/util.h"
#include "util_timer.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <thread>
#include <signal.h>

// #define GPU 7

using namespace uccl;

static bool volatile quit = false;

std::optional<RDMAEndpoint> ep;
struct {
  uint32_t port;
  uint32_t ip;
} local_p2p_listen;
struct {
  uint32_t port;
  uint32_t ip;
} remote_p2p_listen;

DEFINE_bool(server, false, "Whether this is a server receiving traffic.");
DEFINE_string(serverip, "", "Server IP address the client tries to connect.");
DEFINE_string(perftype, "tpt", "Performance type: basic/lat/tpt/bi.");
DEFINE_bool(warmup, false, "Whether to run warmup.");
DEFINE_uint32(nflow, 4, "Number of flows.");
DEFINE_uint32(nmsg, 1, "Number of messages within one request to post.");
DEFINE_uint32(nreq, 2, "Outstanding requests to post.");
DEFINE_uint32(msize, 1000000, "Size of message.");
DEFINE_uint32(iterations, 1000000, "Number of iterations to run.");
DEFINE_bool(flush, false, "Whether to flush after receiving.");
DEFINE_bool(bi, false, "Whether to run bidirectional test.");
DEFINE_uint32(oobport, 19999, "OOB port to use for bootstrapping.");

static void server_basic(ConnID conn_id, struct Mhandle* mhandle, void* data) {
  for (int i = 0; i < FLAGS_iterations; i++) {
    int len = 65536;
    void* recv_data = data;

    struct ucclRequest ureq;
    DCHECK(ep->uccl_recv_async((UcclFlow*)conn_id.context, &mhandle, &recv_data,
                               &len, 1, &ureq) == 0);

    ep->uccl_poll_ureq(&ureq);

    // verify data
    for (int i = 0; i < 65536 / 4; i++) {
      assert(((uint32_t*)data)[i] == 0x123456);
    }

    // VLOG(5) << "Iteration " << i << " done";
    std::cout << "Iteration " << i << " done" << std::endl;
  }
}

static void client_basic(ConnID conn_id, struct Mhandle* mhandle, void* data) {
  // Fill data in a pattern of 0x123456,0x123456,0x123456...
  for (int i = 0; i < 65536 / 4; i++) {
    ((uint32_t*)data)[i] = 0x123456;
  }

  for (int i = 0; i < FLAGS_iterations; i++) {
    void* send_data = data;
    struct ucclRequest ureq;
    while (ep->uccl_send_async((UcclFlow*)conn_id.context, mhandle, send_data,
                               65536, &ureq)) {
    }

    ep->uccl_poll_ureq(&ureq);

    // VLOG(5) << "Iteration " << i << " done";
    std::cout << "Iteration " << i << " done" << std::endl;
  }
}

static void server_lat(ConnID conn_id, struct Mhandle* mhandle, void* data) {
  // Latency is measured at server side as it is asynchronous receive
  std::vector<uint64_t> lat_vec;

  if (FLAGS_warmup) {
    for (int i = 0; i < 1000; i++) {
      int len = FLAGS_msize;
      void* recv_data = data;
      struct ucclRequest ureq;
      DCHECK(ep->uccl_recv_async((UcclFlow*)conn_id.context, &mhandle,
                                 &recv_data, &len, 1, &ureq) == 0);
      ep->uccl_poll_ureq(&ureq);
    }
  }

  for (int i = 0; i < FLAGS_iterations; i++) {
    int len = FLAGS_msize;
    void* recv_data = data;
    auto t1 = rdtsc();
    struct ucclRequest ureq;
    DCHECK(ep->uccl_recv_async((UcclFlow*)conn_id.context, &mhandle, &recv_data,
                               &len, 1, &ureq) == 0);
    ep->uccl_poll_ureq(&ureq);
    auto t2 = rdtsc();
    lat_vec.push_back(to_usec(t2 - t1, freq_ghz));
  }
  std::sort(lat_vec.begin(), lat_vec.end());
  std::cout << "Min: " << lat_vec[0] << "us" << std::endl;
  std::cout << "P50: " << lat_vec[FLAGS_iterations / 2] << "us" << std::endl;
  std::cout << "P90: " << lat_vec[FLAGS_iterations * 9 / 10] << "us"
            << std::endl;
  std::cout << "P99: " << lat_vec[FLAGS_iterations * 99 / 100] << "us"
            << std::endl;
  std::cout << "Max: " << lat_vec[FLAGS_iterations - 1] << "us" << std::endl;
}

static void client_lat(ConnID conn_id, struct Mhandle* mhandle, void* data) {
  if (FLAGS_warmup) {
    for (int i = 0; i < 1000; i++) {
      void* send_data = data;
      struct ucclRequest ureq;
      while (ep->uccl_send_async((UcclFlow*)conn_id.context, mhandle, send_data,
                                 FLAGS_msize, &ureq)) {
      }
      ep->uccl_poll_ureq(&ureq);
    }
  }

  for (int i = 0; i < FLAGS_iterations; i++) {
    void* send_data = data;
    struct ucclRequest ureq;
    while (ep->uccl_send_async((UcclFlow*)conn_id.context, mhandle, send_data,
                               FLAGS_msize, &ureq)) {
    }
    ep->uccl_poll_ureq(&ureq);
  }
}

uint64_t volatile tx_cur_sec_bytes = 0;
uint64_t tx_prev_sec_bytes = 0;
uint64_t volatile rx_cur_sec_bytes = 0;
uint64_t rx_prev_sec_bytes = 0;

uint64_t volatile c_itr = 0;
uint64_t volatile s_itr = 0;

static void server_tpt(std::vector<ConnID>& conn_ids,
                       std::vector<struct Mhandle*>& mhandles,
                       std::vector<void*>& datas) {
  s_itr = FLAGS_iterations;
  s_itr *= FLAGS_nflow;

  int len[FLAGS_nflow][FLAGS_nreq][FLAGS_nmsg];
  struct Mhandle* mhs[FLAGS_nflow][FLAGS_nreq][FLAGS_nmsg];
  void* recv_data[FLAGS_nflow][FLAGS_nreq][FLAGS_nmsg];

  int flag[FLAGS_nflow][FLAGS_nreq];
  memset(flag, 0, sizeof(int) * FLAGS_nflow * FLAGS_nreq);

  std::vector<std::vector<ucclRequest>> ureq_vec(FLAGS_nflow);
  for (int i = 0; i < FLAGS_nflow; i++) {
    ureq_vec[i].resize(FLAGS_nreq);
  }

  for (int f = 0; f < FLAGS_nflow; f++) {
    for (int r = 0; r < FLAGS_nreq; r++)
      for (int m = 0; m < FLAGS_nmsg; m++) {
        len[f][r][m] = FLAGS_msize;
        recv_data[f][r][m] = reinterpret_cast<char*>(datas[f]) +
                             r * FLAGS_msize * FLAGS_nmsg + m * FLAGS_msize;
        mhs[f][r][m] = mhandles[f];
      }
  }

  for (int f = 0; f < FLAGS_nflow; f++) {
    for (int r = 0; r < FLAGS_nreq; r++) {
      DCHECK(ep->uccl_recv_async((UcclFlow*)conn_ids[f].context, mhs[f][r],
                                 recv_data[f][r], len[f][r], FLAGS_nmsg,
                                 &ureq_vec[f][r]) == 0);
      s_itr--;
      rx_cur_sec_bytes += FLAGS_msize * FLAGS_nmsg;
    }
  }

  while (s_itr) {
    for (int f = 0; f < FLAGS_nflow; f++) {
      for (int r = 0; r < FLAGS_nreq; r++) {
        if (quit) {
          s_itr = 0;
          break;
        }
        if (!ep->uccl_poll_ureq_once(&ureq_vec[f][r])) continue;

        if (!FLAGS_flush) {
          s_itr--;
          if (s_itr == 0) break;
          DCHECK(ep->uccl_recv_async((UcclFlow*)conn_ids[f].context, mhs[f][r],
                                     recv_data[f][r], len[f][r], FLAGS_nmsg,
                                     &ureq_vec[f][r]) == 0);
          rx_cur_sec_bytes += FLAGS_msize * FLAGS_nmsg;
          continue;
        }

        if (flag[f][r] == 0) {
          DCHECK(ep->uccl_flush((UcclFlow*)conn_ids[f].context, mhs[f][r],
                                recv_data[f][r], len[f][r], FLAGS_nmsg,
                                &ureq_vec[f][r]) == 0);
          flag[f][r] = 1;
        } else if (flag[f][r] == 1) {
          s_itr--;
          if (s_itr == 0) break;
          DCHECK(ep->uccl_recv_async((UcclFlow*)conn_ids[f].context, mhs[f][r],
                                     recv_data[f][r], len[f][r], FLAGS_nmsg,
                                     &ureq_vec[f][r]) == 0);
          rx_cur_sec_bytes += FLAGS_msize * FLAGS_nmsg;
          flag[f][r] = 0;
        }
      }
    }
  }
}

static void client_tpt(std::vector<ConnID>& conn_ids,
                       std::vector<struct Mhandle*>& mhandles,
                       std::vector<void*>& datas) {
  c_itr = FLAGS_iterations;
  c_itr *= FLAGS_nflow;

  std::vector<std::vector<std::vector<struct ucclRequest>>> ureq_vec(
      FLAGS_nflow);
  for (int f = 0; f < FLAGS_nflow; f++) {
    ureq_vec[f].resize(FLAGS_nreq);
    for (int r = 0; r < FLAGS_nreq; r++) {
      ureq_vec[f][r].resize(FLAGS_nmsg);
      for (int n = 0; n < FLAGS_nmsg; n++) ureq_vec[f][r][n] = {};
    }
  }
  for (int f = 0; f < FLAGS_nflow; f++) {
    for (int r = 0; r < FLAGS_nreq; r++) {
      for (int n = 0; n < FLAGS_nmsg; n++) {
        void* send_data = reinterpret_cast<char*>(datas[f]) +
                          r * FLAGS_msize * FLAGS_nmsg + n * FLAGS_msize;
        while (ep->uccl_send_async((UcclFlow*)conn_ids[f].context, mhandles[f],
                                   send_data, FLAGS_msize,
                                   &ureq_vec[f][r][n]) &&
               !quit) {
        }
        ureq_vec[f][r][n].rtt_tsc = rdtsc();
        tx_cur_sec_bytes += FLAGS_msize;
      }
      c_itr--;
    }
  }

  int fin_msg = 0;

  while (c_itr && !quit) {
    for (int f = 0; f < FLAGS_nflow; f++) {
      for (int r = 0; r < FLAGS_nreq; r++) {
        for (int n = 0; n < FLAGS_nmsg; n++) {
          if (ureq_vec[f][r][n].rtt_tsc == 0 ||
              ep->uccl_poll_ureq_once(&ureq_vec[f][r][n])) {
            ureq_vec[f][r][n].rtt_tsc = 0;

            void* send_data = reinterpret_cast<char*>(datas[f]) +
                              r * FLAGS_msize * FLAGS_nmsg + n * FLAGS_msize;
            if (ep->uccl_send_async((UcclFlow*)conn_ids[f].context, mhandles[f],
                                    send_data, FLAGS_msize,
                                    &ureq_vec[f][r][n])) {
              continue;
            }
            ureq_vec[f][r][n].rtt_tsc = rdtsc();
            tx_cur_sec_bytes += FLAGS_msize;
            if (++fin_msg == FLAGS_nreq) {
              c_itr--;
              fin_msg = 0;
            }
          }
          if (quit) {
            c_itr = 0;
            break;
          }
        }
      }
    }
  }
}

static void server_worker(void) {
  std::string remote_ip;

  std::vector<ConnID> conn_ids;
  std::vector<void*> datas;
  std::vector<struct Mhandle*> mhandles;

  mhandles.resize(FLAGS_nflow);

  for (int i = 0; i < FLAGS_nflow; i++) {
    int remote_dev;
    int remote_gpuidx;
    auto conn_id =
        ep->test_uccl_accept(0, 0, remote_ip, &remote_dev, &remote_gpuidx);
    printf("Server accepted connection from %s (flow#%d)\n", remote_ip.c_str(),
           i);
#ifdef GPU
    void* data;
#ifndef __HIP_PLATFORM_AMD__
    cudaSetDevice(GPU);
    cudaMalloc(&data, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg);
#else
    CHECK(hipSetDevice(GPU) == hipSuccess);
    CHECK(hipMalloc(&data, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg) ==
          hipSuccess);
#endif
#else
    void* data =
        mmap(nullptr, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg,
             PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
    assert(data != MAP_FAILED);
    ep->uccl_regmr((UcclFlow*)conn_id.context, data,
                   FLAGS_msize * FLAGS_nreq * FLAGS_nmsg, 0, &mhandles[i]);

    conn_ids.push_back(conn_id);
    datas.push_back(data);
  }

  if (FLAGS_perftype == "basic") {
    server_basic(conn_ids[0], mhandles[0], datas[0]);
  } else if (FLAGS_perftype == "lat") {
    server_lat(conn_ids[0], mhandles[0], datas[0]);
  } else if (FLAGS_perftype == "tpt") {
    server_tpt(conn_ids, mhandles, datas);
  } else {
    std::cerr << "Unknown performance type: " << FLAGS_perftype << std::endl;
  }

  while (!quit) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  for (int i = 0; i < FLAGS_nflow; i++) {
    ep->uccl_deregmr(mhandles[i]);
    munmap(datas[i], FLAGS_msize * FLAGS_nreq * FLAGS_nmsg);
  }
}

static void client_worker(void) {
  std::vector<ConnID> conn_ids;
  std::vector<void*> datas;
  std::vector<struct Mhandle*> mhandles;

  mhandles.resize(FLAGS_nflow);

  for (int i = 0; i < FLAGS_nflow; i++) {
    std::string dataplane_remote_ip = ip_to_str(remote_p2p_listen.ip);
    int dataplane_remote_port = remote_p2p_listen.port;
    printf("Client connecting to %s:%d (flow#%d)\n",
           dataplane_remote_ip.c_str(), dataplane_remote_port, i);
    auto conn_id = ep->test_uccl_connect(0, 0, 0, 0, dataplane_remote_ip,
                                         dataplane_remote_port);
    printf("Client connected to %s:%d (flow#%d)\n", dataplane_remote_ip.c_str(),
           dataplane_remote_port, i);
#ifdef GPU
    void* data;
#ifndef __HIP_PLATFORM_AMD__
    cudaSetDevice(GPU);
    cudaMalloc(&data, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg);
#else
    CHECK(hipSetDevice(GPU) == hipSuccess);
    CHECK(hipMalloc(&data, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg) ==
          hipSuccess);
#endif
#else
    void* data =
        mmap(nullptr, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg,
             PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
    assert(data != MAP_FAILED);
    ep->uccl_regmr((UcclFlow*)conn_id.context, data,
                   FLAGS_msize * FLAGS_nreq * FLAGS_nmsg, 0, &mhandles[i]);

    conn_ids.push_back(conn_id);
    datas.push_back(data);
  }

  if (FLAGS_perftype == "basic") {
    client_basic(conn_ids[0], mhandles[0], datas[0]);
  } else if (FLAGS_perftype == "lat") {
    client_lat(conn_ids[0], mhandles[0], datas[0]);
  } else if (FLAGS_perftype == "tpt") {
    client_tpt(conn_ids, mhandles, datas);
  } else {
    std::cerr << "Unknown performance type: " << FLAGS_perftype << std::endl;
  }

  for (int i = 0; i < FLAGS_nflow; i++) {
    ep->uccl_deregmr(mhandles[i]);
    munmap(datas[i], FLAGS_msize * FLAGS_nreq * FLAGS_nmsg);
  }
}

// clang-format off
// TO run on AMD cluster:
// export NCCL_IB_HCA="rdma0:1,rdma2:1,rdma3:1,rdma4:1"
// export HIP_VISIBLE_DEVICES=1,2,0,5
// LD_LIBRARY_PATH="${CONDA_LIB_HOME}:/opt/rocm-6.3.1/lib:${LD_LIBRARY_PATH}" ./transport_test --server=true
// LD_LIBRARY_PATH="${CONDA_LIB_HOME}:/opt/rocm-6.3.1/lib:${LD_LIBRARY_PATH}" ./transport_test --serverip=10.42.19.1
// clang-format on

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ep.emplace(ucclParamNUM_ENGINES());
  ep->initialize_engine_by_dev(0, true);
  local_p2p_listen.port = ep->get_p2p_listen_port(0);
  local_p2p_listen.ip = str_to_ip(ep->get_p2p_listen_ip(0));

  // Create a thread to print throughput every second
  std::thread t([&] {
    while (!quit) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      printf("(%d flows) TX Tpt: %.2f Gbps(%lu), RX Tpt: %.2f Gbps(%lu)\n",
             FLAGS_nflow,
             (tx_cur_sec_bytes - tx_prev_sec_bytes) * 8.0 / 1000 / 1000 / 1000,
             c_itr,
             (rx_cur_sec_bytes - rx_prev_sec_bytes) * 8.0 / 1000 / 1000 / 1000,
             s_itr);

      tx_prev_sec_bytes = tx_cur_sec_bytes;
      rx_prev_sec_bytes = rx_cur_sec_bytes;
    }
  });

  if (FLAGS_bi) {
    CHECK(!FLAGS_serverip.empty()) << "Server IP address is required.";

    std::thread listen_thread([&] {
      listen_accept_exchange(FLAGS_oobport, &local_p2p_listen,
                             sizeof(local_p2p_listen), &remote_p2p_listen,
                             sizeof(remote_p2p_listen));
    });
    std::thread connect_thread([&] {
      connect_exchange(FLAGS_oobport, FLAGS_serverip, &local_p2p_listen,
                       sizeof(local_p2p_listen), &remote_p2p_listen,
                       sizeof(remote_p2p_listen));
    });
    listen_thread.join();
    connect_thread.join();

    std::thread server_thread(server_worker);
    std::thread client_thread(client_worker);
    server_thread.join();
    client_thread.join();
  } else if (FLAGS_server) {
    listen_accept_exchange(FLAGS_oobport, &local_p2p_listen,
                           sizeof(local_p2p_listen), &remote_p2p_listen,
                           sizeof(remote_p2p_listen));
    server_worker();
  } else {
    CHECK(!FLAGS_serverip.empty()) << "Server IP address is required.";
    connect_exchange(FLAGS_oobport, FLAGS_serverip, &local_p2p_listen,
                     sizeof(local_p2p_listen), &remote_p2p_listen,
                     sizeof(remote_p2p_listen));
    client_worker();
  }

  t.join();

  return 0;
}