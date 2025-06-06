#include "transport.h"
#include "transport_config.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <deque>
#include <thread>
#include <signal.h>

using namespace uccl;

const size_t kTestIters = 1024000000000UL;
const std::chrono::duration kReportIntervalSec = std::chrono::seconds(2);
const size_t kReportIters = 5000;
const uint32_t kNumConns = 4;

size_t kTestMsgSize = 1024000;
size_t kMaxInflight = 8;

DEFINE_uint64(size, 1024000, "Size of test message.");
DEFINE_uint64(infly, 8, "Max num of test messages in the flight.");
DEFINE_string(serverip, "", "Server IP address the client tries to connect.");
DEFINE_string(clientip, "", "Client IP address the server tries to connect.");
DEFINE_bool(verify, false, "Whether to check data correctness.");
DEFINE_bool(rand, false, "Whether to use randomized data length.");
DEFINE_string(
    test, "basic",
    "Which test to run: basic, async, pingpong, mt (multi-thread), "
    "mc (multi-connection), mq (multi-queue), bimq (bi-directional mq), tput.");

enum TestType { kBasic, kAsync, kPingpong, kMt, kMc, kMq, kBiMq, kTput };

uint64_t* get_host_ptr(uint64_t* dev_ptr, size_t size) {
  uint64_t* host_ptr = (uint64_t*)malloc(size);
  cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost);
  return host_ptr;
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  kTestMsgSize = FLAGS_size;
  kMaxInflight = FLAGS_infly;

  bool is_client;
  if (!FLAGS_serverip.empty()) {
    is_client = true;
  } else if (!FLAGS_clientip.empty()) {
    is_client = false;
  } else {
    LOG(FATAL)
        << "Please specify server IP or client IP, and only one of them.";
  }

  TestType test_type;
  if (FLAGS_test == "basic") {
    test_type = kBasic;
  } else if (FLAGS_test == "async") {
    test_type = kAsync;
  } else if (FLAGS_test == "pingpong") {
    test_type = kPingpong;
  } else if (FLAGS_test == "mt") {
    test_type = kMt;
  } else if (FLAGS_test == "mc") {
    test_type = kMc;
  } else if (FLAGS_test == "mq") {
    test_type = kMq;
  } else if (FLAGS_test == "bimq") {
    test_type = kBiMq;
  } else if (FLAGS_test == "tput") {
    test_type = kTput;
  } else {
    LOG(FATAL) << "Unknown test type: " << FLAGS_test;
  }

  std::mt19937 generator(42);
  std::uniform_int_distribution<int> distribution(1024, kTestMsgSize);
  srand(42);
  pin_thread_to_cpu(0);

  if (is_client) {
    auto ep = Endpoint(0);
    DCHECK(FLAGS_serverip != "");
    int const kMaxArraySize = std::max(kNumConns, kNumVdevices);
    ConnID conn_id, conn_id2;
    ConnID conn_id_vec[kMaxArraySize];
    int remote_vdevs[kMaxArraySize];
    std::string remote_ip[kMaxArraySize];

    for (int i = 0; i < kMaxArraySize; i++) ep.uccl_listen();

    conn_id = ep.uccl_connect(0, 0, FLAGS_serverip, ep.listen_port_vec_[0]);
    if (test_type == kMc) {
      conn_id_vec[0] = conn_id;
      for (int i = 1; i < kNumConns; i++)
        conn_id_vec[i] =
            ep.uccl_connect(0, 0, FLAGS_serverip, ep.listen_port_vec_[0]);
    } else if (test_type == kMq) {
      conn_id_vec[0] = conn_id;
      for (int i = 1; i < kNumVdevices; i++)
        conn_id_vec[i] =
            ep.uccl_connect(i, i, FLAGS_serverip, ep.listen_port_vec_[i]);
    } else if (test_type == kBiMq) {
      conn_id_vec[0] = conn_id;
      for (int i = 1; i < kNumVdevices; i++) {
        if (i % 2 == 0)
          conn_id_vec[i] =
              ep.uccl_connect(i, i, FLAGS_serverip, ep.listen_port_vec_[i]);
        else
          conn_id_vec[i] = ep.uccl_accept(i, &remote_vdevs[i], remote_ip[i],
                                          ep.listen_fd_vec_[i]);
      }
    }

    int send_len = kTestMsgSize, recv_len = kTestMsgSize;
    uint8_t *data[kNumVdevices], *data2[kNumVdevices];
    Mhandle mh[kNumVdevices], mh2[kNumVdevices];

    for (int i = 0; i < kNumVdevices; i++) {
      auto gpu_idx = i;
      auto dev_idx = get_dev_idx_by_gpu_idx(i);

      cudaSetDevice(gpu_idx);
      auto* dev = EFAFactory::GetEFADevice(dev_idx);

      cudaMalloc(&data[i], kTestMsgSize);
      mh[i].mr =
          ibv_reg_mr(dev->pd, data[i], kTestMsgSize, IBV_ACCESS_LOCAL_WRITE);
      cudaMalloc(&data2[i], kTestMsgSize);
      mh2[i].mr =
          ibv_reg_mr(dev->pd, data2[i], kTestMsgSize, IBV_ACCESS_LOCAL_WRITE);
    }
    cudaSetDevice(0);

    uint64_t* data_u64;
    data_u64 = reinterpret_cast<uint64_t*>(data[0]);

    size_t sent_bytes = 0;
    std::vector<uint64_t> rtts;
    auto start_bw_mea = std::chrono::high_resolution_clock::now();

    std::deque<PollCtx*> poll_ctxs;
    PollCtx* last_ctx = nullptr;
    uint32_t inflight_msgs[kNumVdevices] = {0};

    for (size_t i = 0; i < kTestIters;) {
      send_len = kTestMsgSize;
      if (FLAGS_rand) send_len = distribution(generator);

      if (FLAGS_verify) {
        auto host_data_u64 = get_host_ptr(data_u64, send_len);
        for (int j = 0; j < send_len / sizeof(uint64_t); j++) {
          host_data_u64[j] = (uint64_t)i * (uint64_t)j;
        }
        cudaMemcpy(data_u64, host_data_u64, send_len, cudaMemcpyHostToDevice);
      }

      switch (test_type) {
        case kBasic: {
          TscTimer timer;
          timer.start();
          ep.uccl_send(conn_id, data[0], send_len, &mh[0],
                       /*busypoll=*/true);
          timer.stop();
          rtts.push_back(timer.avg_usec(freq_ghz));
          sent_bytes += send_len;
          i++;
          break;
        }
        case kAsync: {
          std::vector<PollCtx*> poll_ctxs;
          size_t step_size = send_len / kMaxInflight + 1;
          for (int j = 0; j < kMaxInflight; j++) {
            auto iter_len = std::min(step_size, send_len - j * step_size);
            auto* iter_data = data[0] + j * step_size;

            PollCtx* poll_ctx;
            poll_ctx = ep.uccl_send_async(conn_id, iter_data, iter_len, &mh[0]);
            poll_ctx->timestamp = rdtsc();
            poll_ctxs.push_back(poll_ctx);
          }
          for (auto poll_ctx : poll_ctxs) {
            auto async_start = poll_ctx->timestamp;
            // after a success poll, poll_ctx is freed
            ep.uccl_poll(poll_ctx);
            rtts.push_back(to_usec(rdtsc() - async_start, freq_ghz));
          }
          sent_bytes += send_len;
          i++;
          break;
        }
        case kPingpong: {
          PollCtx *poll_ctx1, *poll_ctx2;
          TscTimer timer;
          timer.start();
          poll_ctx1 = ep.uccl_send_async(conn_id, data[0], send_len, &mh[0]);
          poll_ctx2 = ep.uccl_recv_async(conn_id, data2[0], &recv_len, &mh2[0]);
          ep.uccl_poll(poll_ctx1);
          ep.uccl_poll(poll_ctx2);
          timer.stop();
          rtts.push_back(timer.avg_usec(freq_ghz));
          sent_bytes += send_len * 2;
          i += 1;
          break;
        }
        case kMt: {
          TscTimer timer;
          timer.start();
          std::thread t1([&ep, conn_id, data, send_len, mh]() mutable {
            PollCtx* poll_ctx =
                ep.uccl_send_async(conn_id, data[0], send_len, &mh[0]);
            ep.uccl_poll(poll_ctx);
          });
          std::thread t2([&ep, conn_id, data2, &recv_len, mh2]() mutable {
            PollCtx* poll_ctx =
                ep.uccl_recv_async(conn_id, data2[0], &recv_len, &mh2[0]);
            ep.uccl_poll(poll_ctx);
          });
          t1.join();
          t2.join();
          timer.stop();
          rtts.push_back(timer.avg_usec(freq_ghz));
          sent_bytes += send_len * 2;
          i += 1;
          break;
        }
        case kMc: {
          TscTimer timer;
          timer.start();
          for (int j = 0; j < kNumConns; j++) {
            auto* poll_ctx =
                ep.uccl_send_async(conn_id_vec[j], data[0], send_len, &mh[0]);
            poll_ctxs.push_back(poll_ctx);
          }
          while (!poll_ctxs.empty()) {
            auto* poll_ctx = poll_ctxs.front();
            ep.uccl_poll(poll_ctx);
            poll_ctxs.pop_front();
          }
          timer.stop();
          rtts.push_back(timer.avg_usec(freq_ghz));
          sent_bytes += send_len * 2;
          i += 1;
          break;
        }
        case kMq: {
          for (int j = 0; j < kNumVdevices; j++) {
            while (inflight_msgs[j] < kMaxInflight) {
              auto& __conn_id = conn_id_vec[j];
              auto poll_ctx =
                  ep.uccl_send_async(__conn_id, data[j], send_len, &mh[j]);
              poll_ctx->timestamp = rdtsc();
              poll_ctxs.push_back(poll_ctx);
              inflight_msgs[j]++;
            }
          }
          auto inflights = poll_ctxs.size();
          for (int j = 0; j < inflights; j++) {
            auto poll_ctx = poll_ctxs.front();
            poll_ctxs.pop_front();
            auto async_start = poll_ctx->timestamp;
            auto vdev_idx = poll_ctx->engine_idx / kNumEnginesPerVdev;
            if (ep.uccl_poll_once(poll_ctx)) {
              rtts.push_back(to_usec(rdtsc() - async_start, freq_ghz));
              sent_bytes += send_len;
              i++;
              inflight_msgs[vdev_idx]--;
            } else {
              poll_ctxs.push_back(poll_ctx);
            }
          }
          break;
        }
        case kBiMq: {
          for (int j = 0; j < kNumVdevices; j++) {
            while (inflight_msgs[j] < kMaxInflight) {
              auto& __conn_id = conn_id_vec[j];
              auto* poll_ctx =
                  (j % 2 == 0)
                      ? ep.uccl_send_async(__conn_id, data[j], send_len, &mh[j])
                      : ep.uccl_recv_multi_async(__conn_id, (void**)&(data[j]),
                                                 &recv_len, (Mhandle**)&mh[j],
                                                 1);
              poll_ctx->timestamp = rdtsc();
              poll_ctxs.push_back(poll_ctx);
              inflight_msgs[j]++;
            }
          }
          auto inflights = poll_ctxs.size();
          for (int j = 0; j < inflights; j++) {
            auto poll_ctx = poll_ctxs.front();
            auto async_start = poll_ctx->timestamp;
            auto vdev_idx = poll_ctx->engine_idx / kNumEnginesPerVdev;
            poll_ctxs.pop_front();
            if (ep.uccl_poll_once(poll_ctx)) {
              rtts.push_back(to_usec(rdtsc() - async_start, freq_ghz));
              sent_bytes += send_len;
              i++;
              inflight_msgs[vdev_idx]--;
            } else {
              poll_ctxs.push_back(poll_ctx);
            }
          }
          CHECK(send_len == recv_len)
              << "send_len: " << send_len << ", recv_len: " << recv_len;
          break;
        }
        case kTput: {
          auto* poll_ctx =
              ep.uccl_send_async(conn_id, data[0], send_len, &mh[0]);
          poll_ctx->timestamp = rdtsc();
          if (last_ctx) {
            auto async_start = last_ctx->timestamp;
            ep.uccl_poll(last_ctx);
            rtts.push_back(to_usec(rdtsc() - async_start, freq_ghz));
            sent_bytes += send_len;
            i++;
          }
          last_ctx = poll_ctx;
          break;
        }
        default:
          break;
      }

      if ((i + 1) % kReportIters == 0) {
        auto end_bw_mea = std::chrono::high_resolution_clock::now();

        auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(
            end_bw_mea - start_bw_mea);

        if (duration_sec < kReportIntervalSec) continue;

        // Clear to avoid Percentile() taking too much time.
        if (rtts.size() > 100000) {
          rtts.assign(rtts.end() - 100000, rtts.end());
        }

        auto duaration_usec =
            std::chrono::duration_cast<std::chrono::microseconds>(end_bw_mea -
                                                                  start_bw_mea)
                .count();

        uint64_t med_latency, tail_latency;
        med_latency = Percentile(rtts, 50);
        tail_latency = Percentile(rtts, 99);

        // 24B: 4B FCS + 8B frame delimiter + 12B interframe gap
        auto bw_gbps = sent_bytes *
                       ((EFA_MTU * 1.0 + 24) / (EFA_MTU - kUcclPktHdrLen)) *
                       8.0 / 1000 / 1000 / 1000 / (duaration_usec * 1e-6);
        auto app_bw_gbps =
            sent_bytes * 8.0 / 1000 / 1000 / 1000 / (duaration_usec * 1e-6);
        sent_bytes = 0;

        LOG(INFO) << "Sent " << i + 1 << " messages, med rtt: " << med_latency
                  << " us, tail rtt: " << tail_latency << " us, link bw "
                  << bw_gbps << " Gbps, app bw " << app_bw_gbps << " Gbps";
        start_bw_mea = std::chrono::high_resolution_clock::now();
      }
    }
  } else {
    auto ep = Endpoint(0);
    int const kMaxArraySize = std::max(kNumConns, kNumVdevices);
    ConnID conn_id, conn_id2;
    ConnID conn_id_vec[kMaxArraySize];
    std::string remote_ip[kMaxArraySize];
    int remote_vdevs[kMaxArraySize];

    for (int i = 0; i < kMaxArraySize; i++) ep.uccl_listen();

    conn_id =
        ep.uccl_accept(0, &remote_vdevs[0], remote_ip[0], ep.listen_fd_vec_[0]);
    if (test_type == kMc) {
      conn_id_vec[0] = conn_id;
      for (int i = 1; i < kNumConns; i++) {
        conn_id_vec[i] = ep.uccl_accept(0, &remote_vdevs[i], remote_ip[i],
                                        ep.listen_fd_vec_[0]);
      }
    } else if (test_type == kMq) {
      conn_id_vec[0] = conn_id;
      for (int i = 1; i < kNumVdevices; i++) {
        conn_id_vec[i] = ep.uccl_accept(i, &remote_vdevs[i], remote_ip[i],
                                        ep.listen_fd_vec_[i]);
      }
    } else if (test_type == kBiMq) {
      conn_id_vec[0] = conn_id;
      for (int i = 1; i < kNumVdevices; i++) {
        if (i % 2 == 0)
          conn_id_vec[i] = ep.uccl_accept(i, &remote_vdevs[i], remote_ip[i],
                                          ep.listen_fd_vec_[i]);
        else
          conn_id_vec[i] =
              ep.uccl_connect(i, i, FLAGS_clientip, ep.listen_port_vec_[i]);
      }
    }

    int send_len = kTestMsgSize, recv_len = kTestMsgSize;
    uint8_t *data[kNumVdevices], *data2[kNumVdevices];
    Mhandle mh[kNumVdevices], mh2[kNumVdevices];

    for (int i = 0; i < kNumVdevices; i++) {
      auto gpu_idx = i;
      auto dev_idx = get_dev_idx_by_gpu_idx(i);

      cudaSetDevice(gpu_idx);
      auto* dev = EFAFactory::GetEFADevice(dev_idx);

      cudaMalloc(&data[i], kTestMsgSize);
      mh[i].mr =
          ibv_reg_mr(dev->pd, data[i], kTestMsgSize, IBV_ACCESS_LOCAL_WRITE);
      cudaMalloc(&data2[i], kTestMsgSize);
      mh2[i].mr =
          ibv_reg_mr(dev->pd, data2[i], kTestMsgSize, IBV_ACCESS_LOCAL_WRITE);
    }
    cudaSetDevice(0);

    uint64_t* data_u64;
    data_u64 = reinterpret_cast<uint64_t*>(data[0]);
    auto start = std::chrono::high_resolution_clock::now();

    std::deque<PollCtx*> poll_ctxs;
    PollCtx* last_ctx = nullptr;
    uint32_t inflight_msgs[kNumVdevices] = {0};

    for (size_t i = 0; i < kTestIters;) {
      send_len = kTestMsgSize;
      if (FLAGS_rand) send_len = distribution(generator);

      switch (test_type) {
        case kBasic: {
          ep.uccl_recv(conn_id, data[0], &recv_len, &mh[0],
                       /*busypoll=*/true);
          i++;
          break;
        }
        case kAsync: {
          size_t step_size = send_len / kMaxInflight + 1;
          int recv_lens[kMaxInflight] = {0};
          std::vector<PollCtx*> poll_ctxs;
          for (int j = 0; j < kMaxInflight; j++) {
            auto iter_len = std::min(step_size, send_len - j * step_size);
            auto* iter_data = data[0] + j * step_size;

            PollCtx* poll_ctx;
            poll_ctx =
                ep.uccl_recv_async(conn_id, iter_data, &recv_lens[j], &mh[0]);
            poll_ctxs.push_back(poll_ctx);
          }
          for (auto poll_ctx : poll_ctxs) {
            ep.uccl_poll(poll_ctx);
          }
          recv_len = 0;
          for (auto len : recv_lens) {
            recv_len += len;
          }
          i++;
          break;
        }
        case kPingpong: {
          PollCtx *poll_ctx1, *poll_ctx2;
          poll_ctx1 = ep.uccl_recv_async(conn_id, data[0], &recv_len, &mh[0]);
          poll_ctx2 = ep.uccl_send_async(conn_id, data2[0], send_len, &mh2[0]);
          ep.uccl_poll(poll_ctx1);
          ep.uccl_poll(poll_ctx2);
          i += 1;
          break;
        }
        case kMt: {
          std::thread t1([&ep, conn_id, data, &recv_len, mh]() mutable {
            PollCtx* poll_ctx =
                ep.uccl_recv_async(conn_id, data[0], &recv_len, &mh[0]);
            ep.uccl_poll(poll_ctx);
          });
          std::thread t2([&ep, conn_id, data2, send_len, mh2]() mutable {
            PollCtx* poll_ctx =
                ep.uccl_send_async(conn_id, data2[0], send_len, &mh2[0]);
            ep.uccl_poll(poll_ctx);
          });
          t1.join();
          t2.join();
          i += 1;
          break;
        }
        case kMc: {
          for (int j = 0; j < kNumConns; j++) {
            auto* poll_ctx =
                ep.uccl_recv_async(conn_id_vec[j], data[0], &recv_len, &mh[0]);
            poll_ctxs.push_back(poll_ctx);
          }
          while (!poll_ctxs.empty()) {
            auto* poll_ctx = poll_ctxs.front();
            ep.uccl_poll(poll_ctx);
            poll_ctxs.pop_front();
          }
          i += 1;
          break;
        }
        case kMq: {
          for (int j = 0; j < kNumVdevices; j++) {
            while (inflight_msgs[j] < kMaxInflight) {
              auto& __conn_id = conn_id_vec[j];
              auto poll_ctx =
                  ep.uccl_recv_async(__conn_id, data[j], &recv_len, &mh[j]);
              poll_ctxs.push_back(poll_ctx);
              inflight_msgs[j]++;
            }
          }
          auto inflights = poll_ctxs.size();
          for (int j = 0; j < inflights; j++) {
            auto poll_ctx = poll_ctxs.front();
            poll_ctxs.pop_front();
            auto vdev_idx = poll_ctx->engine_idx / kNumEnginesPerVdev;
            if (ep.uccl_poll_once(poll_ctx)) {
              inflight_msgs[vdev_idx]--;
              i++;
            } else {
              poll_ctxs.push_back(poll_ctx);
            }
          }
          break;
        }
        case kBiMq: {
          for (int j = 0; j < kNumVdevices; j++) {
            while (inflight_msgs[j] < kMaxInflight) {
              auto& __conn_id = conn_id_vec[j];
              auto* poll_ctx = (j % 2 == 0)
                                   ? ep.uccl_recv_async(__conn_id, data[j],
                                                        &recv_len, &mh[j])
                                   : ep.uccl_send_async(__conn_id, data[j],
                                                        send_len, &mh[j]);
              poll_ctxs.push_back(poll_ctx);
              inflight_msgs[j]++;
            }
          }
          auto inflights = poll_ctxs.size();
          for (int j = 0; j < inflights; j++) {
            auto poll_ctx = poll_ctxs.front();
            auto vdev_idx = poll_ctx->engine_idx / kNumEnginesPerVdev;
            poll_ctxs.pop_front();
            if (ep.uccl_poll_once(poll_ctx)) {
              inflight_msgs[vdev_idx]--;
              i++;
            } else {
              poll_ctxs.push_back(poll_ctx);
            }
          }
          CHECK(send_len == recv_len)
              << "send_len: " << send_len << ", recv_len: " << recv_len;
          break;
        }
        case kTput: {
          auto* poll_ctx = ep.uccl_recv_async(conn_id, data, &recv_len, &mh[0]);
          if (last_ctx) ep.uccl_poll(last_ctx);
          last_ctx = poll_ctx;
          i++;
          break;
        }
        default:
          break;
      }

      if ((i + 1) % kReportIters == 0) {
        auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - start);
        if (duration_sec < kReportIntervalSec) continue;

        LOG(INFO) << "Received " << i + 1 << " messages";

        start = std::chrono::high_resolution_clock::now();
      }

      if (FLAGS_verify) {
        auto host_data_u64 = get_host_ptr(data_u64, send_len);

        bool data_mismatch = false;
        auto expected_len = FLAGS_rand ? send_len : kTestMsgSize;
        if (recv_len != expected_len) {
          LOG(ERROR) << "Received message size mismatches, expected "
                     << expected_len << ", received " << recv_len;
          data_mismatch = true;
        }
        for (int j = 0; j < recv_len / sizeof(uint64_t); j++) {
          if (host_data_u64[j] != (uint64_t)i * (uint64_t)j) {
            data_mismatch = true;
            LOG_EVERY_N(ERROR, 1000)
                << "Data mismatch at index " << j * sizeof(uint64_t)
                << ", expected " << (uint64_t)i * (uint64_t)j << ", received "
                << host_data_u64[j];
          }
        }
        CHECK(!data_mismatch) << "Data mismatch at iter " << i;
        memset(host_data_u64, 0, recv_len);
        cudaMemcpy(data_u64, host_data_u64, send_len, cudaMemcpyHostToDevice);
      }
    }
  }

  return 0;
}