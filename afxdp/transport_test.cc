#include "transport.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <signal.h>

#include <chrono>
#include <thread>

#include "transport_config.h"

using namespace uccl;

const size_t kTestMsgSize = 1024000;
const size_t kTestIters = 1024000000;
size_t kReportIters = 1000;
const size_t kMaxInflight = 8;

DEFINE_bool(client, false, "Whether this is a client sending traffic.");
DEFINE_bool(verify, false, "Whether to check data correctness.");
DEFINE_bool(rand_len, false, "Whether to use randomized data length.");
DEFINE_string(test, "basic",
              "Which test to run: basic, async, pingpong, mt (multi-thread), "
              "mc (multi-connection).");

enum TestType { kBasic, kAsync, kPingpong, kMt, kMc };

volatile bool quit = false;

void interrupt_handler(int signal) {
    (void)signal;
    quit = true;
    AFXDPFactory::shutdown();
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, interrupt_handler);
    signal(SIGHUP, interrupt_handler);
    // signal(SIGALRM, interrupt_handler);
    // alarm(10);

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
    } else {
        LOG(FATAL) << "Unknown test type: " << FLAGS_test;
    }

    if (FLAGS_client) {
        auto ep = Endpoint(DEV_DEFAULT, QID_DEFAULT, NUM_FRAMES, ENGINE_CPUID);
        pin_thread_to_cpu(ENGINE_CPUID + 1);
        auto conn_id = ep.uccl_connect(server_ip_str);
        FlowID conn_id2;
        if (test_type == kMc) conn_id2 = ep.uccl_connect(server_ip_str);

        size_t send_len = kTestMsgSize, recv_len = kTestMsgSize;
        auto* data = new uint8_t[kTestMsgSize];
        auto* data_u64 = reinterpret_cast<uint64_t*>(data);
        auto* data2 = new uint8_t[kTestMsgSize];

        size_t sent_bytes = 0;
        std::vector<uint64_t> rtts;
        auto start_bw = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < kTestIters; i++) {
            send_len = kTestMsgSize;
            if (FLAGS_rand_len)
                send_len = IntRand(1, kTestMsgSize - 1024) + 1024;

            if (FLAGS_verify) {
                for (int j = 0; j < send_len / sizeof(uint64_t); j++) {
                    data_u64[j] = (uint64_t)(i + 1) * (uint64_t)j;
                }
            }

            auto start = std::chrono::high_resolution_clock::now();
            switch (test_type) {
                case kBasic:
                    ep.uccl_send(conn_id, data, send_len);
                    sent_bytes += send_len;
                    break;
                case kAsync: {
                    std::vector<PollCtx*> poll_ctxs;
                    size_t step_size = send_len / kMaxInflight + 1;
                    for (int j = 0; j < kMaxInflight; j++) {
                        auto iter_len =
                            std::min(step_size, send_len - j * step_size);
                        auto* iter_data = data + j * step_size;

                        PollCtx* poll_ctx;
                        poll_ctx =
                            ep.uccl_send_async(conn_id, iter_data, iter_len);
                        poll_ctxs.push_back(poll_ctx);
                    }
                    for (auto poll_ctx : poll_ctxs) {
                        ep.uccl_poll(poll_ctx);
                    }
                    sent_bytes += send_len;
                    break;
                }
                case kPingpong: {
                    PollCtx *poll_ctx1, *poll_ctx2;
                    poll_ctx1 = ep.uccl_send_async(conn_id, data, send_len);
                    poll_ctx2 = ep.uccl_recv_async(conn_id, data2, &recv_len);
                    ep.uccl_poll(poll_ctx1);
                    ep.uccl_poll(poll_ctx2);
                    sent_bytes += send_len;
                    break;
                }
                case kMt: {
                    std::thread t1([&ep, conn_id, data, send_len]() {
                        PollCtx* poll_ctx =
                            ep.uccl_send_async(conn_id, data, send_len);
                        ep.uccl_poll(poll_ctx);
                    });
                    std::thread t2([&ep, conn_id, data2, &recv_len]() {
                        PollCtx* poll_ctx =
                            ep.uccl_recv_async(conn_id, data2, &recv_len);
                        ep.uccl_poll(poll_ctx);
                    });
                    t1.join();
                    t2.join();
                    sent_bytes += send_len;
                    break;
                }
                case kMc: {
                    PollCtx *poll_ctx1, *poll_ctx2;
                    poll_ctx1 = ep.uccl_send_async(conn_id, data, send_len);
                    poll_ctx2 = ep.uccl_send_async(conn_id2, data2, send_len);
                    ep.uccl_poll(poll_ctx1);
                    ep.uccl_poll(poll_ctx2);
                    sent_bytes += send_len * 2;
                    break;
                }
                default:
                    break;
            }

            auto duration_us =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);

            rtts.push_back(duration_us.count());
            if (i % kReportIters == 0 && i != 0) {
                uint64_t med_latency, tail_latency;
                med_latency = Percentile(rtts, 50);
                tail_latency = Percentile(rtts, 99);
                auto end_bw = std::chrono::high_resolution_clock::now();
                // 24B: 4B FCS + 8B frame delimiter + 12B interframe gap
                auto bw_gbps =
                    sent_bytes *
                    (AFXDP_MTU * 1.0 /
                     (AFXDP_MTU - kNetHdrLen - kUcclHdrLen - 24)) *
                    8.0 / 1024 / 1024 / 1024 /
                    (std::chrono::duration_cast<std::chrono::microseconds>(
                         end_bw - start_bw)
                         .count() *
                     1e-6);
                sent_bytes = 0;
                start_bw = end_bw;

                LOG(INFO) << "Sent " << i
                          << " messages, med rtt: " << med_latency
                          << " us, tail rtt: " << tail_latency << " us, bw "
                          << bw_gbps << " Gbps";
            }
        }
    } else {
        auto ep = Endpoint(DEV_DEFAULT, QID_DEFAULT, NUM_FRAMES, ENGINE_CPUID);
        pin_thread_to_cpu(ENGINE_CPUID + 1);
        auto [conn_id, client_ip_str] = ep.uccl_accept();
        FlowID conn_id2;
        std::string client_ip_str2;
        if (test_type == kMc)
            std::tie(conn_id2, client_ip_str2) = ep.uccl_accept();

        size_t send_len = kTestMsgSize, recv_len = kTestMsgSize;
        auto* data = new uint8_t[kTestMsgSize];
        auto* data_u64 = reinterpret_cast<uint64_t*>(data);
        auto* data2 = new uint8_t[kTestMsgSize];

        for (int i = 0; i < kTestIters; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            switch (test_type) {
                case kBasic:
                    ep.uccl_recv(conn_id, data, &send_len);
                    break;
                case kAsync: {
                    size_t step_size = send_len / kMaxInflight + 1;
                    std::vector<PollCtx*> poll_ctxs;
                    for (int j = 0; j < kMaxInflight; j++) {
                        auto iter_len =
                            std::min(step_size, send_len - j * step_size);
                        auto* iter_data = data + j * step_size;

                        PollCtx* poll_ctx;
                        poll_ctx =
                            ep.uccl_recv_async(conn_id, iter_data, &recv_len);
                        poll_ctxs.push_back(poll_ctx);
                    }
                    for (auto poll_ctx : poll_ctxs) {
                        ep.uccl_poll(poll_ctx);
                    }
                    break;
                }
                case kPingpong: {
                    PollCtx *poll_ctx1, *poll_ctx2;
                    poll_ctx1 = ep.uccl_recv_async(conn_id, data, &recv_len);
                    poll_ctx2 = ep.uccl_send_async(conn_id, data2, send_len);
                    ep.uccl_poll(poll_ctx1);
                    ep.uccl_poll(poll_ctx2);
                    break;
                }
                case kMt: {
                    std::thread t1([&ep, conn_id, data, &recv_len]() {
                        PollCtx* poll_ctx =
                            ep.uccl_recv_async(conn_id, data, &recv_len);
                        ep.uccl_poll(poll_ctx);
                    });
                    std::thread t2([&ep, conn_id, data2, send_len]() {
                        PollCtx* poll_ctx =
                            ep.uccl_send_async(conn_id, data2, send_len);
                        ep.uccl_poll(poll_ctx);
                    });
                    t1.join();
                    t2.join();
                    break;
                }
                case kMc: {
                    PollCtx *poll_ctx1, *poll_ctx2;
                    poll_ctx1 = ep.uccl_recv_async(conn_id, data, &recv_len);
                    poll_ctx2 = ep.uccl_recv_async(conn_id2, data2, &recv_len);
                    ep.uccl_poll(poll_ctx1);
                    ep.uccl_poll(poll_ctx2);
                    break;
                }
                default:
                    break;
            }
            auto duration_us =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);

            if (FLAGS_verify) {
                CHECK_LE(recv_len, kTestMsgSize)
                    << "Received message size mismatches";
                for (int j = 0; j < recv_len / sizeof(uint64_t); j++) {
                    CHECK_EQ(data_u64[j], (uint64_t)(i + 1) * (uint64_t)j)
                        << "Data mismatch at index " << j << " at iter "
                        << i + 1;
                }
                memset(data, 0, recv_len);
            }

            LOG_EVERY_N(INFO, kReportIters)
                << "Received " << i << " messages, rtt " << duration_us.count()
                << " us";
        }
    }

    return 0;
}