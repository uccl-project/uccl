#include "transport.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <signal.h>

#include <chrono>
#include <thread>

#include "transport_config.h"

using namespace uccl;

const size_t NUM_FRAMES = 4096 * 64;  // 1GB frame pool
const size_t QUEUE_ID = 0;
const size_t kTestMsgSize = 1024000;
const size_t kTestIters = 1024000000;
const size_t kReportIters = 1000;

DEFINE_bool(client, false, "Whether this is a client sending traffic.");
DEFINE_string(test, "fixed", "Which test to run: fixed, random, async.");

enum TestType { kFixed, kRandom, kAsync };

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
    if (FLAGS_test == "fixed") {
        test_type = kFixed;
    } else if (FLAGS_test == "random") {
        test_type = kRandom;
    } else if (FLAGS_test == "async") {
        test_type = kAsync;
    } else {
        LOG(FATAL) << "Unknown test type: " << FLAGS_test;
    }

    Channel channel;

    if (FLAGS_client) {
        AFXDPFactory::init(interface_name, "ebpf_transport.o",
                           "ebpf_transport");
        UcclEngine engine(QUEUE_ID, NUM_FRAMES, &channel, client_ip_str,
                          client_port, server_ip_str, server_port,
                          client_mac_str, server_mac_str);
        auto engine_th = std::thread([&engine]() {
            pin_thread_to_cpu(2);
            engine.run();
        });

        pin_thread_to_cpu(3);
        auto ep = Endpoint(&channel);
        auto conn_id = ep.uccl_connect(server_ip_str);

        size_t send_len = kTestMsgSize, recv_len = kTestMsgSize;
        auto* data = new uint8_t[kTestMsgSize];
        auto* data_u32 = reinterpret_cast<uint32_t*>(data);
        for (int j = 0; j < kTestMsgSize / sizeof(uint32_t); j++) {
            data_u32[j] = j;
        }

        size_t sent_bytes = 0;
        std::vector<uint64_t> rtts;
        auto start_bw = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < kTestIters; i++) {
            if (test_type == kFixed || test_type == kAsync) {
                send_len = kTestMsgSize;
            } else if (test_type == kRandom) {
                send_len = IntRand(1, kTestMsgSize);
            }

            auto start = std::chrono::high_resolution_clock::now();
            if (test_type == kAsync) {
                ep.uccl_send_async(conn_id, data, send_len);
                ep.uccl_recv_async(conn_id, data, &recv_len);
                ep.uccl_send_poll();
                ep.uccl_recv_poll();
            } else {
                ep.uccl_send(conn_id, data, send_len);
            }
            auto duration_us =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);

            sent_bytes += send_len;

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

        engine.shutdown();
        engine_th.join();
    } else {
        AFXDPFactory::init(interface_name, "ebpf_transport.o",
                           "ebpf_transport");
        UcclEngine engine(QUEUE_ID, NUM_FRAMES, &channel, server_ip_str,
                          server_port, client_ip_str, client_port,
                          server_mac_str, client_mac_str);
        auto engine_th = std::thread([&engine]() {
            pin_thread_to_cpu(2);
            engine.run();
        });

        pin_thread_to_cpu(3);
        auto ep = Endpoint(&channel);
        auto [conn_id, client_ip_str] = ep.uccl_accept();

        size_t send_len = kTestMsgSize, recv_len = kTestMsgSize;
        auto* data = new uint8_t[kTestMsgSize];

        for (int i = 0; i < kTestIters; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            if (test_type == kAsync) {
                ep.uccl_recv_async(conn_id, data, &recv_len);
                ep.uccl_send_async(conn_id, data, send_len);
                ep.uccl_recv_poll();
                ep.uccl_send_poll();
            } else {
                ep.uccl_recv(conn_id, data, &send_len);
            }
            auto duration_us =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            /*
            CHECK_LE(recv_len, kTestMsgSize)
                << "Received message size mismatches";
            for (int j = 0; j < kTestMsgSize / sizeof(uint32_t); j++) {
                CHECK_EQ(reinterpret_cast<uint32_t*>(data)[j], j)
                    << "Data mismatch at index " << j;
            }
            memset(data, 0, kTestMsgSize);
            */
            LOG_EVERY_N(INFO, kReportIters)
                << "Received " << i << " messages, rtt " << duration_us.count()
                << " us";
        }
        engine.shutdown();
        engine_th.join();
    }

    return 0;
}