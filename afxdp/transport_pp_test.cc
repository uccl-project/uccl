#include <gflags/gflags.h>
#include <glog/logging.h>
#include <signal.h>

#include <chrono>
#include <thread>

#include "transport.h"
#include "transport_config.h"

using namespace uccl;

const size_t NUM_FRAMES = 4096 * 64;  // 1GB frame pool
const size_t QUEUE_ID = 0;
const size_t kTestMsgSize = 1024000;
const size_t kTestIters = 1024000000;
const size_t kReportIters = 100;

DEFINE_bool(client, false, "Whether this is a client sending traffic.");

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

    Channel channel;
    int cnt = 0;

    if (FLAGS_client) {
        AFXDPFactory::init(interface_name, "ebpf_transport.o", "ebpf_transport");
        UcclEngine engine(QUEUE_ID, NUM_FRAMES, &channel, client_ip_str,
                          client_port, server_ip_str, server_port,
                          client_mac_char, server_mac_char);
        auto engine_th = std::thread([&engine]() {
            pin_thread_to_cpu(2);
            engine.run();
        });

        pin_thread_to_cpu(3);
        auto ep = Endpoint(&channel);
        auto conn_id = ep.uccl_connect(server_ip_str);

        size_t data_len;
        auto* data = new uint8_t[kTestMsgSize];
        size_t test_msg_size = kTestMsgSize;
        std::vector<uint64_t> rtts;
        auto start_bw = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < kTestIters; i++) {
            auto* data_u32 = reinterpret_cast<uint32_t*>(data);
            for (int j = 0; j < kTestMsgSize / sizeof(uint32_t); j++) {
                data_u32[j] = i * j;
            }
            auto start = std::chrono::high_resolution_clock::now();
            // ep.uccl_send(conn_id, data, kTestMsgSize);
            // ep.uccl_recv(conn_id, data, &data_len);
            ep.uccl_send_async(conn_id, data, test_msg_size);
            ep.uccl_recv_async(conn_id, data, &data_len);
            ep.uccl_send_poll();
            ep.uccl_recv_poll();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start);
            rtts.push_back(duration_us.count());
            if (i % kReportIters == 0) {
                uint64_t med_latency, tail_latency;
                med_latency = Percentile(rtts, 50);
                tail_latency = Percentile(rtts, 99);
                // 24B: 4B FCS + 8B frame delimiter + 12B interframe gap
                double bw_gbps = 0.0;
                auto end_bw = std::chrono::high_resolution_clock::now();
                if (i != 0) {
                    bw_gbps =
                        kTestMsgSize * kReportIters *
                        (AFXDP_MTU * 1.0 /
                         (AFXDP_MTU - kNetHdrLen - kUcclHdrLen - 24)) *
                        8.0 / 1024 / 1024 / 1024 /
                        (std::chrono::duration_cast<std::chrono::microseconds>(
                             end_bw - start_bw)
                             .count() *
                         1e-6);
                }
                start_bw = end_bw;

                LOG(INFO) << "Sent " << i
                          << " pp messages, med rtt: " << med_latency
                          << " us, tail rtt: " << tail_latency << " us, bw "
                          << bw_gbps << " Gbps";
            }
        }

        engine.shutdown();
        engine_th.join();
    } else {
        // AFXDPFactory::init(interface_name, "ebpf_transport_pktloss.o",
        // "ebpf_transport");
        AFXDPFactory::init(interface_name, "ebpf_transport.o", "ebpf_transport");
        UcclEngine engine(QUEUE_ID, NUM_FRAMES, &channel, server_ip_str,
                          server_port, client_ip_str, client_port,
                          server_mac_char, client_mac_char);
        auto engine_th = std::thread([&engine]() {
            pin_thread_to_cpu(2);
            engine.run();
        });

        pin_thread_to_cpu(3);
        auto ep = Endpoint(&channel);
        auto [conn_id, client_ip_str] = ep.uccl_accept();

        auto* data = new uint8_t[kTestMsgSize];
        size_t len;
        size_t test_msg_size = kTestMsgSize;
        for (int i = 0; i < kTestIters; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            // ep.uccl_recv(conn_id, data, &len);
            // ep.uccl_send(conn_id, data, kTestMsgSize);
            ep.uccl_recv_async(conn_id, data, &len);
            ep.uccl_send_async(conn_id, data, test_msg_size);
            ep.uccl_recv_poll();
            ep.uccl_send_poll();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start);
            // This cannot run frequently as it will slow down the client side
            // latency measurements.
            /*CHECK_EQ(len, kTestMsgSize) << "Received message size mismatches";
            for (int j = 0; j < kTestMsgSize / sizeof(uint32_t); j++) {
                CHECK_EQ(reinterpret_cast<uint32_t*>(data)[j], i * j)
                    << "Data mismatch at index " << j;
            }*/
            LOG_EVERY_N(INFO, kReportIters)
                << "Handled " << i << " pp messages, rtt "
                << duration_us.count() << " us";
        }
        engine.shutdown();
        engine_th.join();
    }

    return 0;
}