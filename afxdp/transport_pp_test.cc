#include <gflags/gflags.h>
#include <glog/logging.h>
#include <signal.h>

#include <chrono>
#include <thread>

#include "transport.h"

using namespace uccl;

const uint8_t SERVER_ETHERNET_ADDRESS[] = {0x0a, 0xff, 0xea, 0x86, 0x04, 0xd9};
const uint8_t CLIENT_ETHERNET_ADDRESS[] = {0x0a, 0xff, 0xdf, 0x30, 0xe7, 0x59};
const std::string server_addr_str = "172.31.22.249";
const std::string client_addr_str = "172.31.16.198";
const uint16_t SERVER_PORT = 40000;
const uint16_t CLIENT_PORT = 40000;
const size_t NUM_FRAMES = 4096 * 4;  // 256MB frame pool
const size_t QUEUE_ID = 0;
const size_t kTestMsgSize = 1024000;
const size_t kTestIters = 1024000000;
const size_t kReportIters = 1000;

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
        AFXDPFactory::init("ens6", "ebpf_transport.o", "ebpf_transport");
        UcclEngine engine(QUEUE_ID, NUM_FRAMES, &channel, client_addr_str,
                          CLIENT_PORT, server_addr_str, SERVER_PORT,
                          CLIENT_ETHERNET_ADDRESS, SERVER_ETHERNET_ADDRESS);
        auto engine_th = std::thread([&engine]() {
            pin_thread_to_cpu(2);
            engine.run();
        });

        pin_thread_to_cpu(3);
        auto ep = Endpoint(&channel);
        auto conn_id = ep.uccl_connect(server_addr_str);

        size_t data_len;
        auto* data = new uint8_t[kTestMsgSize];
        auto* data_u32 = reinterpret_cast<uint32_t*>(data);
        for (int j = 0; j < kTestMsgSize / sizeof(uint32_t); j++) {
            data_u32[j] = j;
        }
        size_t test_msg_size = kTestMsgSize;
        std::vector<uint64_t> rtts;
        auto start_bw = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < kTestIters; i++) {
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
        // AFXDPFactory::init("ens6", "ebpf_transport_pktloss.o",
        // "ebpf_transport");
        AFXDPFactory::init("ens6", "ebpf_transport.o", "ebpf_transport");
        UcclEngine engine(QUEUE_ID, NUM_FRAMES, &channel, server_addr_str,
                          SERVER_PORT, client_addr_str, CLIENT_PORT,
                          SERVER_ETHERNET_ADDRESS, CLIENT_ETHERNET_ADDRESS);
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
            ep.uccl_recv_poll();
            ep.uccl_send_async(conn_id, data, test_msg_size);
            ep.uccl_send_poll();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start);
            /*
            CHECK_EQ(len, kTestMsgSize) << "Received message size mismatches";
            for (int j = 0; j < kTestMsgSize / sizeof(uint32_t); j++) {
            CHECK_EQ(reinterpret_cast<uint32_t*>(data)[j], j)
            << "Data mismatch at index " << j;
            }
            memset(data, 0, kTestMsgSize);
            */
            LOG_EVERY_N(INFO, kReportIters)
                << "Handled " << i << " pp messages, rtt "
                << duration_us.count() << " us";
        }
        engine.shutdown();
        engine_th.join();
    }

    return 0;
}