#include "transport.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <signal.h>

#include <thread>

using namespace uccl;

const uint8_t SERVER_ETHERNET_ADDRESS[] = {0x0a, 0xff, 0xea, 0x86, 0x04, 0xd9};
const uint8_t CLIENT_ETHERNET_ADDRESS[] = {0x0a, 0xff, 0xdf, 0x30, 0xe7, 0x59};
const uint32_t SERVER_IPV4_ADDRESS = 0xac1f16f9;  // 172.31.22.249
const uint32_t CLIENT_IPV4_ADDRESS = 0xac1f10c6;  // 172.31.16.198
const uint16_t SERVER_PORT = 40000;
const uint16_t CLIENT_PORT = 40000;
const size_t NUM_FRAMES = 4096 * 16;
const size_t QUEUE_ID = 0;
const size_t kTestMsgSize = 1024;

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
    signal(SIGALRM, interrupt_handler);
    alarm(10);

    AFXDPFactory::init("ens6", "ebpf_client.o", "ebpf_client");

    Channel channel;

    auto engine_th = std::thread([&channel]() {
        if (FLAGS_client) {
            UcclEngine client(QUEUE_ID, NUM_FRAMES, &channel,
                              CLIENT_IPV4_ADDRESS, CLIENT_PORT,
                              SERVER_IPV4_ADDRESS, SERVER_PORT,
                              CLIENT_ETHERNET_ADDRESS, SERVER_ETHERNET_ADDRESS);
            client.Run();
        } else {
            UcclEngine server(QUEUE_ID, NUM_FRAMES, &channel,
                              SERVER_IPV4_ADDRESS, SERVER_PORT,
                              CLIENT_IPV4_ADDRESS, CLIENT_PORT,
                              SERVER_ETHERNET_ADDRESS, CLIENT_ETHERNET_ADDRESS);
            server.Run();
        }
    });

    if (FLAGS_client) {
        auto ep = Endpoint(&channel);
        auto conn_id = ep.Connect(SERVER_IPV4_ADDRESS);
        auto* data = new uint8_t[kTestMsgSize];
        auto* data_u32 = reinterpret_cast<uint32_t*>(data);
        for (int i = 0; i < kTestMsgSize / sizeof(uint32_t); i++) {
            data_u32[i] = i;
        }
        ep.Send(conn_id, data, kTestMsgSize);
    } else {
        auto ep = Endpoint(&channel);
        auto conn_id = ep.Accept();
        auto* data = new uint8_t[kTestMsgSize];
        size_t len;
        ep.Recv(conn_id, data, &len);
        for (int i = 0; i < kTestMsgSize / sizeof(uint32_t); i++) {
            if (reinterpret_cast<uint32_t*>(data)[i] != i) {
                LOG(ERROR) << "Data mismatch at index " << i;
            }
        }
    }

    return 0;
}