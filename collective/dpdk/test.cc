#include "dpdk.h"
#include "util_dpdk.h"
#include <glog/logging.h>
#include <signal.h>

using namespace uccl;

static char mac_str[] = "6c:92:bf:f3:2e:1a";

bool volatile quit;

void interrupt_handler(int signal) {
  (void)signal;
  quit = true;
}



int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;

  signal(SIGINT, interrupt_handler);
  signal(SIGTERM, interrupt_handler);

  (void)argc;
  Dpdk dpdk;
  dpdk.InitDpdk(1, argv);

  uint16_t client_port_id = dpdk.GetPmdPortIdByMac(mac_str);

  DPDKFactory dpdk_factory(client_port_id, 1, 1);
  dpdk_factory.Init();

  DPDKSocket* socket = dpdk_factory.CreateSocket(0);
  while (!quit) {
    Packet* pkt = nullptr;
    do {
      pkt = socket->pop_packet();
    } while (!pkt && !quit);
    LOG(INFO) << "popped packet: " << pkt;
  }

  return 0;
}