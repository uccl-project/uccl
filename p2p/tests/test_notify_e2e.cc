// Two-process notification test over the real network control path.
// Exercises uccl_engine_send_notif -> (transport wire) -> uccl_engine_get_notifs
// with payloads far beyond the old 256-byte and 16 KiB limits.
//
//   build:  make notif_e2e   (from p2p/)
//   server: ./notif_e2e server
//   client: ./notif_e2e client <ip> <port>
//
// Run with UCCL_P2P_TRANSPORT=tcp UCCL_P2P_DISABLE_IPC=1 so a localhost pair
// takes the network path instead of the IPC shortcut.
#include "uccl_engine.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

static std::vector<size_t> const kSizes = {5, 305, 4096, 16385, 262144,
                                           2u << 20};

static std::string pattern_msg(size_t idx, size_t size) {
  std::string m = std::to_string(idx) + ":";
  m.resize(size, static_cast<char>('a' + idx % 26));
  return m;
}

static bool init_engine_gpu(uccl_engine_t* eng) {
  // The endpoint resolves its local GPU lazily on first registration; give it
  // a small CUDA buffer, as any real NIXL client would.
  void* buf = nullptr;
  if (cudaMalloc(&buf, 1 << 20) != cudaSuccess) return false;
  uccl_mr_t mr = 0;
  return uccl_engine_reg(eng, reinterpret_cast<uintptr_t>(buf), 1 << 20,
                         mr) == 0;
}

int run_server() {
  uccl_engine_t* eng = uccl_engine_create(1, false);
  if (!eng) return fprintf(stderr, "engine create failed\n"), 1;
  if (!init_engine_gpu(eng)) return fprintf(stderr, "reg failed\n"), 1;

  char* meta = nullptr;
  if (uccl_engine_get_metadata(eng, &meta) != 0)
    return fprintf(stderr, "get_metadata failed\n"), 1;
  printf("META %s\n", meta);
  fflush(stdout);

  char ipbuf[64] = {0};
  int gpu_idx = -1;
  uccl_conn_t* conn = uccl_engine_accept(eng, ipbuf, sizeof(ipbuf), &gpu_idx);
  if (!conn) return fprintf(stderr, "accept failed\n"), 1;
  fprintf(stderr, "accepted from %s gpu=%d\n", ipbuf, gpu_idx);

  std::map<size_t, bool> got;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (got.size() < kSizes.size() &&
         std::chrono::steady_clock::now() < deadline) {
    for (auto const& n : uccl_engine_get_notifs()) {
      size_t idx = std::strtoul(n.msg.c_str(), nullptr, 10);
      if (idx >= kSizes.size()) {
        fprintf(stderr, "FAIL: unparseable notif index\n");
        return 1;
      }
      std::string expect = pattern_msg(idx, kSizes[idx]);
      if (n.msg != expect) {
        fprintf(stderr, "FAIL: size %zu corrupted (got %zu bytes)\n",
                kSizes[idx], n.msg.size());
        return 1;
      }
      if (n.name != "notif-e2e-client") {
        fprintf(stderr, "FAIL: bad sender name '%s'\n", n.name.c_str());
        return 1;
      }
      got[idx] = true;
      fprintf(stderr, "ok: received intact notification of %zu bytes\n",
              kSizes[idx]);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  if (got.size() != kSizes.size()) {
    fprintf(stderr, "FAIL: only %zu/%zu notifications arrived\n", got.size(),
            kSizes.size());
    return 1;
  }
  printf("SERVER-OK all %zu notification sizes intact\n", kSizes.size());
  return 0;
}

int run_client(char const* ip, int port) {
  uccl_engine_t* eng = uccl_engine_create(1, false);
  if (!eng) return fprintf(stderr, "engine create failed\n"), 1;
  if (!init_engine_gpu(eng)) return fprintf(stderr, "reg failed\n"), 1;

  uccl_conn_t* conn = uccl_engine_connect(eng, ip, "0", port);
  if (!conn) return fprintf(stderr, "connect failed\n"), 1;

  for (size_t i = 0; i < kSizes.size(); ++i) {
    notify_msg_t nm;
    nm.name = "notif-e2e-client";
    nm.msg = pattern_msg(i, kSizes[i]);
    int rc = uccl_engine_send_notif(conn, &nm);
    if (rc < 0) {
      fprintf(stderr, "FAIL: send_notif(%zu bytes) rc=%d\n", kSizes[i], rc);
      return 1;
    }
    fprintf(stderr, "sent %zu-byte notification (rc=%d)\n", kSizes[i], rc);
  }
  // Give the control thread time to flush before teardown.
  std::this_thread::sleep_for(std::chrono::seconds(2));
  printf("CLIENT-OK\n");
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "server") == 0) return run_server();
  if (argc >= 4 && std::strcmp(argv[1], "client") == 0)
    return run_client(argv[2], std::atoi(argv[3]));
  fprintf(stderr, "usage: %s server | client <ip> <port>\n", argv[0]);
  return 2;
}
