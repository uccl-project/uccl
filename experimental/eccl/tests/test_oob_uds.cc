#include "oob.h"
#include "test.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

void test_uds_oob() {
  std::cout << "[TEST] UDS OOB test start\n";

  std::thread rank0([&]() {
    UdsExchanger uds0(/*self_rank=*/0);

    if (!uds0.accept_from(/*peer_rank=*/1, /*timeout_ms=*/5000)) {
      std::cerr << "[ERROR] rank0 accept_from(1) failed\n";
      return;
    }
    std::cout << "[INFO] rank0 accepted connection from rank1\n";

    IpcCacheWire got{};
    uint64_t seq = 0;
    if (!uds0.recv_ipc_cache(/*peer_rank=*/1, got, &seq, /*timeout_ms=*/5000)) {
      std::cerr << "[ERROR] rank0 recv_ipc_cache(1) failed\n";
      return;
    }

    std::cout << "[INFO] rank0 received IpcCacheWire"
              << " seq=" << seq
              << " is_send=" << (int)got.is_send
              << " offset=" << got.offset
              << " size=" << got.size << "\n";

    bool ok = true;
    ok &= (seq == 42);
    ok &= (got.is_send == 1);
    ok &= (got.offset == 0x12345678ULL);
    ok &= (got.size == 4096ULL);

    // handle is opaque; here we just sanity-check that it matches the pattern we set
    const uint8_t* hb = reinterpret_cast<const uint8_t*>(&got.handle);
    ok &= (hb[0] == 0xA0);
    ok &= (hb[1] == 0xA1);

    if (!ok) {
      std::cerr << "[ERROR] rank0 validation failed\n";
      return;
    }

    std::cout << "[INFO] rank0 validation OK\n";
  });

  std::thread rank1([&]() {
    // small delay so rank0 can start listening; connect_to has retry anyway
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    UdsExchanger uds1(/*self_rank=*/1);

    if (!uds1.connect_to(/*peer_rank=*/0, /*timeout_ms=*/5000)) {
      std::cerr << "[ERROR] rank1 connect_to(0) failed\n";
      return;
    }
    std::cout << "[INFO] rank1 connected to rank0\n";

    IpcCacheWire w{};
    std::memset(&w.handle, 0, sizeof(w.handle));
    // Fill recognizable pattern into handle bytes (no CUDA call needed for unit test)
    uint8_t* hb = reinterpret_cast<uint8_t*>(&w.handle);
    for (size_t i = 0; i < sizeof(gpuIpcMemHandle_t); ++i) {
      hb[i] = static_cast<uint8_t>(0xA0 + (i & 0x0F));
    }

    w.is_send = 1;
    w.offset = 0x12345678ULL;
    w.size = 4096ULL;

    const uint64_t seq = 42;
    if (!uds1.send_ipc_cache(/*peer_rank=*/0, seq, w)) {
      std::cerr << "[ERROR] rank1 send_ipc_cache failed\n";
      return;
    }

    std::cout << "[INFO] rank1 sent IpcCacheWire seq=" << seq << "\n";
  });

  rank0.join();
  rank1.join();

  std::cout << "[TEST] UDS OOB test completed\n";
}
