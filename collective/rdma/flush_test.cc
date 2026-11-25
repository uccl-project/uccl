#include "transport.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <sys/socket.h>

using namespace uccl;

// Requires one active RDMA NIC. Run with --gpu to flush registered GPU memory.
// --noop runs only the no-op cases, including on a reused request.
int main(int argc, char** argv) {
  bool use_gpu = false;
  bool noop_only = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--gpu") == 0) {
      use_gpu = true;
    } else if (std::strcmp(argv[i], "--noop") == 0) {
      noop_only = true;
    } else {
      std::cerr << "Usage: " << argv[0] << " [--gpu] [--noop]\n";
      return 1;
    }
  }

  RDMAEndpoint ep(ucclParamNUM_ENGINES());
  // Only the receiver's self-connected flush QP is needed; no data engines
  // or remote host are involved in these request-lifecycle tests.
  UcclFlow sender(&ep, 0, 0, 0, "", 0, true);
  UcclFlow receiver(&ep, 0, 0, 1, "", 0, false);
  int sockets[2];
  UCCL_CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  std::thread setup([&] { sender.create_fifo_and_gpuflush(sockets[0], 0); });
  receiver.create_fifo_and_gpuflush(sockets[1], 0);
  setup.join();
  close(sockets[0]);
  close(sockets[1]);

  constexpr int kSize = 4096;
  void* buffer;
  if (use_gpu) {
    GPU_RT_CHECK(gpuSetDevice(0));
    GPU_RT_CHECK(gpuMalloc(&buffer, kSize));
    GPU_RT_CHECK(gpuMemsetAsync(buffer, 0, kSize, nullptr));
    GPU_RT_CHECK(gpuStreamSynchronize(nullptr));
  } else {
    buffer = mmap(nullptr, kSize, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    UCCL_CHECK(buffer != MAP_FAILED);
  }
  Mhandle* mh;
  UCCL_CHECK(ep.uccl_regmr(&receiver, buffer, kSize, 0, &mh) == 0);

  auto poll = [&](ucclRequest* req) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!ep.uccl_poll_ureq_once(req)) {
      UCCL_CHECK(std::chrono::steady_clock::now() < deadline)
          << "Flush completion timed out";
    }
  };

  auto check_noop = [&](ucclRequest* req, int n) {
    int sizes[kMaxRecv] = {};
    // No buffer or MR should be accessed for an empty flush.
    UCCL_CHECK(ep.uccl_flush(&receiver, nullptr, nullptr, sizes, n, req) == 0);
    UCCL_CHECK(req->type == ReqFlush);
    UCCL_CHECK(req->context == &receiver);
    UCCL_CHECK(req->rc_or_flush_done == true);
    poll(req);
  };

  ucclRequest req{};
  if (noop_only) {
    for (int n : {0, 1, static_cast<int>(kMaxRecv)}) {
      req.type = ReqRx;
      req.context = nullptr;
      req.rc_or_flush_done = false;
      check_noop(&req, n);
      check_noop(&req, n);
    }
    std::cout << "PASS: empty and all-zero flush requests (fresh and reused)\n";
  } else {
    for (int i = 0; i < 1000; ++i) {
      // Alternate fresh state and a previously completed request. A stale
      // completion flag must not make the next RDMA READ appear completed.
      if (i % 2 == 0) {
        req = {};
        req.type = ReqRx;
      }
      req.context = nullptr;
      Mhandle* handles[kMaxRecv] = {};
      void* data[kMaxRecv] = {};
      int sizes[kMaxRecv] = {};
      constexpr int kLast = kMaxRecv / 2;
      handles[kLast] = mh;
      data[kLast] = buffer;
      sizes[kLast] = kSize;
      UCCL_CHECK(
          ep.uccl_flush(&receiver, handles, data, sizes, kMaxRecv, &req) == 0);
      UCCL_CHECK(req.type == ReqFlush);
      UCCL_CHECK(req.context == &receiver);
#ifndef __HIP_PLATFORM_AMD__
      UCCL_CHECK(req.rc_or_flush_done == false);
#endif
      poll(&req);
      UCCL_CHECK(req.rc_or_flush_done == true);
      check_noop(&req, kMaxRecv);
    }
    std::cout << "PASS: 1000 posted flushes, each followed by a no-op flush\n";
  }

  ep.uccl_deregmr(mh);
  if (use_gpu) {
    GPU_RT_CHECK(gpuFree(buffer));
  } else {
    UCCL_CHECK(munmap(buffer, kSize) == 0);
  }
  return 0;
}
