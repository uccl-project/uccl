#include "oob/oob.h"
#include "test.h"
#include "test_utils.h"
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>

using ShmRingExchanger = UKernel::Transport::ShmRingExchanger;
using IpcCacheWire = UKernel::Transport::IpcCacheWire;

namespace {

using UKernel::Transport::TestUtil::require;
using UKernel::Transport::TestUtil::run_case;

std::string unique_shm_namespace(char const* prefix) {
  std::ostringstream oss;
  oss << prefix << "-" << static_cast<long long>(::getpid()) << "-"
      << std::chrono::steady_clock::now().time_since_epoch().count();
  return oss.str();
}

void test_shm_ipc_cache_exchange() {
  std::string const ring_namespace = unique_shm_namespace("oob-shm-test");
  std::exception_ptr rank0_error;
  std::exception_ptr rank1_error;
  std::mutex done_mu;
  std::condition_variable done_cv;
  bool rank0_finished = false;

  auto mark_rank0_finished = [&] {
    std::lock_guard<std::mutex> lk(done_mu);
    rank0_finished = true;
    done_cv.notify_all();
  };

  std::thread rank0([&]() {
    try {
      ShmRingExchanger shm0(/*self_rank=*/0, /*world_size=*/2,
                            ring_namespace);

      require(shm0.accept_from(/*peer_rank=*/1, /*timeout_ms=*/5000),
              "rank0 accept_from(1) failed");

      IpcCacheWire got{};
      uint64_t seq = 0;
      require(
          shm0.recv_ipc_cache(/*peer_rank=*/1, got, &seq, /*timeout_ms=*/5000),
          "rank0 recv_ipc_cache(1) failed");

      require(seq == 42, "rank0 recv sequence mismatch");
      require(got.is_send == 1, "rank0 recv is_send mismatch");
      require(got.offset == 0x12345678ULL, "rank0 recv offset mismatch");
      require(got.size == 4096ULL, "rank0 recv size mismatch");

      auto const* hb = reinterpret_cast<uint8_t const*>(&got.handle);
      require(hb[0] == 0xA0, "rank0 recv handle prefix[0] mismatch");
      require(hb[1] == 0xA1, "rank0 recv handle prefix[1] mismatch");
      mark_rank0_finished();
    } catch (...) {
      rank0_error = std::current_exception();
      mark_rank0_finished();
    }
  });

  std::thread rank1([&]() {
    try {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));

      ShmRingExchanger shm1(/*self_rank=*/1, /*world_size=*/2,
                            ring_namespace);

      require(shm1.connect_to(/*peer_rank=*/0, /*timeout_ms=*/5000),
              "rank1 connect_to(0) failed");

      IpcCacheWire w{};
      std::memset(&w.handle, 0, sizeof(w.handle));
      auto* hb = reinterpret_cast<uint8_t*>(&w.handle);
      for (size_t i = 0; i < sizeof(gpuIpcMemHandle_t); ++i) {
        hb[i] = static_cast<uint8_t>(0xA0 + (i & 0x0F));
      }

      w.is_send = 1;
      w.offset = 0x12345678ULL;
      w.size = 4096ULL;

      require(shm1.send_ipc_cache(/*peer_rank=*/0, /*seq=*/42, w),
              "rank1 send_ipc_cache failed");

      std::unique_lock<std::mutex> lk(done_mu);
      bool done = done_cv.wait_for(
          lk, std::chrono::seconds(5), [&] { return rank0_finished; });
      require(done, "rank1 timed out waiting for rank0 to finish");
    } catch (...) {
      rank1_error = std::current_exception();
    }
  });

  rank0.join();
  rank1.join();
  if (rank0_error) std::rethrow_exception(rank0_error);
  if (rank1_error) std::rethrow_exception(rank1_error);
}

}  // namespace

void test_shm_oob() {
  run_case("transport unit", "shm oob ipc cache exchange",
           test_shm_ipc_cache_exchange);
}
