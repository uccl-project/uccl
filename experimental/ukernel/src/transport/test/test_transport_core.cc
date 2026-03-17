#include "memory_registry.h"
#include "oob.h"
#include "request.h"
#include "test.h"
#include "transport_engine.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void require(bool cond, char const* message) {
  if (!cond) {
    throw std::runtime_error(message);
  }
}

template <typename Fn>
bool throws(Fn&& fn) {
  try {
    fn();
  } catch (...) {
    return true;
  }
  return false;
}

void test_memory_registry() {
  using MemoryRegistry = UKernel::Transport::MemoryRegistry;
  using MR = UKernel::Transport::MR;
  using IpcCache = UKernel::Transport::IpcCache;

  MemoryRegistry registry;
  std::vector<uint8_t> buf_a(512, 0x11);
  std::vector<uint8_t> buf_b(1024, 0x22);

  MR mr_a = registry.track_local_buffer(buf_a.data(), buf_a.size());
  MR mr_a_again = registry.track_local_buffer(buf_a.data(), buf_a.size());
  MR mr_b = registry.track_local_buffer(buf_b.data(), buf_b.size());

  require(mr_a.id == mr_a_again.id, "local MR id should be stable");
  require(mr_a.id != mr_b.id, "different buffers should produce different ids");
  require(registry.get_local_mr(buf_a.data()).id == mr_a.id,
          "exact local MR lookup failed");
  require(registry.get_local_mr(buf_a.data() + 128).id == mr_a.id,
          "range-based local MR lookup failed");
  require(registry.get_local_mr(mr_b.id).address ==
              reinterpret_cast<uint64_t>(buf_b.data()),
          "MR lookup by id failed");

  MR remote0{7, 0x1000ULL, 128, 0, 77};
  MR remote1{8, 0x2000ULL, 256, 0, 88};
  MR remote2{9, 0x3000ULL, 512, 0, 99};
  registry.cache_remote_mrs(/*remote_rank=*/3, {remote0, remote1});
  MR out{};
  require(registry.take_pending_remote_mr(3, out) && out.id == remote0.id,
          "first pending remote MR mismatch");
  registry.cache_remote_mrs(/*remote_rank=*/3, {remote0, remote1, remote2});
  require(registry.take_pending_remote_mr(3, out) && out.id == remote1.id,
          "duplicate remote MR should not be re-enqueued");
  require(registry.take_pending_remote_mr(3, out) && out.id == remote2.id,
          "new remote MR should be enqueued");
  require(!registry.take_pending_remote_mr(3, out),
          "pending remote MR queue should be empty");
  require(registry.get_remote_mr(3, remote2.id).address == remote2.address,
          "cached remote MR lookup failed");

  gpuIpcMemHandle_t handle{};
  std::memset(&handle, 0, sizeof(handle));
  auto* hb = reinterpret_cast<uint8_t*>(&handle);
  hb[0] = 0x5A;
  hb[1] = 0xC3;
  IpcCache cache{};
  cache.handle = handle;
  cache.is_send = true;
  cache.direct_ptr = reinterpret_cast<void*>(0x1234000ULL);
  cache.offset = 64;
  cache.size = 2048;
  cache.device_idx = 5;
  require(registry.register_remote_ipc_cache(4, handle, cache),
          "failed to register remote IPC cache");
  IpcCache cached = registry.get_remote_ipc_cache(4, handle);
  require(cached.direct_ptr == cache.direct_ptr &&
              cached.offset == cache.offset && cached.size == cache.size &&
              cached.device_idx == cache.device_idx,
          "remote IPC cache round-trip mismatch");

  auto released = registry.release_local_buffer(buf_a.data());
  require(released.has_local_mr_id && released.local_mr_id == mr_a.id,
          "release_local_buffer should return released MR id");
  require(throws([&] { registry.get_local_mr(buf_a.data()); }),
          "released local buffer should not be queryable");
}

void test_request_completion() {
  using Request = UKernel::Transport::Request;
  using RequestType = UKernel::Transport::RequestType;

  Request req_single(/*id=*/11, /*buf=*/reinterpret_cast<void*>(0x1000), 32, 64,
                     1, 2, true, RequestType::SEND);
  req_single.pending_signaled.store(1, std::memory_order_release);
  req_single.on_comm_done();
  require(req_single.finished.load(std::memory_order_acquire),
          "single-signal request should complete immediately");

  Request req_multi(/*id=*/12, /*buf=*/reinterpret_cast<void*>(0x2000), 0, 128,
                    3, 4, true, RequestType::RECV);
  req_multi.pending_signaled.store(2, std::memory_order_release);
  req_multi.on_comm_done();
  require(!req_multi.finished.load(std::memory_order_acquire),
          "request should not complete before final signal");
  req_multi.on_comm_done();
  require(req_multi.finished.load(std::memory_order_acquire),
          "request should complete on final signal");
}

void test_peer_transport_kind() {
  using CommunicatorConfig = UKernel::Transport::CommunicatorConfig;
  using CommunicatorMeta = UKernel::Transport::CommunicatorMeta;
  using PeerTransportKind = UKernel::Transport::PeerTransportKind;

  CommunicatorConfig cfg;
  CommunicatorMeta local{};
  local.host_id = "host-a";
  local.ip = "10.0.0.1";
  local.is_ready = true;

  CommunicatorMeta same = local;
  same.ip = "10.0.0.2";
  require(UKernel::Transport::resolve_peer_transport_kind(cfg, local, same) ==
              PeerTransportKind::Ipc,
          "same host_id should resolve to IPC");

  CommunicatorMeta remote{};
  remote.host_id = "host-b";
  remote.ip = "10.0.0.3";
  remote.is_ready = true;
  require(UKernel::Transport::resolve_peer_transport_kind(cfg, local, remote) ==
              PeerTransportKind::Uccl,
          "different host_id should resolve to UCCL");

  cfg.preferred_transport = UKernel::Transport::PreferredTransport::Ipc;
  require(throws([&] {
            (void)UKernel::Transport::resolve_peer_transport_kind(cfg, local,
                                                                  remote);
          }),
          "preferred IPC transport should reject cross-host peers");

  cfg.preferred_transport = UKernel::Transport::PreferredTransport::Uccl;
  require(UKernel::Transport::resolve_peer_transport_kind(cfg, local, same) ==
              PeerTransportKind::Uccl,
          "preferred UCCL transport should override topology");
}

void test_shm_ack_filtering() {
  using ShmRingExchanger = UKernel::Transport::ShmRingExchanger;

  std::exception_ptr rank0_error;
  std::exception_ptr rank1_error;

  std::thread rank0([&] {
    try {
      ShmRingExchanger shm0(/*self_rank=*/0, /*world_size=*/2,
                            "core-shm-ack-filter");
      require(shm0.accept_from(/*peer_rank=*/1, /*timeout_ms=*/5000),
              "rank0 accept_from failed");

      uint32_t status = 0;
      uint64_t seq = 0;
      require(shm0.recv_ack(/*peer_rank=*/1, &status, &seq, /*timeout_ms=*/5000,
                            /*expected_seq=*/99),
              "recv_ack with expected sequence failed");
      require(seq == 99 && status == 1,
              "recv_ack did not filter to the expected sequence");
    } catch (...) {
      rank0_error = std::current_exception();
    }
  });

  std::thread rank1([&] {
    try {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      ShmRingExchanger shm1(/*self_rank=*/1, /*world_size=*/2,
                            "core-shm-ack-filter");
      require(shm1.connect_to(/*peer_rank=*/0, /*timeout_ms=*/5000),
              "rank1 connect_to failed");
      require(shm1.send_ack(/*peer_rank=*/0, /*seq=*/7, /*status=*/5),
              "failed to send first ack");
      require(shm1.send_ack(/*peer_rank=*/0, /*seq=*/99, /*status=*/1),
              "failed to send second ack");
    } catch (...) {
      rank1_error = std::current_exception();
    }
  });

  rank0.join();
  rank1.join();
  if (rank0_error) std::rethrow_exception(rank0_error);
  if (rank1_error) std::rethrow_exception(rank1_error);
}

void test_shm_dual_waiters() {
  using ShmRingExchanger = UKernel::Transport::ShmRingExchanger;
  using IpcCacheWire = UKernel::Transport::IpcCacheWire;

  std::exception_ptr rank0_error;
  std::exception_ptr rank1_error;

  std::thread rank0([&] {
    try {
      ShmRingExchanger shm0(/*self_rank=*/0, /*world_size=*/2,
                            "core-shm-dual");
      require(shm0.accept_from(/*peer_rank=*/1, /*timeout_ms=*/5000),
              "rank0 accept_from failed");

      IpcCacheWire got{};
      uint64_t got_seq = 0;
      uint32_t ack_status = 0;
      uint64_t ack_seq = 0;

      std::thread wait_ipc([&] {
        require(shm0.recv_ipc_cache(/*peer_rank=*/1, got, &got_seq,
                                    /*timeout_ms=*/5000, /*expected_seq=*/11),
                "recv_ipc_cache failed");
      });
      std::thread wait_ack([&] {
        require(shm0.recv_ack(/*peer_rank=*/1, &ack_status, &ack_seq,
                              /*timeout_ms=*/5000, /*expected_seq=*/22),
                "recv_ack failed");
      });

      wait_ipc.join();
      wait_ack.join();

      require(got_seq == 11, "ipc cache sequence mismatch");
      require(got.size == 2048 && got.offset == 64,
              "ipc cache payload mismatch");
      require(ack_seq == 22 && ack_status == 3, "ack payload mismatch");
    } catch (...) {
      rank0_error = std::current_exception();
    }
  });

  std::thread rank1([&] {
    try {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      ShmRingExchanger shm1(/*self_rank=*/1, /*world_size=*/2,
                            "core-shm-dual");
      require(shm1.connect_to(/*peer_rank=*/0, /*timeout_ms=*/5000),
              "rank1 connect_to failed");

      IpcCacheWire wire{};
      std::memset(&wire.handle, 0, sizeof(wire.handle));
      wire.is_send = 0;
      wire.offset = 64;
      wire.size = 2048;
      wire.remote_gpu_idx_ = 0;

      require(shm1.send_ipc_cache(/*peer_rank=*/0, /*seq=*/11, wire),
              "send_ipc_cache failed");
      require(shm1.send_ack(/*peer_rank=*/0, /*seq=*/22, /*status=*/3),
              "send_ack failed");
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

void test_transport_core() {
  test_memory_registry();
  test_request_completion();
  test_peer_transport_kind();
  test_shm_ack_filtering();
  test_shm_dual_waiters();
}
