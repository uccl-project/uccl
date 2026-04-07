#include "memory/memory_manager.h"
#include "oob/oob.h"
#include "request.h"
#include "test.h"
#include "test_utils.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <sstream>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {

using UKernel::Transport::TestUtil::require;
using UKernel::Transport::TestUtil::run_case;
using UKernel::Transport::TestUtil::throws;

std::string unique_shm_namespace(char const* prefix) {
  std::ostringstream oss;
  oss << prefix << "-" << static_cast<long long>(::getpid()) << "-"
      << std::chrono::steady_clock::now().time_since_epoch().count();
  return oss.str();
}

void test_memory_manager() {
  using MemoryManager = UKernel::Transport::MemoryManager;
  using LocalMR = UKernel::Transport::LocalMR;
  using RemoteMR = UKernel::Transport::RemoteMR;
  using RemoteIpc = UKernel::Transport::RemoteIpc;

  MemoryManager mm;
  std::vector<uint8_t> buf_a(512, 0x11);
  std::vector<uint8_t> buf_b(1024, 0x22);

  auto tracked_a = mm.register_local(buf_a.data(), buf_a.size());
  auto tracked_a_again = mm.register_local(buf_a.data(), buf_a.size());
  auto tracked_b = mm.register_local(buf_b.data(), buf_b.size());
  LocalMR mr_a = tracked_a.mr;
  LocalMR mr_a_again = tracked_a_again.mr;
  LocalMR mr_b = tracked_b.mr;

  require(mr_a.id == mr_a_again.id, "local MR id should be stable");
  require(mr_a.id != mr_b.id, "different buffers should produce different ids");
  require(mm.find_local_by_ptr(buf_a.data()).id == mr_a.id,
          "exact local MR lookup failed");
  require(mm.find_local_by_ptr(buf_a.data() + 128).id == mr_a.id,
          "range-based local MR lookup failed");
  require(mm.get_local_mr(mr_b.id).address ==
              reinterpret_cast<uint64_t>(buf_b.data()),
          "MR lookup by id failed");

  RemoteMR remote0{7, 0x1000ULL, 77, 128};
  RemoteMR remote1{8, 0x2000ULL, 88, 256};
  RemoteMR remote2{9, 0x3000ULL, 99, 512};
  mm.cache_remote_mrs(/*remote_rank=*/3, {remote0, remote1});
  mm.cache_remote_mrs(/*remote_rank=*/3, {remote0, remote1, remote2});
  require(mm.get_remote_mr(3, remote0.id).address == remote0.address,
          "cached remote MR lookup for first entry failed");
  require(mm.get_remote_mr(3, remote1.id).address == remote1.address,
          "cached remote MR lookup for second entry failed");
  require(mm.get_remote_mr(3, remote2.id).address == remote2.address,
          "cached remote MR lookup failed");

  gpuIpcMemHandle_t handle{};
  std::memset(&handle, 0, sizeof(handle));
  auto* hb = reinterpret_cast<uint8_t*>(&handle);
  hb[0] = 0x5A;
  hb[1] = 0xC3;
  RemoteIpc cache{};
  cache.handle = handle;
  cache.direct_ptr = reinterpret_cast<void*>(0x1234000ULL);
  cache.offset = 64;
  cache.size = 2048;
  cache.device_idx = 5;
  require(mm.register_remote_ipc(4, handle, cache),
          "failed to register remote IPC cache");
  RemoteIpc cached = mm.get_remote_ipc(4, handle);
  require(cached.direct_ptr == cache.direct_ptr &&
              cached.offset == cache.offset && cached.size == cache.size &&
              cached.device_idx == cache.device_idx,
          "remote IPC cache round-trip mismatch");

  auto released_shared_ref = mm.deregister_local(buf_a.data());
  require(released_shared_ref.mr_id == mr_a.id &&
              !released_shared_ref.fully_released,
          "releasing one retained reference should not fully release the MR");

  auto resized = mm.register_local(buf_a.data(), buf_a.size() / 2);
  require(resized.replaced && resized.replaced_mr_id == mr_a.id,
          "resized registration should report fully replaced MR");
  require(resized.mr.id != mr_a.id,
          "resized registration should allocate a new MR id");
  require(mm.find_local_by_ptr(buf_a.data()).id == resized.mr.id,
          "resized exact lookup should resolve to new MR");
  require(throws([&] {
            (void)mm.find_local_by_ptr(buf_a.data() + buf_a.size() / 2 + 1);
          }),
          "lookup beyond resized range should fail");

  auto released = mm.deregister_local(buf_a.data());
  require(released.mr_id == resized.mr.id && released.fully_released,
          "release_local_buffer should return the fully released MR id");
  require(throws([&] { (void)mm.find_local_by_ptr(buf_a.data()); }),
          "released local buffer should not be queryable");

  auto released_a_once = mm.deregister_local(buf_a.data());
  require(released_a_once.mr_id == 0,
          "released resized buffer should no longer be tracked");

  auto released_b_once = mm.deregister_local(buf_b.data());
  require(released_b_once.mr_id == mr_b.id && released_b_once.fully_released,
          "single-use MR should be fully released on first release");
}

void test_request_completion() {
  using Request = UKernel::Transport::Request;
  using RequestState = UKernel::Transport::RequestState;
  using RequestType = UKernel::Transport::RequestType;

  Request req_single(/*id=*/11, /*match_seq=*/101,
                     /*buffer=*/reinterpret_cast<void*>(0x1000),
                     /*size_bytes=*/64, UKernel::Transport::RemoteSlice{},
                     RequestType::Send);
  req_single.mark_queued(1);
  req_single.mark_running();
  req_single.complete_one();
  require(req_single.load_state(std::memory_order_acquire) ==
              RequestState::Completed,
          "single-signal request should complete immediately");

  Request req_multi(/*id=*/12, /*match_seq=*/202,
                    /*buffer=*/reinterpret_cast<void*>(0x2000),
                    /*size_bytes=*/128, UKernel::Transport::RemoteSlice{},
                    RequestType::Recv);
  req_multi.mark_queued(2);
  req_multi.mark_running();
  req_multi.complete_one();
  require(!req_multi.is_finished(std::memory_order_acquire),
          "request should not complete before final signal");
  req_multi.complete_one();
  require(req_multi.load_state(std::memory_order_acquire) ==
              RequestState::Completed,
          "request should complete on final signal");

  Request req_failed(/*id=*/13, /*match_seq=*/303,
                     /*buffer=*/reinterpret_cast<void*>(0x3000),
                     /*size_bytes=*/64, UKernel::Transport::RemoteSlice{},
                     RequestType::Recv);
  req_failed.mark_queued(1);
  req_failed.mark_running();
  req_failed.mark_failed();
  require(req_failed.has_failed(std::memory_order_acquire),
          "failed request should report terminal failure");
}

void test_peer_transport_kind() {
  using CommunicatorConfig = UKernel::Transport::CommunicatorConfig;
  using CommunicatorMeta = UKernel::Transport::CommunicatorMeta;
  using PeerTransportKind = UKernel::Transport::PeerTransportKind;

  CommunicatorConfig cfg;
  CommunicatorMeta local{};
  local.host_id = "host-a";
  local.ip = "10.0.0.1";

  CommunicatorMeta same = local;
  same.ip = "10.0.0.2";
  require(UKernel::Transport::resolve_peer_transport_kind(cfg, local, same) ==
              PeerTransportKind::Ipc,
          "same host_id should resolve to IPC");

  CommunicatorMeta remote{};
  remote.host_id = "host-b";
  remote.ip = "10.0.0.3";
  local.rdma_capable = true;
  remote.rdma_capable = true;
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
  require(throws([&] {
            (void)UKernel::Transport::resolve_peer_transport_kind(cfg, local,
                                                                  same);
          }),
          "preferred UCCL transport should require RDMA-capable peers");

  same.rdma_capable = true;
  require(
      UKernel::Transport::resolve_peer_transport_kind(cfg, local, same) ==
          PeerTransportKind::Uccl,
      "preferred UCCL transport should use UCCL when peers are RDMA-capable");
}

void test_shm_ack_filtering() {
  using ShmRingExchanger = UKernel::Transport::ShmRingExchanger;
  std::string const ring_namespace =
      unique_shm_namespace("core-shm-ack-filter");

  std::exception_ptr rank0_error;
  std::exception_ptr rank1_error;

  std::thread rank0([&] {
    try {
      ShmRingExchanger shm0(/*self_rank=*/0, /*world_size=*/2, ring_namespace);
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
      ShmRingExchanger shm1(/*self_rank=*/1, /*world_size=*/2, ring_namespace);
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
  std::string const ring_namespace = unique_shm_namespace("core-shm-dual");

  std::exception_ptr rank0_error;
  std::exception_ptr rank1_error;

  std::thread rank0([&] {
    try {
      ShmRingExchanger shm0(/*self_rank=*/0, /*world_size=*/2, ring_namespace);
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
      ShmRingExchanger shm1(/*self_rank=*/1, /*world_size=*/2, ring_namespace);
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
  run_case("transport unit", "memory manager", test_memory_manager);
  run_case("transport unit", "request completion", test_request_completion);
  run_case("transport unit", "peer transport kind", test_peer_transport_kind);
}
