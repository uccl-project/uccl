#include "memory/ipc_manager.h"
#include "memory/mr_manager.h"
#include "oob/oob.h"
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
  using MRManager = UKernel::Transport::MRManager;
  using MR = UKernel::Transport::MR;
  using MRItem = UKernel::Transport::MRItem;
  using IPCItem = UKernel::Transport::IPCItem;

  MRManager mrm;
  UKernel::Transport::IPCManager ipcm;
  std::vector<uint8_t> buf_a(512, 0x11);
  std::vector<uint8_t> buf_b(1024, 0x22);
  std::vector<uint32_t> reg_ids;
  std::vector<uint32_t> dereg_ids;

  mrm.bind_backend(
      [&](uint32_t mr_id, void* ptr, size_t len) {
        require(ptr != nullptr && len > 0, "backend register args invalid");
        reg_ids.push_back(mr_id);
        return true;
      },
      [&](uint32_t mr_id) { dereg_ids.push_back(mr_id); });

  auto tracked_a = mrm.create_local_mr(buf_a.data(), buf_a.size());
  auto tracked_a_again = mrm.create_local_mr(buf_a.data(), buf_a.size());
  auto tracked_b = mrm.create_local_mr(buf_b.data(), buf_b.size());
  MR mr_a = tracked_a.mr;
  MR mr_a_again = tracked_a_again.mr;
  MR mr_b = tracked_b.mr;

  require(mr_a.id == mr_a_again.id, "local MR id should be stable");
  require(mr_a.id != mr_b.id, "different buffers should produce different ids");
  require(reg_ids.size() == 2,
          "backend register should run once per unique local buffer");
  require(mrm.get_mr(buf_a.data()).mr.id == mr_a.id,
          "exact local MR lookup failed");
  require(mrm.get_mr(buf_a.data() + 128).mr.id == mr_a.id,
          "range-based local MR lookup failed");
  require(mrm.get_mr(mr_b.id).mr.address ==
              reinterpret_cast<uint64_t>(buf_b.data()),
          "MR lookup by id failed");

  MR remote0{7, 0x1000ULL, 128, 0, 77};
  MR remote1{8, 0x2000ULL, 256, 0, 88};
  MR remote2{9, 0x3000ULL, 512, 0, 99};
  MRItem ri0{};
  ri0.mr = remote0;
  ri0.valid = true;
  MRItem ri1{};
  ri1.mr = remote1;
  ri1.valid = true;
  MRItem ri2{};
  ri2.mr = remote2;
  ri2.valid = true;
  mrm.register_remote_mrs(/*remote_rank=*/3, {ri0, ri1});
  mrm.register_remote_mrs(/*remote_rank=*/3, {ri0, ri1, ri2});
  require(mrm.get_mr(3, remote0.id).mr.address == remote0.address,
          "cached remote MR lookup for first entry failed");
  require(mrm.get_mr(3, remote1.id).mr.address == remote1.address,
          "cached remote MR lookup for second entry failed");
  require(mrm.get_mr(3, remote2.id).mr.address == remote2.address,
          "cached remote MR lookup failed");

  gpuIpcMemHandle_t handle{};
  std::memset(&handle, 0, sizeof(handle));
  auto* hb = reinterpret_cast<uint8_t*>(&handle);
  hb[0] = 0x5A;
  hb[1] = 0xC3;
  IPCItem cache{};
  cache.handle = handle;
  cache.direct_ptr = reinterpret_cast<void*>(0x1234000ULL);
  cache.base_offset = 64;
  cache.bytes = 2048;
  cache.device_idx = 5;
  require(ipcm.register_remote_ipc(4, cache),
          "failed to register remote IPC cache");
  IPCItem cached = ipcm.get_ipc(4, handle);
  require(cached.direct_ptr == cache.direct_ptr &&
              cached.base_offset == cache.base_offset &&
              cached.bytes == cache.bytes &&
              cached.device_idx == cache.device_idx,
          "remote IPC cache round-trip mismatch");

  require(mrm.delete_mr(buf_a.data()), "first local MR delete should succeed");
  require(dereg_ids.size() == 1 && dereg_ids.back() == mr_a.id,
          "backend deregister should run when local MR is deleted");

  auto resized = mrm.create_local_mr(buf_a.data(), buf_a.size() / 2);
  require(resized.mr.id != mr_a.id,
          "resized registration should allocate a new MR id");
  require(reg_ids.size() == 3 && reg_ids.back() == resized.mr.id,
          "resized local MR should trigger backend register for new id");
  require(mrm.get_mr(buf_a.data()).mr.id == resized.mr.id,
          "resized exact lookup should resolve to new MR");
  require(!mrm.get_mr(buf_a.data() + buf_a.size() / 2 + 1).valid,
          "lookup beyond resized range should fail");

  require(mrm.delete_mr(buf_a.data()),
          "resized local MR delete should succeed");
  require(dereg_ids.size() == 2 && dereg_ids.back() == resized.mr.id,
          "resized local MR delete should trigger backend deregister");
  require(!mrm.get_mr(buf_a.data()).valid,
          "released local buffer should not be queryable");

  require(!mrm.delete_mr(buf_a.data()),
          "released resized buffer should no longer be tracked");

  require(mrm.delete_mr(buf_b.data()), "single-use MR should be deletable");
  require(dereg_ids.size() == 3 && dereg_ids.back() == mr_b.id,
          "single-use local MR delete should trigger backend deregister");

  // Verify "create first, bind/sync later" path.
  MRManager late_bound;
  std::vector<uint32_t> late_reg_ids;
  std::vector<uint32_t> late_dereg_ids;
  std::vector<uint8_t> buf_c(256, 0x33);

  auto tracked_c = late_bound.create_local_mr(buf_c.data(), buf_c.size());
  require(tracked_c.valid, "late-bound test local MR should be valid");

  late_bound.bind_backend(
      [&](uint32_t mr_id, void* ptr, size_t len) {
        require(ptr == buf_c.data() && len == buf_c.size(),
                "late-bound backend register args mismatch");
        late_reg_ids.push_back(mr_id);
        return true;
      },
      [&](uint32_t mr_id) { late_dereg_ids.push_back(mr_id); });
  late_bound.sync_local_backend();

  require(late_reg_ids.size() == 1 && late_reg_ids.back() == tracked_c.mr.id,
          "sync_local_backend should register pre-existing local MR");
  require(late_bound.delete_mr(buf_c.data()),
          "late-bound local MR delete should succeed");
  require(
      late_dereg_ids.size() == 1 && late_dereg_ids.back() == tracked_c.mr.id,
      "late-bound local MR delete should trigger backend deregister");
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
              PeerTransportKind::Rdma,
          "different host_id should resolve to RDMA");

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

  cfg.preferred_transport = UKernel::Transport::PreferredTransport::Rdma;
  require(
      UKernel::Transport::resolve_peer_transport_kind(cfg, local, same) ==
          PeerTransportKind::Rdma,
      "preferred RDMA transport should use RDMA when peers are RDMA-capable");
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
  run_case("transport unit", "peer transport kind", test_peer_transport_kind);
}
