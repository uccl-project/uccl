"""Run the production proxy adapter/loops with CPU transport doubles.

This compiles the adapter and thread-loop bodies from engine.cc, together with
the real scheduler. It verifies the production wiring without Linux/GPU SDKs;
it is not a substitute for compiling the complete engine or hardware tests.
"""

import os
from pathlib import Path
import shutil
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

STUBS = r"""
#include "util/proxy_scheduler.h"
#include <array>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>

#define UCCL_CHECK(value) assert(value)
#define UCCL_LOG(...) std::cerr
namespace uccl {
inline void pin_thread_to_numa(int) {}
namespace device { struct Backend { static constexpr bool staged_network = false; }; }
}
enum class TaskType { WRITE_NET, READ_NET, WRITEV, READV, OTHER };
enum class SendType { Write, Read };
struct FifoItem { uint64_t addr; size_t size; };
struct MRArray {};
struct P2PMhandle {
  MRArray mr_array;
  bool compress_only = false;
  std::shared_ptr<int> compress_ctx;
  uint64_t label = 0;
};
struct MR { uint64_t mr_id_; P2PMhandle* mhandle_; };
struct Conn { struct { uint64_t peer_id; } uccl_conn_id_; };
struct UnifiedTask;
struct TransferStatus {
  std::atomic<bool> done{false};
  bool success = true;
  std::shared_ptr<UnifiedTask> task_ptr;
};
struct TaskBatch {
  size_t num_iovs = 0;
  std::shared_ptr<std::vector<void*>> data_ptr;
  std::shared_ptr<std::vector<size_t>> size_ptr;
  std::shared_ptr<std::vector<uint64_t>> mr_id_ptr;
  std::shared_ptr<std::vector<FifoItem>> slot_item_ptr;
};
struct UnifiedTask {
  TaskType type;
  void* data = nullptr;
  size_t size = 0;
  uint64_t conn_id = 0, mr_id = 1;
  TransferStatus* status_ptr = nullptr;
  TaskBatch batch;
  FifoItem item{};
  TaskBatch const& task_batch() const { return batch; }
  FifoItem const& slot_item() const { return item; }
};
struct SendConnection;
struct Transport {
  int tick = 0, recv_drives = 0;
  std::unordered_map<uint64_t, SendConnection*> groups;
  std::vector<std::string> events;
  std::unordered_map<uint64_t, int> releases, first_post, last_complete;
};
struct SendConnection {
  struct OneSidedBatchOp {
    void* local_addr;
    size_t size;
    MRArray const* local_mr_array;
    FifoItem const* slot_item;
  };
  struct RawBatchWait { int64_t wr_id = -1; size_t iov_count = 0; };
  Transport* transport = nullptr;
  int label = 0, next_id = 1, delay = 3;
  bool allowed = true, grouped = true, reject = false;
  std::unordered_map<int64_t, int> due;
  std::vector<size_t> posted_counts;
  bool can_use_raw_one_sided_batch(SendType, size_t) { return allowed; }
  bool post_write_or_read_batch(SendType, OneSidedBatchOp const* ops,
      size_t count, int64_t* ids, RawBatchWait* waits, size_t* num_waits) {
    if (reject) return false;
    transport->events.push_back("post" + std::to_string(label));
    posted_counts.push_back(count);
    transport->first_post.emplace(label, transport->tick);
    for (size_t i = 0; i < count; ++i) {
      assert(ops[i].local_addr && ops[i].local_mr_array && ops[i].slot_item);
      ids[i] = next_id++;
      due[ids[i]] = transport->tick + delay;
      if (!grouped) waits[i] = {ids[i], 1};
    }
    if (grouped) waits[0] = {ids[count - 1], count};
    *num_waits = grouped ? 1 : count;
    return true;
  }
};
inline SendConnection* uccl_resolve_send_group(Transport* transport, uint64_t id) {
  auto it = transport->groups.find(id);
  return it == transport->groups.end() ? nullptr : it->second;
}
inline bool uccl_check_wr_fast(SendConnection* group, int64_t id) {
  bool done = group->transport->tick >= group->due.at(id);
  if (done) group->transport->last_complete[group->label] = group->transport->tick;
  return done;
}
inline void uccl_drive_send(Transport* transport) { ++transport->tick; }
inline void uccl_drive_recv(Transport* transport) { ++transport->recv_drives; }
inline void uccl_deregmr(Transport* transport, P2PMhandle* handle) {
  ++transport->releases[handle->label];
}
struct ChunkSplitStrategy {
  static size_t get_message_chunk_count(size_t bytes) {
    return bytes == 0 ? 0 : std::min<size_t>(16, (bytes + 524287) / 524288);
  }
};
struct Ring {
  std::deque<UnifiedTask*> items;
  std::atomic<bool>* stop_on_empty = nullptr;
};
inline int jring_sc_dequeue_bulk(Ring* ring, void* output, int, void*) {
  if (ring->items.empty()) {
    if (ring->stop_on_empty) ring->stop_on_empty->store(true);
    return 0;
  }
  auto* task = ring->items.front();
  ring->items.pop_front();
  std::memcpy(output, &task, sizeof(task));
  return 1;
}
struct Sleeper {
  int sleeps = 0;
  void update_timer() {}
  void maybe_sleep() { ++sleeps; }
};
struct Endpoint {
  static constexpr size_t kMaxInflightOps = 256;
  static size_t max_one_sided_inflight_ops() { return 8; }
  struct ProxyTaskState;
  struct CooperativeProxyBackend;
  void cooperative_proxy_thread_func(bool);
  void send_proxy_thread_func();
  void recv_proxy_thread_func();
  std::shared_mutex conn_mu_, mr_mu_;
  std::unordered_map<uint64_t, Conn*> conn_id_to_conn_;
  std::unordered_map<uint64_t, std::shared_ptr<MR>> mr_id_to_mr_;
  std::shared_ptr<MR> make_mr(uint64_t, P2PMhandle*);
  bool dereg(uint64_t);
  ~Endpoint() { mr_id_to_mr_.clear(); }
  Transport transport;
  Transport* ep_ = &transport;
  int numa_node_ = 0, fallbacks = 0;
  bool simulate_blocking = false;
  std::atomic<bool> stop_{false};
  Ring send_ring, recv_ring;
  Ring* send_unified_task_ring_ = &send_ring;
  Ring* recv_unified_task_ring_ = &recv_ring;
  Sleeper send_proxy_adaptive_sleeper_, recv_proxy_adaptive_sleeper_;
  bool registration_unlocked() {
    bool available = false;
    std::thread probe([&] {
      available = mr_mu_.try_lock();
      if (available) mr_mu_.unlock();
    });
    probe.join();
    return available;
  }
  bool fallback() {
    // No pending or active request may hold the registry mutex across DMA.
    assert(registration_unlocked());
    ++fallbacks;
    transport.events.push_back("exclusive");
    return true;
  }
  bool write(uint64_t, uint64_t, void*, size_t, FifoItem const&) { return fallback(); }
  bool read(uint64_t, uint64_t, void*, size_t, FifoItem const&) { return fallback(); }
  bool blocking_vector(uint64_t conn, size_t count) {
    if (simulate_blocking) {
      auto* group = ep_->groups.at(conn);
      ep_->first_post.emplace(conn, ep_->tick);
      ep_->tick += group->delay * ((count + 7) / 8);
      ep_->last_complete[conn] = ep_->tick;
    }
    return fallback();
  }
  bool writev(uint64_t conn, std::vector<uint64_t> const&, std::vector<void*> const&,
      std::vector<size_t> const&, std::vector<FifoItem> const&, size_t count) {
    return blocking_vector(conn, count);
  }
  bool readv(uint64_t conn, std::vector<uint64_t> const&, std::vector<void*> const&,
      std::vector<size_t> const&, std::vector<FifoItem> const&, size_t count) {
    return blocking_vector(conn, count);
  }
};
"""

CHECKS = r"""
struct Fixture {
  Endpoint endpoint;
  P2PMhandle* handle;
  Conn first_conn{{1}}, second_conn{{2}};
  SendConnection first, second;
  Fixture() {
    first.transport = second.transport = &endpoint.transport;
    first.label = 1; second.label = 2;
    endpoint.transport.groups = {{1, &first}, {2, &second}};
    endpoint.conn_id_to_conn_ = {{1, &first_conn}, {2, &second_conn}};
    handle = add_mr(1);
  }
  P2PMhandle* add_mr(uint64_t id) {
    auto* result = new P2PMhandle;
    result->label = id;
    std::unique_lock<std::shared_mutex> lock(endpoint.mr_mu_);
    endpoint.mr_id_to_mr_[id] = endpoint.make_mr(id, result);
    return result;
  }
};
struct Request {
  std::shared_ptr<UnifiedTask> task;
  std::unique_ptr<TransferStatus> status;
};
Request request(TaskType type, uint64_t conn, std::vector<size_t> sizes,
                uintptr_t local = 0x1000000, uint64_t remote = 0x2000000) {
  Request r{std::make_shared<UnifiedTask>(), std::make_unique<TransferStatus>()};
  r.task->type = type; r.task->conn_id = conn;
  r.task->data = reinterpret_cast<void*>(local);
  r.task->size = sizes.empty() ? 0 : sizes[0];
  r.task->item = {remote, r.task->size};
  r.task->status_ptr = r.status.get();
  r.status->task_ptr = r.task;
  auto& batch = r.task->batch;
  batch.num_iovs = sizes.size();
  batch.size_ptr = std::make_shared<std::vector<size_t>>(sizes);
  batch.data_ptr = std::make_shared<std::vector<void*>>();
  batch.mr_id_ptr = std::make_shared<std::vector<uint64_t>>(sizes.size(), 1);
  batch.slot_item_ptr = std::make_shared<std::vector<FifoItem>>();
  for (size_t size : sizes) {
    batch.data_ptr->push_back(reinterpret_cast<void*>(local));
    batch.slot_item_ptr->push_back({remote, size});
    local += size; remote += size;
  }
  return r;
}
void pins_and_counts() {
  Fixture f;
  Endpoint::CooperativeProxyBackend backend{f.endpoint, true};
  auto a = request(TaskType::WRITEV, 1, {524288, 524288, 524288});
  auto b = request(TaskType::WRITEV, 2, {4096, 4096}, 0x4000000, 0x5000000);
  auto sa = backend.prepare(a.task.get());
  assert(sa->raw && f.endpoint.registration_unlocked());
  auto pa = backend.post(sa, 0, 16);
  assert(pa.posted == 2 && pa.waits.size() == 1 && pa.waits[0].iov_count == 2);
  assert(f.endpoint.registration_unlocked());
  auto sb = backend.prepare(b.task.get());
  f.second.grouped = false;
  auto pb = backend.post(sb, 0, 16);
  assert(pb.posted == 2 && pb.waits.size() == 2);
  assert(pb.waits[0].iov_count == 1 && pb.waits[1].iov_count == 1);
  assert(sa->registrations[0] == sb->registrations[0]);
  for (int i = 0; i < 4; ++i) backend.drive();
  assert(backend.poll(sa, pa.waits[0]) == uccl::ProxyCompletion::Complete);
  backend.finish(sa, true);
  assert(a.status->done && f.endpoint.registration_unlocked());
  backend.finish(sb, true);
  assert(b.status->done && f.endpoint.registration_unlocked());
}
void conflicts_and_eligibility() {
  Fixture f;
  Endpoint::CooperativeProxyBackend backend{f.endpoint, true};
  auto a = request(TaskType::WRITEV, 1, {4096});
  auto same = request(TaskType::WRITEV, 1, {4096}, 0x9000000, 0xa000000);
  auto overlap = request(TaskType::WRITEV, 2, {4096}, 0x4000000, 0x2000800);
  auto adjacent = request(TaskType::WRITEV, 2, {4096}, 0x1000000, 0x2001000);
  auto sa = backend.prepare(a.task.get());
  assert(backend.conflicts(sa, backend.prepare(same.task.get())));
  assert(backend.conflicts(sa, backend.prepare(overlap.task.get())));
  assert(!backend.conflicts(sa, backend.prepare(adjacent.task.get())));
  auto empty = request(TaskType::WRITEV, 2, {});
  auto large = request(TaskType::WRITEV, 2, {524289});
  auto overflow = request(TaskType::WRITEV, 2, {4096}, 0x1000000, UINT64_MAX - 10);
  assert(!backend.prepare(empty.task.get())->raw);
  assert(!backend.prepare(large.task.get())->raw);
  assert(!backend.prepare(overflow.task.get())->raw);
  f.handle->compress_only = true;
  assert(!backend.prepare(a.task.get())->raw);
  f.handle->compress_only = false;
  f.first.allowed = false;
  assert(!backend.prepare(a.task.get())->raw);
  f.first.allowed = true;
  Endpoint::CooperativeProxyBackend reader{f.endpoint, false};
  auto ra = request(TaskType::READV, 1, {4096});
  auto rb = request(TaskType::READV, 2, {4096}, 0x1000800, 0x9000000);
  auto rc = request(TaskType::READV, 2, {4096}, 0x1001000, 0x2000000);
  assert(reader.conflicts(reader.prepare(ra.task.get()), reader.prepare(rb.task.get())));
  assert(!reader.conflicts(reader.prepare(ra.task.get()), reader.prepare(rc.task.get())));
}
void deregistered_while_queued() {
  Fixture f;
  Endpoint::CooperativeProxyBackend backend{f.endpoint, true};
  auto a = request(TaskType::WRITEV, 1, {4096});
  auto state = backend.prepare(a.task.get());
  assert(f.endpoint.dereg(1));
  auto result = backend.post(state, 0, 1);
  assert(result.status == uccl::ProxyPostStatus::Failed && result.posted == 0);
  assert(f.first.posted_counts.empty());
  backend.finish(state, false);
  assert(a.status->done && !a.status->success);
}
void exclusive_barrier() {
  Fixture f;
  Endpoint::CooperativeProxyBackend backend{f.endpoint, true};
  uccl::ProxyScheduler<Endpoint::CooperativeProxyBackend::Task> scheduler;
  auto a = request(TaskType::WRITEV, 1, {4096, 4096});
  auto barrier = request(TaskType::WRITE_NET, 1, {4096});
  auto b = request(TaskType::WRITEV, 2, {4096}, 0x4000000, 0x5000000);
  for (Request* r : {&a, &barrier, &b}) {
    auto state = backend.prepare(r->task.get());
    assert(scheduler.enqueue(state, state->num_iovs, !state->raw));
  }
  for (int i = 0; i < 30 && !scheduler.empty(); ++i) scheduler.progress(backend);
  assert(scheduler.empty() && f.endpoint.fallbacks == 1);
  assert((f.endpoint.transport.events == std::vector<std::string>{"post1", "exclusive", "post2"}));
  assert(a.status->done && barrier.status->done && b.status->done);
}
void entry_and_shutdown(bool sending, bool enabled) {
  Fixture f;
  auto a = request(sending ? TaskType::WRITEV : TaskType::READV, 1,
                   std::vector<size_t>(12, 4096));
  auto b = request(sending ? TaskType::WRITEV : TaskType::READV, 2,
                   {4096}, 0x4000000, 0x5000000);
  Ring& ring = sending ? f.endpoint.send_ring : f.endpoint.recv_ring;
  ring.items = {a.task.get(), b.task.get()};
  // Request shutdown as soon as admission runs out of queued tasks. The
  // cooperative loop still has to post/drain the accepted requests.
  ring.stop_on_empty = &f.endpoint.stop_;
  if (enabled) setenv("UCCL_P2P_COOPERATIVE_PROXY", "1", 1);
  else unsetenv("UCCL_P2P_COOPERATIVE_PROXY");
  if (sending) f.endpoint.send_proxy_thread_func();
  else f.endpoint.recv_proxy_thread_func();
  assert(a.status->done && b.status->done);
  assert(a.status->success && b.status->success);
  if (enabled) {
    assert(f.endpoint.fallbacks == 0 && f.first.posted_counts.size() >= 2);
    assert(f.endpoint.transport.tick >= f.first.delay);
    assert(f.endpoint.send_proxy_adaptive_sleeper_.sleeps == 0);
    assert(f.endpoint.recv_proxy_adaptive_sleeper_.sleeps == 0);
    if (!sending) assert(f.endpoint.transport.recv_drives > 0);
  } else {
    assert(f.endpoint.fallbacks == 2 && f.first.posted_counts.empty());
  }
  assert(f.endpoint.registration_unlocked());
}
void independent_mr_release() {
  Fixture f;
  f.add_mr(2);
  Endpoint::CooperativeProxyBackend backend{f.endpoint, true};
  auto a = request(TaskType::WRITEV, 1, {4096});
  auto b = request(TaskType::WRITEV, 2, {4096}, 0x4000000, 0x5000000);
  (*b.task->batch.mr_id_ptr)[0] = 2;
  auto sa = backend.prepare(a.task.get());
  auto sb = backend.prepare(b.task.get());
  f.second.delay = 100;
  auto pa = backend.post(sa, 0, 1);
  auto pb = backend.post(sb, 0, 1);
  assert(f.endpoint.dereg(1) && f.endpoint.dereg(2));
  assert(f.endpoint.transport.releases.empty());
  // Another thread can register/deregister while both DMA requests are live.
  std::thread writer([&] { f.add_mr(3); assert(f.endpoint.dereg(3)); });
  writer.join();
  assert(f.endpoint.transport.releases[3] == 1);
  for (int i = 0; i < 4; ++i) backend.drive();
  assert(backend.poll(sa, pa.waits[0]) == uccl::ProxyCompletion::Complete);
  assert(backend.poll(sb, pb.waits[0]) == uccl::ProxyCompletion::Pending);
  backend.finish(sa, true);
  // B staying active cannot delay release of A's separate registration.
  assert(f.endpoint.transport.releases[1] == 1);
  assert(f.endpoint.transport.releases[2] == 0);
  for (int i = 0; i < 100; ++i) backend.drive();
  backend.finish(sb, true);
  assert(f.endpoint.transport.releases[2] == 1);
}
void shared_mr_between_proxies() {
  Fixture f;
  Endpoint::CooperativeProxyBackend sender{f.endpoint, true};
  Endpoint::CooperativeProxyBackend receiver{f.endpoint, false};
  auto a = request(TaskType::WRITEV, 1, {4096});
  auto b = request(TaskType::READV, 2, {4096}, 0x4000000, 0x5000000);
  auto sa = sender.prepare(a.task.get());
  auto sb = receiver.prepare(b.task.get());
  sender.post(sa, 0, 1);
  receiver.post(sb, 0, 1);
  assert(f.endpoint.dereg(1));
  for (int i = 0; i < 4; ++i) sender.drive();
  sender.finish(sa, true);
  assert(f.endpoint.transport.releases[1] == 0);
  receiver.finish(sb, true);
  assert(f.endpoint.transport.releases[1] == 1);

  // A failed first post must also drop any pins acquired before a missing MR.
  f.add_mr(1); f.add_mr(2);
  auto partial = request(TaskType::WRITEV, 1, {4096, 4096});
  (*partial.task->batch.mr_id_ptr)[1] = 2;
  auto state = sender.prepare(partial.task.get());
  assert(f.endpoint.dereg(2));
  auto failed = sender.post(state, 0, 2);
  assert(failed.status == uccl::ProxyPostStatus::Failed && failed.posted == 0);
  assert(f.endpoint.dereg(1));
  assert(f.endpoint.transport.releases[1] == 1);
  sender.finish(state, false);
  assert(f.endpoint.transport.releases[1] == 2);
  assert(partial.status->done && !partial.status->success);
}
void null_handle_skips_transport_release() {
  // Endpoint teardown frees handles itself and nulls them before the last
  // owner drops the registration; the deleter must not touch the transport.
  Fixture f;
  f.add_mr(2);
  {
    std::unique_lock<std::shared_mutex> lock(f.endpoint.mr_mu_);
    auto& mr = f.endpoint.mr_id_to_mr_.at(2);
    delete mr->mhandle_;
    mr->mhandle_ = nullptr;
    f.endpoint.mr_id_to_mr_.erase(2);
  }
  assert(f.endpoint.transport.releases.count(2) == 0);
  assert(f.endpoint.dereg(1) && f.endpoint.transport.releases[1] == 1);
}
std::pair<int, int> run_production_loop(bool enabled) {
  Fixture f;
  f.endpoint.simulate_blocking = true;
  f.first.delay = 10;
  f.second.delay = 1;
  auto large = request(TaskType::WRITEV, 1, std::vector<size_t>(12, 4096));
  auto small = request(TaskType::WRITEV, 2, {4096}, 0x4000000, 0x5000000);
  f.endpoint.send_ring.items = {large.task.get(), small.task.get()};
  f.endpoint.send_ring.stop_on_empty = &f.endpoint.stop_;
  if (enabled) setenv("UCCL_P2P_COOPERATIVE_PROXY", "1", 1);
  else unsetenv("UCCL_P2P_COOPERATIVE_PROXY");
  f.endpoint.send_proxy_thread_func();
  assert(large.status->done && small.status->done);
  return {f.endpoint.transport.first_post.at(2),
          f.endpoint.transport.last_complete.at(1)};
}
void production_loop_head_of_line() {
  // Both runs use the real send_proxy_thread_func. Only blocking transport
  // I/O and asynchronous hardware completions use the deterministic clock.
  auto legacy = run_production_loop(false);
  auto cooperative = run_production_loop(true);
  assert(legacy.first >= legacy.second);
  assert(cooperative.first < cooperative.second);
  std::cout << "fake ticks: legacy small_first=" << legacy.first
            << " large_done=" << legacy.second
            << "; cooperative small_first=" << cooperative.first
            << " large_done=" << cooperative.second << '\n';
}
int main(int argc, char** argv) {
  assert(argc == 2);
  switch (std::atoi(argv[1])) {
    case 0: pins_and_counts(); break;
    case 1: conflicts_and_eligibility(); break;
    case 2: deregistered_while_queued(); break;
    case 3: exclusive_barrier(); break;
    case 4: entry_and_shutdown(true, true); break;
    case 5: entry_and_shutdown(false, true); break;
    case 6: entry_and_shutdown(true, false); break;
    case 7: independent_mr_release(); break;
    case 8: shared_mr_between_proxies(); break;
    case 9: production_loop_head_of_line(); break;
    case 10: null_handle_skips_transport_release(); break;
    default: return 1;
  }
}
"""


class CooperativeProxyAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = os.environ.get("CXX", "c++")
        if not shutil.which(compiler):
            raise RuntimeError(f"C++ compiler unavailable: {compiler}")
        engine = (ROOT / "p2p/engine.cc").read_text()
        helpers_start = engine.index("static inline bool raw_one_sided_batch_eligible(")
        helpers_end = engine.index(
            "size_t Endpoint::max_one_sided_inflight_ops()", helpers_start
        )
        start = engine.index("struct Endpoint::ProxyTaskState {")
        end = engine.index("void Endpoint::passive_accept_thread_func()", start)
        lifetime_start = engine.index("std::shared_ptr<MR> Endpoint::make_mr(")
        lifetime_end = engine.index("bool Endpoint::read(", lifetime_start)
        source = (
            STUBS
            + engine[helpers_start:helpers_end]
            + engine[lifetime_start:lifetime_end]
            + engine[start:end]
            + CHECKS
        )
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.executable = Path(cls.directory.name) / "adapter-test"
        command = (
            [compiler, "-std=c++17", "-pthread"]
            + shlex.split(os.environ.get("CXXFLAGS", ""))
            + ["-I", str(ROOT / "p2p"), "-x", "c++", "-", "-o", str(cls.executable)]
            + shlex.split(os.environ.get("LDFLAGS", ""))
        )
        result = subprocess.run(
            command, input=source, text=True, capture_output=True, timeout=60
        )
        if result.returncode:
            raise AssertionError(result.stderr)

    def test_production_adapter_and_thread_loops(self):
        cases = (
            "MR pins, byte quantum, IB/EFA completion counts",
            "same connection and write-range conflicts, fallback eligibility",
            "registration removed before first post",
            "exclusive barrier has no retained registry locks",
            "opt-in send entry drains accepted work on stop",
            "opt-in receive entry drains accepted work on stop",
            "default entry retains the original blocking path",
            "independent MR releases and concurrent registration progress",
            "shared MR pins across proxies and failed pin acquisition",
            "production old/new send loops differ on head-of-line blocking",
            "teardown-style null handle skips transport release",
        )
        for index, case in enumerate(cases):
            with self.subTest(case=case):
                result = subprocess.run(
                    [str(self.executable), str(index)],
                    text=True,
                    capture_output=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                print(f"PASS: {case}")
                if result.stdout:
                    print(result.stdout, end="")


if __name__ == "__main__":
    unittest.main()
