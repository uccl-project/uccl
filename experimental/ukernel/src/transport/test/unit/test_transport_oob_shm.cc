#include "oob/oob.h"
#include "adapter/shmring_exchanger.h"
#include "test.h"
#include "test_utils.h"
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using ShmRingExchanger = UKernel::Transport::ShmRingExchanger;
using ShmExchanger = UKernel::Transport::ShmExchanger;
using Exchanger = UKernel::Transport::Exchanger;
using CommunicatorMeta = UKernel::Transport::CommunicatorMeta;

namespace {

using UKernel::Transport::TestUtil::require;
using UKernel::Transport::TestUtil::run_case;

std::string unique_shm_namespace(char const* prefix) {
  std::ostringstream oss;
  oss << prefix << "-" << static_cast<long long>(::getpid()) << "-"
      << std::chrono::steady_clock::now().time_since_epoch().count();
  return oss.str();
}

std::string unique_posix_shm_name(char const* prefix) {
  std::ostringstream oss;
  oss << "/" << prefix << "-" << static_cast<long long>(::getpid()) << "-"
      << std::chrono::steady_clock::now().time_since_epoch().count();
  return oss.str();
}

bool contains_key(
    std::vector<std::pair<std::string, std::string>> const& entries,
    std::string const& key) {
  for (auto const& entry : entries) {
    if (entry.first == key) return true;
  }
  return false;
}

void test_shm_exchanger_put_wait_and_relay_state() {
  std::string const ns = unique_posix_shm_name("oob-kv-shm");
  ShmExchanger writer(ns);
  ShmExchanger reader(ns);
  require(writer.valid(), "writer shm exchanger should be valid");
  require(reader.valid(), "reader shm exchanger should be valid");

  CommunicatorMeta local{};
  local.host_id = "writer";
  local.ip = "127.0.0.1";
  local.local_id = 0;
  local.rdma_capable = true;
  require(writer.put("meta:local", local), "writer put(meta:local) failed");

  CommunicatorMeta fetched{};
  require(reader.get("meta:local", fetched), "reader get(meta:local) failed");
  require(fetched.host_id == "writer", "reader fetched host_id mismatch");

  std::vector<std::pair<std::string, std::string>> unrelayed;
  require(writer.collect_unrelayed(unrelayed, 16) > 0,
          "writer should collect at least one unrelayed entry");
  require(contains_key(unrelayed, "meta:local"),
          "collect_unrelayed should include meta:local");
  require(writer.mark_relayed("meta:local"),
          "mark_relayed(meta:local) should succeed");
  require(writer.collect_unrelayed(unrelayed, 16) == 0,
          "meta:local should not stay unrelayed after mark");

  CommunicatorMeta remote{};
  remote.host_id = "remote";
  remote.ip = "10.0.0.2";
  remote.local_id = 1;
  remote.rdma_capable = false;
  std::string encoded_remote;
  require(UKernel::Transport::serialize_object(remote, encoded_remote),
          "serialize remote meta failed");
  require(reader.apply_remote("meta:remote", encoded_remote),
          "apply_remote(meta:remote) failed");

  require(writer.collect_unrelayed(unrelayed, 16) == 0,
          "apply_remote data should be marked relayed");

  CommunicatorMeta waited{};
  bool delayed_put_ok = true;
  std::thread delayed_put([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    CommunicatorMeta delayed{};
    delayed.host_id = "delayed";
    delayed.ip = "127.0.0.1";
    delayed.local_id = 9;
    delayed.rdma_capable = false;
    delayed_put_ok = writer.put("meta:delayed", delayed);
  });
  require(reader.wait("meta:delayed", waited, Exchanger::WaitOptions{120, 10}),
          "reader wait(meta:delayed) failed");
  delayed_put.join();
  require(delayed_put_ok, "writer put(meta:delayed) failed");
  require(waited.host_id == "delayed", "waited payload host_id mismatch");
}

void test_shm_ack_exchange() {
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
      ShmRingExchanger shm0(/*self_rank=*/0, /*world_size=*/2, ring_namespace);

      require(shm0.accept_from(/*peer_rank=*/1, /*timeout_ms=*/5000),
              "rank0 accept_from(1) failed");

      uint32_t status = 0;
      uint64_t seq = 0;
      require(shm0.recv_ack(/*peer_rank=*/1, &status, &seq, /*timeout_ms=*/5000,
                            /*expected_seq=*/42),
              "rank0 recv_ack(1) failed");

      require(seq == 42, "rank0 recv sequence mismatch");
      require(status == 0xABCD1234UL, "rank0 recv status mismatch");
      mark_rank0_finished();
    } catch (...) {
      rank0_error = std::current_exception();
      mark_rank0_finished();
    }
  });

  std::thread rank1([&]() {
    try {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));

      ShmRingExchanger shm1(/*self_rank=*/1, /*world_size=*/2, ring_namespace);

      require(shm1.connect_to(/*peer_rank=*/0, /*timeout_ms=*/5000),
              "rank1 connect_to(0) failed");

      require(shm1.send_ack(/*peer_rank=*/0, /*seq=*/42, /*status=*/0xABCD1234UL),
              "rank1 send_ack failed");

      std::unique_lock<std::mutex> lk(done_mu);
      bool done = done_cv.wait_for(lk, std::chrono::seconds(5),
                                   [&] { return rank0_finished; });
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
  run_case("transport unit", "shm exchanger put/wait/relay-state",
           test_shm_exchanger_put_wait_and_relay_state);
  run_case("transport unit", "shm oob ack exchange",
            test_shm_ack_exchange);
}
