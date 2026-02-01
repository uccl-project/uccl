#pragma once

#include "operator.h"
#include "transport.h"
#include <iostream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace UKernel {
namespace Runtime {

enum class TaskType { COMM, COMPUTE };
enum class CommKind { SEND, RECV };

struct CommTaskPayload {
  CommKind kind;
  int peer;
  bool on_gpu;
  void* ptr;
  size_t offset;
  size_t len;
  void* mr_base_ptr;
  size_t mr_base_len;
};

struct ComputeTaskPayload {
  // TODO:
};

using TaskPayload =
    std::variant<std::monostate, CommTaskPayload, ComputeTaskPayload>;

struct Task {
  uint64_t id;
  TaskType type;
  std::vector<uint64_t> deps;
  int indegree = 0;

  int tile_offset = 0;
  int tile_count = 0;

  uint64_t op_id = 0;
  TaskPayload payload;
};

// Fake backend
struct Executor {
  static void run_compute(Task const& task) {
    std::cout << "[COMPUTE]"
              << " task=" << task.id << " op=" << task.op_id
              << " tile_offset=" << task.tile_offset
              << " tile_count=" << task.tile_count << std::endl;
  }
};

struct SchedulerConfig {
  int gpu_id = 0;
  int rank = 0;
  int world_size = 1;
  std::shared_ptr<UKernel::Transport::CommunicatorConfig> comm_cfg =
      std::make_shared<UKernel::Transport::CommunicatorConfig>();
};

class Scheduler {
 public:
  static Scheduler& instance() {
    static Scheduler inst;
    return inst;
  }

  void init(SchedulerConfig cfg);
  bool is_initialized() const;

  // Reset scheduler graph/task state only.
  // Does NOT touch communicator, connections, or initialized_ flag.
  void reset();

  // enqueue
  void add_operator(Operator const& op);
  // execution
  void run();

  void sync(uint64_t op_id);
  void sync_all();
  bool poll(uint64_t op_id) const;

  void connect_to(int peer_rank);
  void accept_from(int peer_rank);

 private:
  Scheduler() = default;
  ~Scheduler() = default;
  Scheduler(Scheduler const&) = delete;
  Scheduler& operator=(Scheduler const&) = delete;

  TaskType task_type_for_op(Operator const& op) const;
  void expand_operator(Operator const& op);
  void on_task_finish(uint64_t tid);

  // execute a single ready task
  bool step_one();
  bool idle() const;

  // Transport
  UKernel::Transport::MR get_or_reg_local_mr_for_buffer_(void* ptr, size_t len);
  void notify_local_mr_once_(int peer, UKernel::Transport::MR const& mr);
  UKernel::Transport::MR get_remote_default_mr_(int peer);
  void run_comm(Task const& task);

 private:
  bool initialized_ = false;
  SchedulerConfig cfg_{};

  std::unordered_map<uint64_t, Operator> ops_;
  std::unordered_map<uint64_t, Task> tasks_;

  std::queue<uint64_t> ready_compute_;
  std::queue<uint64_t> ready_comm_;

  uint64_t next_task_id_ = 0;
  size_t pending_tasks_ = 0;

  std::unordered_map<uint64_t, int> op_remaining_tasks_;
  std::unordered_set<uint64_t> completed_ops_;

  // Transport
  std::shared_ptr<UKernel::Transport::Communicator> comm_;
  // peer -> default remote mr id
  std::unordered_map<int, uint16_t> peer_default_remote_mr_id_;
  // (peer, local_mr_id), if already notified
  struct NotifiedKey {
    int peer;
    decltype(UKernel::Transport::MR::id) mr_id;
    bool operator==(NotifiedKey const& o) const noexcept {
      return peer == o.peer && mr_id == o.mr_id;
    }
  };
  struct NotifiedKeyHash {
    size_t operator()(NotifiedKey const& k) const noexcept {
      return std::hash<int>{}(k.peer) ^ (std::hash<uint16_t>{}(k.mr_id) << 1);
    }
  };
  std::unordered_set<NotifiedKey, NotifiedKeyHash> notified_local_mr_;
};

}  // namespace Runtime
}  // namespace UKernel
