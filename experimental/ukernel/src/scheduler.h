#pragma once

#include "operator.h"
#include "transport.h"
#include <iostream>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace UKernel {
namespace Runtime {

enum class TaskType {
  COMM,
  COMPUTE,
};

struct Task {
  uint64_t id;
  TaskType type;

  // dependency
  std::vector<uint64_t> deps;
  int indegree = 0;

  // execution meta
  int tile_offset = 0;
  int tile_count = 0;

  // back reference
  uint64_t op_id;
};

// Fake backend
struct Executor {
  static void run_compute(Task const& task, Operator const& op) {
    std::cout << "[COMPUTE]"
              << " task=" << task.id << " op=" << op.id
              << " tile_offset=" << task.tile_offset
              << " tile_count=" << task.tile_count
              << " inputs=" << op.inputs.size()
              << " outputs=" << op.outputs.size() << std::endl;
  }

  static void run_comm(Task const& task, Operator const& op) {
    std::cout << "[COMM]"
              << " task=" << task.id << " op=" << op.id
              << " op_type=" << static_cast<int>(op.type)
              << " inputs=" << op.inputs.size()
              << " outputs=" << op.outputs.size() << std::endl;
  }
};

struct SchedulerConfig {
  // TODO:
  int dummy = 0;
};

class Scheduler {
 public:
  static Scheduler& instance() {
    static Scheduler inst;
    return inst;
  }

  void init(SchedulerConfig cfg);
  bool is_initialized() const;

  void reset();

  // enqueue
  void add_operator(Operator const& op);
  // execution
  void run();

  void sync(uint64_t op_id);
  void sync_all();
  bool poll(uint64_t op_id) const;

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
};

}  // namespace Runtime
}  // namespace UKernel
