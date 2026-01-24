#pragma once

#include "operator.h"
#include "transport.h"
#include <iostream>
#include <queue>
#include <unordered_map>
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
              << " tile_count=" << task.tile_count << std::endl;
  }

  static void run_comm(Task const& task, Operator const& op) {
    std::cout << "[COMM]"
              << " task=" << task.id << " op=" << op.id
              << " op_type=" << static_cast<int>(op.type) << std::endl;
  }
};

class Scheduler {
 public:
  void add_operator(Operator const& op);
  void run();

 private:
  TaskType task_type_for_op(Operator const& op) const;
  void expand_operator(Operator const& op);
  void on_task_finish(uint64_t tid);

 private:
  std::unordered_map<uint64_t, Operator> ops_;
  std::unordered_map<uint64_t, Task> tasks_;

  std::queue<uint64_t> ready_compute_;
  std::queue<uint64_t> ready_comm_;

  uint64_t next_task_id_ = 0;
};

}  // namespace Runtime
}  // namespace UKernel
