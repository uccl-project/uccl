#include "scheduler.h"

namespace UKernel {
namespace Runtime {

TaskType Scheduler::task_type_for_op(Operator const& op) const {
  switch (op.type) {
    case OpType::P2P:
    case OpType::Collective:
      return TaskType::COMM;

    case OpType::Compute:
    case OpType::Moe:
      return TaskType::COMPUTE;
  }
  return TaskType::COMPUTE;
}

void Scheduler::add_operator(Operator const& op) {
  ops_[op.id] = op;
  expand_operator(op);
}

void Scheduler::expand_operator(Operator const& op) {
  auto const& rule = op.parallel_rule;

  for (int i = 0; i < rule.num_tasks; ++i) {
    Task t;
    t.id = next_task_id_++;
    t.op_id = op.id;
    t.type = task_type_for_op(op);

    t.tile_offset = i * rule.tiles_per_task;
    t.tile_count = rule.tiles_per_task;

    // operator deps → task deps
    for (auto dep_op : op.deps) {
      for (auto& [_, prev] : tasks_) {
        if (prev.op_id == dep_op) {
          t.deps.push_back(prev.id);
          t.indegree++;
        }
      }
    }

    tasks_[t.id] = t;

    if (t.indegree == 0) {
      (t.type == TaskType::COMPUTE ? ready_compute_ : ready_comm_).push(t.id);
    }
  }
}

void Scheduler::on_task_finish(uint64_t tid) {
  for (auto& [_, t] : tasks_) {
    for (auto d : t.deps) {
      if (d == tid && --t.indegree == 0) {
        (t.type == TaskType::COMPUTE ? ready_compute_ : ready_comm_).push(t.id);
      }
    }
  }
}

void Scheduler::run() {
  while (!ready_compute_.empty() || !ready_comm_.empty()) {
    if (!ready_compute_.empty()) {
      auto tid = ready_compute_.front();
      ready_compute_.pop();

      auto& task = tasks_[tid];
      auto& op = ops_[task.op_id];

      Executor::run_compute(task, op);
      on_task_finish(tid);
    }

    if (!ready_comm_.empty()) {
      auto tid = ready_comm_.front();
      ready_comm_.pop();

      auto& task = tasks_[tid];
      auto& op = ops_[task.op_id];

      Executor::run_comm(task, op);
      on_task_finish(tid);
    }
  }
}

}  // namespace Runtime
}  // namespace UKernel
