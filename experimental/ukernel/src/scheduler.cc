#include "scheduler.h"

namespace UKernel {
namespace Runtime {

void Scheduler::init(SchedulerConfig cfg) {
  cfg_ = cfg;
  initialized_ = true;
}

bool Scheduler::is_initialized() const { return initialized_; }

void Scheduler::reset() {
  ops_.clear();
  tasks_.clear();
  while (!ready_compute_.empty()) ready_compute_.pop();
  while (!ready_comm_.empty()) ready_comm_.pop();

  next_task_id_ = 0;
  pending_tasks_ = 0;
  op_remaining_tasks_.clear();
  completed_ops_.clear();
}

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
  if (!initialized_) {
    throw std::runtime_error("Scheduler not initialized. Call init() first.");
  }

  ops_[op.id] = op;
  expand_operator(op);
}

void Scheduler::expand_operator(Operator const& op) {
  auto const& rule = op.parallel_rule;

  int created = 0;
  for (int i = 0; i < rule.num_tasks; ++i) {
    Task t;
    t.id = next_task_id_++;
    t.op_id = op.id;
    t.type = task_type_for_op(op);

    t.tile_offset = i * rule.tiles_per_task;
    t.tile_count = rule.tiles_per_task;

    // operator deps → task deps
    for (auto dep_op : op.deps) {
      for (auto const& kv : tasks_) {
        auto const& prev = kv.second;
        if (prev.op_id == dep_op) {
          t.deps.push_back(prev.id);
          t.indegree++;
        }
      }
    }

    tasks_[t.id] = t;
    pending_tasks_++;
    created++;

    if (t.indegree == 0) {
      (t.type == TaskType::COMPUTE ? ready_compute_ : ready_comm_).push(t.id);
    }
  }

  op_remaining_tasks_[op.id] += created;
  completed_ops_.erase(op.id);
}

bool Scheduler::step_one() {
  uint64_t tid;
  bool is_compute = false;

  if (!ready_compute_.empty()) {
    tid = ready_compute_.front();
    ready_compute_.pop();
    is_compute = true;
  } else if (!ready_comm_.empty()) {
    tid = ready_comm_.front();
    ready_comm_.pop();
  } else {
    return false;
  }

  auto& task = tasks_.at(tid);
  auto& op = ops_.at(task.op_id);

  if (is_compute)
    Executor::run_compute(task, op);
  else
    Executor::run_comm(task, op);

  on_task_finish(tid);
  return true;
}

void Scheduler::on_task_finish(uint64_t tid) {
  // update deps -> push newly ready tasks
  for (auto& kv : tasks_) {
    auto& t = kv.second;
    for (auto d : t.deps) {
      if (d == tid) {
        if (--t.indegree == 0) {
          (t.type == TaskType::COMPUTE ? ready_compute_ : ready_comm_)
              .push(t.id);
        }
      }
    }
  }

  pending_tasks_--;

  // per-op remaining tasks
  auto op_id = tasks_.at(tid).op_id;
  auto it = op_remaining_tasks_.find(op_id);
  if (it != op_remaining_tasks_.end()) {
    it->second--;
    if (it->second == 0) {
      completed_ops_.insert(op_id);
    }
  }
}

bool Scheduler::idle() const {
  return pending_tasks_ == 0 && ready_compute_.empty() && ready_comm_.empty();
}

void Scheduler::run() {
  if (!initialized_) {
    throw std::runtime_error("Scheduler not initialized. Call init() first.");
  }

  // run until no more ready tasks
  while (step_one()) {
  }
}

bool Scheduler::poll(uint64_t op_id) const {
  return completed_ops_.count(op_id) > 0;
}

void Scheduler::sync(uint64_t op_id) {
  if (!initialized_) {
    throw std::runtime_error("Scheduler not initialized. Call init() first.");
  }

  // drive execution until op completes or cannot make progress
  while (!poll(op_id)) {
    if (!step_one()) {
      throw std::runtime_error(
          "sync(op_id) cannot make progress: "
          "no ready tasks but op not completed.");
    }
  }
}

void Scheduler::sync_all() {
  if (!initialized_) {
    throw std::runtime_error("Scheduler not initialized. Call init() first.");
  }

  while (!idle()) {
    if (!step_one()) {
      throw std::runtime_error(
          "sync_all() cannot make progress: "
          "no ready tasks but pending tasks exist.");
    }
  }
}

}  // namespace Runtime
}  // namespace UKernel
