#include "scheduler.h"
#include <stdexcept>

namespace UKernel {
namespace Runtime {

static inline size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

static inline size_t tensor_nbytes(torch::Tensor const& t) {
  return (size_t)t.numel() * (size_t)t.element_size();
}

static inline int get_attr_i(
    std::unordered_map<std::string, std::string> const& a, std::string const& k,
    int defv) {
  auto it = a.find(k);
  return it == a.end() ? defv : std::stoi(it->second);
}
static inline size_t get_attr_z(
    std::unordered_map<std::string, std::string> const& a, std::string const& k,
    size_t defv) {
  auto it = a.find(k);
  return it == a.end() ? defv : (size_t)std::stoull(it->second);
}
static inline bool get_attr_b(
    std::unordered_map<std::string, std::string> const& a, std::string const& k,
    bool defv) {
  auto it = a.find(k);
  if (it == a.end()) return defv;
  auto const& v = it->second;
  return (v == "1" || v == "true" || v == "True");
}

void Scheduler::init(SchedulerConfig cfg) {
  cfg_ = cfg;
  comm_ = std::make_shared<UKernel::Transport::Communicator>(
      cfg.gpu_id, cfg.rank, cfg.world_size, cfg.comm_cfg);
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

  // Transport-side handshake bookkeeping (scheduler-level only).
  // Communicator keeps its own MR caches internally.
  peer_default_remote_mr_id_.clear();
  notified_local_mr_.clear();
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

  // force deps
  for (auto dep_op : op.deps) {
    if (ops_.find(dep_op) == ops_.end()) {
      throw std::runtime_error("add_operator order violation: missing dep op " +
                               std::to_string(dep_op));
    }
  }

  ops_[op.id] = op;
  expand_operator(op);
}

void Scheduler::expand_operator(Operator const& op) {
  auto const& rule = op.parallel_rule;

  std::vector<Task> built;
  built.reserve(std::max(1, rule.num_tasks));

  // TODO: op with both comm and compute
  if (task_type_for_op(op) == TaskType::COMM) {
    if (op.type != OpType::P2P || !op.p2p_kind.has_value()) {
      throw std::runtime_error("Only P2P comm supported for now");
    }

    int peer = get_attr_i(op.attrs, "peer", -1);
    if (peer < 0) throw std::runtime_error("P2P op missing attrs['peer']");

    size_t base_offset = get_attr_z(op.attrs, "offset", 0);
    bool on_gpu = get_attr_b(op.attrs, "on_gpu", true);

    CommKind ck;
    torch::Tensor buf;

    if (op.p2p_kind.value() == P2PKind::Send) {
      ck = CommKind::SEND;
      if (op.inputs.empty() || !op.inputs[0].defined())
        throw std::runtime_error("P2P Send requires inputs[0]");
      buf = op.inputs[0];
    } else {
      ck = CommKind::RECV;
      if (op.outputs.empty() || !op.outputs[0].defined())
        throw std::runtime_error("P2P Recv requires outputs[0]");
      buf = op.outputs[0];
    }

    if (on_gpu && !buf.is_cuda())
      throw std::runtime_error("on_gpu=1 but tensor not CUDA");

    void* base_ptr = buf.data_ptr();
    size_t total_len = get_attr_z(op.attrs, "len", tensor_nbytes(buf));
    if (total_len == 0) {
      completed_ops_.insert(op.id);
      op_remaining_tasks_[op.id] += 0;
      return;
    }

    int nt = std::max(1, rule.num_tasks);
    size_t chunk = ceil_div(total_len, (size_t)nt);

    for (int i = 0; i < nt; ++i) {
      size_t slice_off = (size_t)i * chunk;
      if (slice_off >= total_len) continue;
      size_t slice_len = std::min(chunk, total_len - slice_off);

      Task t;
      t.type = TaskType::COMM;
      t.op_id = op.id;
      t.tile_offset = i * rule.tiles_per_task;
      t.tile_count = rule.tiles_per_task;

      CommTaskPayload p;
      p.kind = ck;
      p.peer = peer;
      p.on_gpu = on_gpu;
      p.ptr = (void*)((uint8_t*)base_ptr + slice_off);
      p.offset = base_offset + slice_off;
      p.len = slice_len;
      p.mr_base_ptr = base_ptr;
      p.mr_base_len = total_len;

      t.payload = std::move(p);
      built.push_back(std::move(t));
    }
  } else {
    // TODO: COMPUTE tasks
    int nt = std::max(1, rule.num_tasks);
    for (int i = 0; i < nt; ++i) {
      Task t;
      t.type = TaskType::COMPUTE;
      t.op_id = op.id;
      t.tile_offset = i * rule.tiles_per_task;
      t.tile_count = rule.tiles_per_task;
      t.payload = std::monostate{};
      built.push_back(std::move(t));
    }
  }

  // attach deps + assign ids + enqueue
  int created = 0;
  for (auto& t : built) {
    for (auto dep_op : op.deps) {
      for (auto const& kv : tasks_) {
        auto const& prev = kv.second;
        if (prev.op_id == dep_op) {
          t.deps.push_back(prev.id);
          t.indegree++;
        }
      }
    }

    t.id = next_task_id_++;
    tasks_[t.id] = std::move(t);

    pending_tasks_++;
    created++;

    if (tasks_[t.id].indegree == 0) {
      (tasks_[t.id].type == TaskType::COMPUTE ? ready_compute_ : ready_comm_)
          .push(tasks_[t.id].id);
    }
  }

  op_remaining_tasks_[op.id] += created;
  if (created > 0)
    completed_ops_.erase(op.id);
  else
    completed_ops_.insert(op.id);
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

  if (is_compute)
    Executor::run_compute(task);
  else
    run_comm(task);

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

  if (pending_tasks_ > 0) {
    throw std::runtime_error(
        "run() stopped with pending tasks but no ready tasks. "
        "Likely missing deps or cyclic deps.");
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

// Transport
void Scheduler::connect_to(int peer_rank) {
  if (!initialized_ || !comm_)
    throw std::runtime_error("Scheduler not initialized");
  if (!comm_->connect_to(peer_rank))
    throw std::runtime_error("connect_to failed");
}

void Scheduler::accept_from(int peer_rank) {
  if (!initialized_ || !comm_)
    throw std::runtime_error("Scheduler not initialized");
  if (!comm_->accept_from(peer_rank))
    throw std::runtime_error("accept_from failed");
}

UKernel::Transport::MR Scheduler::get_or_reg_local_mr_for_buffer_(void* ptr,
                                                                  size_t len) {
  try {
    return comm_->get_local_mr(ptr);
  } catch (std::exception const&) {
    return comm_->reg_mr(ptr, len);
  }
}

void Scheduler::notify_local_mr_once_(int peer,
                                      UKernel::Transport::MR const& mr) {
  NotifiedKey k{peer, mr.id};
  if (notified_local_mr_.count(k)) return;

  auto tmp = mr;
  if (!comm_->notify_mr(peer, tmp)) {
    throw std::runtime_error("notify_mr failed");
  }
  notified_local_mr_.insert(k);
}

UKernel::Transport::MR Scheduler::get_remote_default_mr_(int peer) {
  auto it = peer_default_remote_mr_id_.find(peer);
  if (it != peer_default_remote_mr_id_.end()) {
    if (it->second == 0) return UKernel::Transport::MR{};  // dummy
    return comm_->get_remote_mr(peer, it->second);
  }

  UKernel::Transport::MR remote_mr;
  if (!comm_->wait_mr_notify(peer, remote_mr)) {
    throw std::runtime_error("wait_mr_notify failed");
  }

  peer_default_remote_mr_id_[peer] = remote_mr.id;
  return remote_mr;
}

void Scheduler::run_comm(Task const& task) {
  if (!comm_) throw std::runtime_error("Communicator is null");

  if (task.type != TaskType::COMM) {
    throw std::runtime_error("run_comm called with non-COMM task");
  }

  auto* p = std::get_if<CommTaskPayload>(&task.payload);
  if (!p) {
    throw std::runtime_error("COMM task payload is not CommTaskPayload");
  }

  if (p->peer < 0) throw std::runtime_error("COMM payload missing peer");
  if (!p->ptr) throw std::runtime_error("COMM payload missing ptr");
  if (!p->mr_base_ptr || p->mr_base_len == 0)
    throw std::runtime_error("COMM payload missing mr_base_ptr/mr_base_len");
  if (p->len == 0) return;

  auto local_mr =
      get_or_reg_local_mr_for_buffer_(p->mr_base_ptr, p->mr_base_len);

  notify_local_mr_once_(p->peer, local_mr);

  if (p->kind == CommKind::SEND) {
    auto remote_mr = get_remote_default_mr_(p->peer);
    unsigned req = comm_->isend(p->peer, p->ptr, p->offset, p->len, local_mr.id,
                                remote_mr.id, p->on_gpu);
    if (req == 0) throw std::runtime_error("isend failed (returned 0)");
    if (!comm_->wait_finish(req))
      throw std::runtime_error("wait_finish(send) failed");
    return;
  }

  if (p->kind == CommKind::RECV) {
    unsigned req = comm_->irecv(p->peer, p->ptr, p->offset, p->len, p->on_gpu);
    if (req == 0) throw std::runtime_error("irecv failed (returned 0)");
    if (!comm_->wait_finish(req))
      throw std::runtime_error("wait_finish(recv) failed");
    return;
  }

  throw std::runtime_error("Unknown CommKind");
}

}  // namespace Runtime
}  // namespace UKernel
