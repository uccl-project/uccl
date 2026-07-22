# CCL Executor — Dependency Tracking Design

## Problem

In the CCL executor (`SprayExecutor`), collective operations (allreduce, alltoall) are decomposed into a DAG of fine-grained ops (`Put`, `Reduce`, `Signal`, `WaitSignal`). These ops form a partial order defined by dependencies. The executor must:

1. **Track which ops are ready** — all their dependencies have completed
2. **Efficiently schedule** ready ops to device / transport / signal backends
3. **Handle concurrency** — 3 drain threads (device, transport, signal) mark ops done; 1 enqueue thread collects ready ops

### Current Implementation

```cpp
// SprayRun
std::vector<uint8_t>   done;       // op completion bitmap

// collect_ready() — runs in enqueue thread under run->mtx
for (uint32_t op : ops_in_layer) {
  if (done[op] || submitted[op]) continue;
  bool ok = true;
  for (uint32_t dep : ops[op].deps)    // ← O(N×D) scan every cycle
    if (!done[dep]) { ok = false; break; }
  if (ok) ready.push_back(op);
}

// drain_batch() — runs in drain threads under run->mtx
m.run->done[m.op_idx] = 1;
m.run->done_count.fetch_add(1);
```

**Problems**:
- `collect_ready` scans all dependencies for every op, every enqueue cycle — O(N × D)
- Both drain and enqueue contend on `run->mtx`

---

## Solution 1: Bitmask Ready Set

**Best for**: small op graphs (≤ 64 ops, typical for ring allreduce)

**Key idea**: Replace dependency scanning with a per-op indegree counter. Instead of a queue, use a 64-bit atomic bitmask. Drain threads set bits. Enqueue thread reads and clears the whole mask in one atomic exchange.

### Data Layout

```cpp
struct SprayRun {
  std::vector<std::atomic<uint32_t>> indegree;  // remaining deps per op
  std::atomic<uint64_t> ready_mask{0};           // bitmask of ready ops
};
```

### Operations

**drain — per-op completion (under run->mtx)**:
```cpp
// For each successor of the completed op:
for (uint32_t succ : successors[op]) {
  if (indegree[succ].fetch_sub(1, std::memory_order_release) == 1)
    ready_mask.fetch_or(1ull << succ, std::memory_order_release);
}
```

**enqueue — collect_ready (under run->mtx)**:
```cpp
uint64_t batch = ready_mask.exchange(0, std::memory_order_acquire);
while (batch) {
  int op = __builtin_ctzll(batch);       // find lowest set bit
  batch &= batch - 1;                     // clear it
  if (!submitted[op]) ready.push_back(op);
}
```

### Analysis

| Aspect | Detail |
|--------|--------|
| Enqueue cost | O(ready_count) — one atomic exchange + popcount iterations |
| Drain cost | O(successor_count) — one atomic fetch_sub + optional fetch_or per successor |
| Locking | `run->mtx` still held by both paths (small scope) |
| Memory | 8 bytes for ready_mask + 4 bytes per op for indegree |
| Allocation | Zero — all stack/register |
| Per-op instructions | Drain: 2 atomics (fetch_sub + maybe fetch_or). Enqueue: 1 atomic (exchange) |
| Limitation | 64 ops max (allreduce ring: ~8 ops; alltoall pairwise DMA: ~44 ops for 8-rank) |
| Extendability | For > 64 ops, use multiple uint64_t (e.g., `std::array<std::atomic<uint64_t>, N>`) or fall back to Solution 2 |

### Pros
- Minimal code, zero allocation
- 2 atomic instructions total per drain cycle
- `collect_ready` reduced from O(N×D) scanning to O(ready_count)

### Cons
- Hard 64-op limit
- Still uses `run->mtx` for both paths

---

## Solution 2: Flat Array + fetch_sub (General Purpose)

**Best for**: arbitrary op graph sizes, minimal code change

**Key idea**: Same indegree counter model, but use a flat successor array and a `ready_batch` vector. Drain decrements indegree via `fetch_sub`. When indegree hits 0, push to `ready_batch`. Enqueue grabs `ready_batch`. Both under `run->mtx` — same lock model as current.

### Data Layout

```cpp
struct SprayRun {
  std::vector<uint32_t>          successor_data;  // contiguous successor list
  std::vector<uint32_t>          successor_off;   // offset into data per op
  std::vector<std::atomic<uint32_t>> indegree;    // fetch_sub decrement
  std::vector<uint32_t>          ready_batch;     // accumulated ready (under mtx)
};
```

Initialization:
```cpp
// Build flat successor table
successor_off.resize(nops + 1);
for (uint32_t i = 0; i < nops; ++i) {
  successor_off[i] = successor_data.size();
  for (uint32_t dep : ops[i].deps)
    successor_data.push_back(i);   // op i is a successor of dep
  successor_off[i + 1] = successor_data.size();
}
indegree.resize(nops);
for (uint32_t i = 0; i < nops; ++i)
  indegree[i].store(ops[i].deps.size(), std::memory_order_relaxed);
```

### Operations

**drain — per-op completion (under run->mtx)**:
```cpp
uint32_t off = successor_off[op];
uint32_t end = successor_off[op + 1];
for (uint32_t j = off; j < end; ++j) {
  uint32_t succ = successor_data[j];
  if (indegree[succ].fetch_sub(1, std::memory_order_release) == 1)
    ready_batch.push_back(succ);
}
```

**enqueue — collect_ready (under run->mtx)**:
```cpp
ready = std::move(ready_batch);   // steal accumulated ready ops
ready_batch.clear();
// Filter out already-submitted
for (uint32_t op : ready)
  if (!submitted[op]) run.ready.push_back(op);
```

### Analysis

| Aspect | Detail |
|--------|--------|
| Enqueue cost | O(ready_count) — iterate ready_batch |
| Drain cost | O(successor_count) — cache-friendly linear scan + atomic per successor |
| Locking | `run->mtx` — same as current, no new locking |
| Memory | ~8 bytes per op for successor_off + ~4 bytes per dependency edge for successor_data + 4 bytes per op for indegree |
| Allocation | One-time — successor arrays built at submit time |
| Limitation | None — works for any graph size |

### Pros
- Works for arbitrary graph sizes
- Cache-friendly flat arrays
- `collect_ready` reduced from O(N×D) to O(ready_count)
- Minimal code change — same lock model

### Cons
- Still uses `run->mtx` for both paths
- Extra memory for flat successor table (but dep edges already exist in `ops[i].deps`)

---

## Solution 3: Lock-Free Drain

**Best for**: maximum concurrency, production-grade HPC

**Key idea**: Drain threads never acquire `run->mtx`. All drain-side state is atomic. Ready ops are pushed through a lock-free MPSC ring buffer. Enqueue thread is the single consumer.

### Data Layout

```cpp
// ── Lock-free MPSC ready ring ──
struct ReadyRing {
  static constexpr size_t kMask = 255;       // 256 slots, power of 2
  std::atomic<uint32_t> head_{0};            // consumer only (enqueue thread)
  std::atomic<uint32_t> tail_{0};            // producers CAS (drain threads)
  uint32_t buf_[kMask + 1];

  // Multi-producer push (drain threads)
  bool push(uint32_t op) {
    uint32_t t = tail_.load(std::memory_order_relaxed);
    uint32_t next = (t + 1) & kMask;
    if (next == head_.load(std::memory_order_acquire))
      return false;  // full — retry next cycle
    while (!tail_.compare_exchange_weak(t, next,
             std::memory_order_release, std::memory_order_relaxed))
      ;
    buf_[t] = op;
    return true;
  }

  // Single-consumer pop (enqueue thread)
  uint32_t pop() {
    uint32_t h = head_.load(std::memory_order_relaxed);
    if (h == tail_.load(std::memory_order_acquire))
      return ~0u;  // empty
    uint32_t op = buf_[h];
    head_.store((h + 1) & kMask, std::memory_order_release);
    return op;
  }
};

struct SprayRun {
  std::atomic<CollectiveOpStatus> status{Queued};
  std::atomic<size_t>              done_count{0};

  // ── Lock-free drain path ──
  std::vector<uint32_t>            successor_data;
  std::vector<uint32_t>            successor_off;
  std::vector<std::atomic<uint32_t>> indegree;
  ReadyRing                        ready_ring;

  // ── run->mtx protected (enqueue thread only) ──
  std::mutex                       mtx;
  std::vector<uint8_t>             submitted;
  std::vector<CmdWithId>           dev_cmds, tpt_cmds;
};
```

### Operations

**drain — per-op completion (NO lock)**:
```cpp
// drain_batch: mark op done without run->mtx
uint32_t off = successor_off[op];
uint32_t end = successor_off[op + 1];
for (uint32_t j = off; j < end; ++j) {
  uint32_t succ = successor_data[j];
  if (indegree[succ].fetch_sub(1, std::memory_order_release) == 1)
    ready_ring.push(succ);       // lock-free push, no mtx
}
m.run->done[op] = 1;            // atomic write (vector<atomic<uint8_t>>)
m.run->done_count.fetch_add(1, std::memory_order_release);
```

**enqueue — collect_ready (under run->mtx)**:
```cpp
// collect_ready: drain ready ring under run->mtx
for (;;) {
  uint32_t op = ready_ring.pop();
  if (op == ~0u) break;
  if (!submitted[op]) run.ready.push_back(op);
}
```

### Locking Model

```
                   enqueue thread          drain threads
                   ─────────────          ─────────────
runs_mutex_        ACQUIRE                (none)
run->mtx           ACQUIRE                (none)
submitted[]        READ/WRITE             (none)
dev_cmds/tpt_cmds  READ/WRITE             (none)
indegree[]          —                     fetch_sub (lock-free)
ready_ring          pop (single-consumer)  push (CAS, lock-free)
done[]               —                     write (atomic)
done_count           —                     fetch_add (atomic)
status               —                     store (atomic)
```

### Analysis

| Aspect | Detail |
|--------|--------|
| Enqueue lock | `run->mtx` only (no drain contention) |
| Drain lock | Zero — all atomic / CAS |
| Enqueue cost | O(ready_count) from lock-free ring |
| Drain cost | O(successor_count) + 1 CAS per successor that becomes ready |
| Ring capacity | 256 — sufficient for typical burst; full-push retries next cycle |
| Memory | ~8 bytes per op + 4KB ring buffer |
| ABA risk | None — each op pushed at most once (indegree hits 0 exactly once) |

### Pros
- Zero lock contention between enqueue and drain
- All HPC canonical patterns: flat arrays, fetch_sub counter, MPSC ring, CAS
- Scales to many drain threads / large graphs

### Cons
- More code (lock-free ring, atomics discipline)
- Ring full edge case (rare; handles gracefully)
- Requires `vector<atomic<uint8_t>>` for `done[]` (or manual atomic operations)

---

## Comparison Matrix

| | Current | S1: Bitmask | S2: Flat Array | S3: Lock-Free |
|---|---|---|---|---|
| `collect_ready` cost | O(N×D) scan | O(ready_count) | O(ready_count) | O(ready_count) |
| Drain lock | `run->mtx` | `run->mtx` | `run->mtx` | None |
| Enqueue lock | `run->mtx` | `run->mtx` | `run->mtx` | `run->mtx` |
| Max ops | Unlimited | 64 | Unlimited | Unlimited |
| Code complexity | Baseline | +10 lines | +30 lines | +80 lines |
| Allocation | Per-cycle | Zero | One-time | One-time |
| Memory overhead | 0 | 12 bytes/op | ~16 bytes/op | ~20 bytes/op + 4KB ring |
| Concurrency | Drain & enqueue contend | Same | Same | Zero contention |

## Recommendation

1. **Ship S2** immediately — flat array + `fetch_sub` + `ready_batch` under existing lock. Drops `collect_ready` from O(N×D) to O(ready), minimal risk, ~30 lines of new code.

2. **Bench S3** as follow-up — if drain-side `run->mtx` contention is measurable in production (high-frequency small collectives), the lock-free ring eliminates it.

3. **S1 as tactical** — if a specific collective (e.g., allreduce ring) is the 90% case and fits in 64 ops, the bitmask is two instructions and zero allocation. Can coexist with S2/S3 fallback.
