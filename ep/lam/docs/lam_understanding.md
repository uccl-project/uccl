# rdma_recv_x Layout and Usage

`rdma_recv_x` is the receive data buffer used in the low-latency dispatch phase (`ep/src/internode_ll.cu`). Each rank has its **own** buffer in its **own** GPU memory; when rank A sends to an expert on rank B, A does an RDMA put **into B’s** `rdma_recv_x`.

---

## Dispatch send flow (sender side)

- **Input** goes into **`rdma_x`** (send buffer): each token’s hidden is read from input `x`, cast/copied into `rdma_x[token_idx]` (one message per token, `num_bytes_per_msg` each).
- **Multiple SMs**: the grid runs many blocks (SMs). Tokens are split across SMs with:
  ```cpp
  for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms)
  ```
  So each SM handles every `num_sms`-th token (e.g. SM 0 → token 0, num_sms, 2*num_sms, …).
- **Per token, one warp per top-k**: for each token, `num_topk` warps (warp_id 0 .. num_topk-1) each take one top-k destination. Each warp reads `dst_expert_idx = topk_idx[token_idx][warp_id]`, allocates a `slot_idx` for that expert, then puts one message from `rdma_x[token_idx]` to the receiver’s `rdma_recv_x` at the slot given by the layout below.
- **Write to dst**: the remote address is computed with the **rdma_recv_x layout** (see next section): `dst_ptr = rdma_recv_x + dst_expert_local_idx * ... + rank * ... + slot_idx * num_bytes_per_msg`, and the sender does an RDMA put (or IPC copy) of that one message to `dst_ptr`.

---

## Count send (after token sends)

After all token puts to each expert are done, the sender issues a **count send** for that expert.

- **Not a broadcast to all ranks**: one count is sent **per expert**, to the **rank that owns that expert** (`dst_rank = expert_idx / num_local_experts`). So we send one count to the owner of expert 0, one to the owner of expert 1, etc. Each rank only receives counts for its own local experts (from each source rank).
- **Content**: “I (this rank) sent you (dst_rank) this many tokens for expert `dst_expert_local_idx`” — i.e. the number of tokens this rank routed to that expert (`num_tokens_sent`), encoded as `-num_tokens_sent - 1` and written into the receiver’s `rdma_recv_count[dst_expert_local_idx][rank]` (or `rdma_recv_count_internode` for cross-node). The receiver polls until non-zero, then decodes `num_tokens = -value - 1` and uses that to know how many messages to read from `rdma_recv_x` for that (expert, source_rank).

---

## Layout

**Logical 3D layout** (on the rank that owns the buffer):

```
rdma_recv_x[expert_local_idx][source_rank][slot_idx]
```

| Dimension | Index | Size | Meaning |
|-----------|--------|------|---------|
| 0 | `expert_local_idx` | `num_local_experts` | Which local expert on this rank |
| 1 | `source_rank` | `num_ranks` | Which rank sent the data |
| 2 | `slot_idx` | `num_max_dispatch_tokens_per_rank` | Which message slot from that source |

- **One slot** = `num_bytes_per_msg` bytes (control `int4` + hidden + scales).
- **Total size per GPU:** `num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg`.

Declared in `ep/include/ep_config.hpp` as `LowLatencyBuffer::dispatch_rdma_recv_data_buffer`; allocated from `rdma_buffer` in `LowLatencyLayout`, passed to the kernel as `rdma_recv_x`.

---

## Usage

**Sender (writing):** The sender computes the **remote** address on the receiver’s `rdma_recv_x` with:

```cpp
dst_ptr = rdma_recv_x
  + dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg
  + rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg
  + slot_idx * num_bytes_per_msg;
```

So the sender uses `(dst_expert_local_idx, rank, slot_idx)` to pick one slot in the receiver’s buffer and does an RDMA put (or IPC copy) of one message there.

**Receiver (reading):** The receiver waits for the send count for each `(local_expert_idx, src_rank)`, then reads that many messages from the corresponding block:

```cpp
rdma_recv_x_uint8 = rdma_recv_x
  + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg
  + src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
// then read slots 0 .. num_recv_tokens - 1 (each num_bytes_per_msg)
```

So the receiver uses the same layout: for each of its local experts and each source rank, it reads `num_recv_tokens` contiguous slots from that block and packs them into `packed_recv_x` etc.

---

## Dispatch Receive Flow (receiver side)

### Overview

The receiver side of dispatch:
1. **Polls** for incoming tokens from each `(local_expert, src_rank)` pair
2. **Copies** tokens from `rdma_recv_x` (RDMA buffer, may have gaps) to `packed_recv_x` (contiguous output buffer)
3. Records metadata for the combine phase

### Input / Output

| Buffer | Type | Description |
|--------|------|-------------|
| **Input:** `rdma_recv_x` | RDMA recv buffer | Organized by `[local_expert][src_rank][slot]`, may have unused slots |
| **Input:** `rdma_recv_count` | Atomic counters | Token counts per `(local_expert, src_rank)`, set by senders |
| **Output:** `packed_recv_x` | Packed buffer | Contiguous tokens per expert, ready for Expert Forward |
| **Output:** `packed_recv_src_info` | int array | Original token index at source rank (for combine phase) |
| **Output:** `packed_recv_count` | int array | Total tokens received per local expert |
| **Output:** `packed_recv_layout_range` | int64 array | `(num_tokens, begin_idx)` per `(local_expert, src_rank)` |

### Parallelization Hierarchy

```
Block (SM)
└── Warp Group (handles one (local_expert, src_rank) pair)
    └── Sub-warp (= 1 warp = 32 threads, handles multiple tokens in strided fashion)
        └── Lane (= 1 thread, handles part of one token's hidden data)
```

> **Note:** "Sub-warp" and "warp" are the **same thing** (32 threads). The naming depends on perspective:
> - From **block** perspective: called "warp" (`warp_id = thread_id / 32`)
> - From **warp group** perspective: called "sub-warp" (`sub_warp_id = warp_id % num_warps_per_group`)
>
> A warp group contains multiple warps; each warp within the group is called a "sub-warp" to indicate it's a sub-unit of the group.

### ASCII Diagram: Block Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Block 0 (SM 0)                                 │
│                         blockDim.x = 1024 threads                           │
│                                                                             │
│  ┌─────────────────────────────────────┐ ┌─────────────────────────────────┐│
│  │         Warp Group 0                │ │         Warp Group 1            ││
│  │      (handles Expert Pair 0)        │ │      (handles Expert Pair 1)    ││
│  │  num_warps_per_group = 4            │ │  num_warps_per_group = 4        ││
│  │                                     │ │                                 ││
│  │  ┌─────────────────────────────┐    │ │  ┌─────────────────────────────┐││
│  │  │ Sub-warp 0 (warp_id=0)     │    │ │  │ Sub-warp 0 (warp_id=4)     │││
│  │  │ ┌──┬──┬──┬──┬─────┬──┬──┐  │    │ │  │ 32 threads                 │││
│  │  │ │T0│T1│T2│T3│ ... │T30│T31│  │    │ │  └─────────────────────────────┘││
│  │  │ └──┴──┴──┴──┴─────┴──┴──┘  │    │ │  ┌─────────────────────────────┐││
│  │  │      32 threads (lanes)    │    │ │  │ Sub-warp 1 (warp_id=5)     │││
│  │  └─────────────────────────────┘    │ │  │ 32 threads (waiter+copier) │││
│  │  ┌─────────────────────────────┐    │ │  └─────────────────────────────┘││
│  │  │ Sub-warp 1 (warp_id=1)     │    │ │  ┌─────────────────────────────┐││
│  │  │ 32 threads                 │    │ │  │ Sub-warp 2 (warp_id=6)     │││
│  │  │ ★ waiter + copy tokens     │    │ │  │ 32 threads                 │││
│  │  └─────────────────────────────┘    │ │  └─────────────────────────────┘││
│  │  ┌─────────────────────────────┐    │ │  ┌─────────────────────────────┐││
│  │  │ Sub-warp 2 (warp_id=2)     │    │ │  │ Sub-warp 3 (warp_id=7)     │││
│  │  │ 32 threads                 │    │ │  │ 32 threads                 │││
│  │  └─────────────────────────────┘    │ │  └─────────────────────────────┘││
│  │  ┌─────────────────────────────┐    │ │                                 ││
│  │  │ Sub-warp 3 (warp_id=3)     │    │ │                                 ││
│  │  │ 32 threads                 │    │ │                                 ││
│  │  └─────────────────────────────┘    │ │                                 ││
│  └─────────────────────────────────────┘ └─────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘

Index calculations:
┌────────────────────────────────────────────────────────────────┐
│ thread_id     = threadIdx.x                   (0 ~ 1023)       │
│ warp_id       = thread_id / 32                (0 ~ 31)         │
│ lane_id       = thread_id % 32                (0 ~ 31)         │
│                                                                │
│ warp_group_id = warp_id / num_warps_per_group (which group)    │
│ sub_warp_id   = warp_id % num_warps_per_group (which in group) │
│                                                                │
│ responsible_expert_idx = sm_id * num_warp_groups + warp_group_id│
└────────────────────────────────────────────────────────────────┘
```

### ASCII Diagram: (local_expert, src_rank) Distribution

Each `responsible_expert_idx` maps to one `(local_expert, src_rank)` pair:

```
responsible_expert_idx ∈ [0, num_experts - 1]

src_rank         = responsible_expert_idx / num_local_experts
local_expert_idx = responsible_expert_idx % num_local_experts

Example: num_local_experts = 9, num_ranks = 32, num_experts = 288

┌───────────────────────────────────────────────────────────────┐
│ idx │ src_rank │ local_expert │        Meaning                │
├─────┼──────────┼──────────────┼───────────────────────────────┤
│  0  │    0     │      0       │ Recv from Rank0 for Expert0   │
│  1  │    0     │      1       │ Recv from Rank0 for Expert1   │
│ ... │   ...    │     ...      │                               │
│  8  │    0     │      8       │ Recv from Rank0 for Expert8   │
│  9  │    1     │      0       │ Recv from Rank1 for Expert0   │
│ 10  │    1     │      1       │ Recv from Rank1 for Expert1   │
│ ... │   ...    │     ...      │                               │
│ 287 │   31     │      8       │ Recv from Rank31 for Expert8  │
└───────────────────────────────────────────────────────────────┘

Total 288 pairs = 288 warp groups needed
```

### ASCII Diagram: Receive Matrix (on one rank)

```
Rank 0 as Receiver, has 9 local experts (Expert 0~8)
Needs to receive from 32 src_ranks (including itself)

┌─────────────────────────────────────────────────────────────────────────┐
│                         Rank 0's Receive Matrix                         │
│                                                                         │
│              src_rank                                                   │
│         0    1    2    3   ...   31                                     │
│       ┌────┬────┬────┬────┬─────┬────┐                                  │
│     0 │ WG │ WG │ WG │ WG │ ... │ WG │  ← recv from each rank for Exp0  │
│       ├────┼────┼────┼────┼─────┼────┤                                  │
│     1 │ WG │ WG │ WG │ WG │ ... │ WG │  ← recv from each rank for Exp1  │
│  l    ├────┼────┼────┼────┼─────┼────┤                                  │
│  o  2 │ WG │ WG │ WG │ WG │ ... │ WG │                                  │
│  c    ├────┼────┼────┼────┼─────┼────┤                                  │
│  a  3 │ WG │ WG │ WG │ WG │ ... │ WG │                                  │
│  l    ├────┼────┼────┼────┼─────┼────┤                                  │
│     4 │ WG │ WG │ WG │ WG │ ... │ WG │                                  │
│  e    ├────┼────┼────┼────┼─────┼────┤                                  │
│  x  5 │ WG │ WG │ WG │ WG │ ... │ WG │                                  │
│  p    ├────┼────┼────┼────┼─────┼────┤                                  │
│  e  6 │ WG │ WG │ WG │ WG │ ... │ WG │                                  │
│  r    ├────┼────┼────┼────┼─────┼────┤                                  │
│  t  7 │ WG │ WG │ WG │ WG │ ... │ WG │                                  │
│       ├────┼────┼────┼────┼─────┼────┤                                  │
│     8 │ WG │ WG │ WG │ WG │ ... │ WG │                                  │
│       └────┴────┴────┴────┴─────┴────┘                                  │
│                                                                         │
│   Each cell = 1 Warp Group = 1 (local_expert, src_rank) pair            │
│   Total: 9 × 32 = 288 Warp Groups                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### ASCII Diagram: Sub-warp Token Distribution

Within a warp group, sub-warps divide tokens by index:

```
Warp Group for (local_expert=2, src_rank=5) receives 30 tokens
num_warps_per_group = 10

Token distribution (for loop: i = sub_warp_id; i < num_recv_tokens; i += 10):
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Sub-warp 0: processes token 0, 10, 20                             │
│  Sub-warp 1: processes token 1, 11, 21  (also polls for arrival)   │
│  Sub-warp 2: processes token 2, 12, 22                             │
│  Sub-warp 3: processes token 3, 13, 23                             │
│  Sub-warp 4: processes token 4, 14, 24                             │
│  Sub-warp 5: processes token 5, 15, 25                             │
│  Sub-warp 6: processes token 6, 16, 26                             │
│  Sub-warp 7: processes token 7, 17, 27                             │
│  Sub-warp 8: processes token 8, 18, 28                             │
│  Sub-warp 9: processes token 9, 19, 29                             │
│                                                                     │
│  All 10 sub-warps process 10 different tokens in PARALLEL!         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### ASCII Diagram: Single Token Copy (within one sub-warp)

```
┌─────────────────────────────────────────────────────────────────────┐
│               Sub-warp processes 1 Token                            │
│                                                                     │
│   Token structure: [src_info: 16B] [hidden: 7168B] [scales: ~56B]  │
│                                                                     │
│   32 threads copy in parallel:                                      │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │ Lane 0:  src_info + hidden[0:16]                           │   │
│   │ Lane 1:  hidden[16:32]                                     │   │
│   │ Lane 2:  hidden[32:48]                                     │   │
│   │ ...                                                        │   │
│   │ Lane 31: hidden[496:512]                                   │   │
│   │                                                            │   │
│   │ (then loop: Lane 0 copies hidden[512:528], ...)            │   │
│   │                                                            │   │
│   │ UNROLLED_WARP_COPY: 32 lanes copy in parallel              │   │
│   └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   7168B / 32 threads / 16B per load ≈ 14 iterations                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Understanding token_idx: Local Index on Sender

**IMPORTANT:** `token_idx` in the recv logs is the **sender's local token index**, NOT a global index.

```
Each rank has its OWN set of input tokens (e.g., 128 tokens per rank):

┌─────────────────────────────────────────────────────────────────────┐
│  Rank 0: token 0, 1, 2, ..., 83, ..., 127   (local to Rank 0)      │
│  Rank 1: token 0, 1, 2, ..., 83, ..., 127   (local to Rank 1)      │
│  Rank 2: token 0, 1, 2, ..., 83, ..., 127   (local to Rank 2)      │
│  ...                                                                │
│  Rank 8: token 0, 1, 2, ..., 83, ..., 127   (local to Rank 8)      │
│                                                                     │
│  These are DIFFERENT tokens with the same local index!             │
└─────────────────────────────────────────────────────────────────────┘

Example log interpretation:
  [RECV] rank=0 expert=156 src_rank=8 slot=3 token_idx=83 ...
  
  This means:
  - Rank 0 received a token
  - The token came from Rank 8
  - On Rank 8, this token was token #83 (out of Rank 8's 128 tokens)
  - token_idx=83 is Rank 8's LOCAL index, not global
```

**Why multiple logs show the same token_idx (e.g., token_idx=83)?**

```
If you see 11 recv logs all with token_idx=83, they are 11 DIFFERENT tokens:
  - Rank 0's token #83, Rank 2's token #83, Rank 6's token #83, etc.
  - They happen to share the same local index on their respective ranks
  - They are routed to different experts on Rank 0 based on top-k routing

Example (top-k=2, 128 tokens per rank):
┌─────────────────────────────────────────────────────────────────────┐
│  Rank 6's token #83:                                               │
│    → top-k routing sends to Expert 3 (on Rank 0)  → expert=111     │
│    → top-k routing sends to Expert 10 (on Rank 0) → expert=118     │
│                                                                     │
│  Rank 8's token #83:                                               │
│    → top-k routing sends to Expert 5 (on Rank 0)  → expert=149     │
│    → top-k routing sends to Expert 12 (on Rank 0) → expert=156     │
│                                                                     │
│  These are 4 different recv operations for 2 different tokens      │
│  (each token sent to 2 experts due to top-k=2)                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Complete Data Flow Example

```
Rank 5 (Sender) has 128 local tokens (token 0 ~ 127):
┌─────────────────────────────────────────────────────────────────────┐
│  Input tokens: T0, T1, T2, T3, T4, T5, ..., T83, ..., T127         │
│                (these are Rank 5's LOCAL tokens)                   │
│                                                                     │
│  Top-k routing results (top-k=2, each token → 2 experts):          │
│    T0 → Expert 2 (on Rank 0), Expert 15 (on Rank 1)               │
│    T1 → Expert 7 (on Rank 0), Expert 22 (on Rank 2)               │
│    T2 → Expert 2 (on Rank 0), Expert 31 (on Rank 3)               │
│    T83 → Expert 4 (on Rank 0), Expert 19 (on Rank 1)              │
│    ...                                                              │
│                                                                     │
│  Sent to Rank 0: T0, T1, T2, T83, ... (with their local indices)  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Rank 0 (Receiver):
┌─────────────────────────────────────────────────────────────────────┐
│  Warp Group for (local_expert=2, src_rank=5)                       │
│                                                                     │
│  Received tokens from Rank 5 for Expert 2:                         │
│    slot[0] → token_idx=0  (Rank 5's T0)                            │
│    slot[1] → token_idx=2  (Rank 5's T2)                            │
│    ...                                                              │
│                                                                     │
│  token_idx is preserved so combine phase can send results back!    │
│                                                                     │
│  Sub-warps divide work (assuming 10 sub-warps):                    │
│    Sub-warp 0: slot[0], slot[10], ...                              │
│    Sub-warp 1: slot[1], slot[11], ...                              │
│    ...                                                              │
└─────────────────────────────────────────────────────────────────────┘
```

**Key formulas:**
```cpp
warp_group_id = warp_id / num_warps_per_group;
sub_warp_id   = warp_id % num_warps_per_group;
responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;
src_rank = responsible_expert_idx / num_local_experts;
local_expert_idx = responsible_expert_idx % num_local_experts;
```

**Configuration (computed at launch):**
```cpp
num_warp_groups     = ceil_div(num_experts, num_device_sms);   // warp groups per block
num_warps_per_group = kNumMaxWarpGroups / num_warp_groups;     // sub-warps per warp group
num_sms             = ceil_div(num_experts, num_warp_groups);  // blocks to launch
// kNumMaxWarpGroups = 32 (NVIDIA) or 16 (AMD)
```

**Example (256 experts, H100 with 132 SMs):**
- `num_warp_groups = ceil(256/132) = 2` warp groups per block
- `num_warps_per_group = 32/2 = 16` sub-warps per warp group
- `num_sms = ceil(256/2) = 128` blocks launched
- Total: 128 blocks × 2 warp groups = 256 responsible_expert_idx values

### What Each Level Does

| Level | Handles | Description |
|-------|---------|-------------|
| **Warp Group** | One `(local_expert, src_rank)` pair | Polls for tokens, then copies all tokens from that src_rank to that expert |
| **Sub-warp** | Multiple tokens (strided) | Each sub-warp copies tokens `i, i+n, i+2n, ...` where `n = num_warps_per_group` |
| **Lane** | Part of one token's hidden | 32 lanes cooperatively copy one token's ~7KB hidden data |

### Key Insight: Warp Group = (local_expert, src_rank)

**Each warp group processes exactly one `(local_expert, src_rank)` pair:**

1. **One warp group** is assigned one `responsible_expert_idx` which maps to one `(local_expert, src_rank)` pair
2. **Many tokens** may arrive from this `src_rank` to this `local_expert` (stored in `num_recv_tokens`)
3. **All sub-warps in this warp group** cooperatively process these tokens in parallel

```
Code verification (internode_ll.cu):

Line 527-529: Map responsible_expert_idx → (local_expert, src_rank)
┌─────────────────────────────────────────────────────────────────────┐
│ if (responsible_expert_idx < num_experts) {                        │
│   src_rank = responsible_expert_idx / num_local_experts;           │
│   local_expert_idx = responsible_expert_idx % num_local_experts;   │
│ }                                                                   │
└─────────────────────────────────────────────────────────────────────┘

Line 620: Count tokens from this (local_expert, src_rank) pair
┌─────────────────────────────────────────────────────────────────────┐
│ num_recv_tokens = num_recv_tokens_internode + num_recv_tokens_ipc; │
│ // This is the total tokens from src_rank to local_expert          │
└─────────────────────────────────────────────────────────────────────┘

Line 656: All sub-warps in warp group process these tokens together
┌─────────────────────────────────────────────────────────────────────┐
│ for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) │
│   // sub_warp 0: token 0, 10, 20, ...                              │
│   // sub_warp 1: token 1, 11, 21, ...                              │
│   // ...                                                            │
│ }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Summary:**
- **Warp Group scope:** one `(local_expert, src_rank)` pair
- **What it receives:** all tokens that `src_rank` sent to `local_expert`
- **How it processes:** all sub-warps (warps) in the group work together, dividing tokens by index

### recv_token_begin_idx: Dynamic Slot Allocation

Multiple warp groups (different `src_rank`) write to the same expert's output buffer. They use **atomicAdd** for thread-safe slot allocation:

```cpp
// Line 444-445 in internode_ll.cu
recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
```

**How it works:**
1. `packed_recv_count[local_expert_idx]` is an atomic counter, initialized to 0
2. `atomicAdd` returns the value **before** adding, and atomically increments the counter
3. The returned value is the starting index for this warp group's tokens

**Example (Expert 0 receives from 4 src_ranks, in arbitrary order):**

| Order | Warp Group | src_rank | num_tokens | atomicAdd returns | counter after | Write slots |
|-------|------------|----------|------------|-------------------|---------------|-------------|
| 1st | B | 2 | 8 | 0 | 8 | [0..7] |
| 2nd | A | 0 | 5 | 8 | 13 | [8..12] |
| 3rd | D | 5 | 3 | 13 | 16 | [13..15] |
| 4th | C | 1 | 4 | 16 | 20 | [16..19] |

**Key points:**
- **No ordering required**: First-come, first-served. Order depends on network latency and GPU scheduling.
- **Correctness**: `recv_src_info[i]` records the original token index, so combine phase can send results back correctly.
- **recv_range**: `recv_range[src_rank] = pack2(num_recv_tokens, recv_token_begin_idx)` records where each src_rank's tokens ended up.

### Receive Flow Detail

```
1. Grid Sync
   └── Wait for send phase to complete (makes packed_recv_count visible)

2. Compute buffer pointers
   └── Each warp group calculates its slice of rdma_recv_x based on (local_expert, src_rank)

3. Poll for token count (sub_warp_id == 1, lane_id == 0 only)
   └── Spin on rdma_recv_count[local_expert][src_rank] until non-zero
   └── Decode: num_recv_tokens = -value - 1

4. Allocate output slots (sub_warp_id == 1, lane_id == 0 only)
   └── recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens)
   └── Store to shared memory for other sub-warps

5. Warp group barrier
   └── All sub-warps now have num_recv_tokens and recv_token_begin_idx

6. Copy tokens (all sub-warps in parallel)
   └── for (i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group):
       ├── Lane 0: copy src_info (4 bytes)
       ├── All 32 lanes: copy hidden data (~7KB, strided by lane_id)
       └── All 32 lanes: copy FP8 scales (if applicable)
```

### Why Pack?

| rdma_recv_x (Source) | packed_recv_x (Destination) |
|----------------------|-----------------------------|
| Organized by `[expert][src_rank][slot]` | Organized by `[expert][token]` |
| Fixed slot size, may have unused slots | Contiguous, no gaps |
| Scattered across src_ranks | All tokens for one expert together |
| Not suitable for batched compute | Ready for Expert Forward (batched MLP) |
