# UCCL Intra-Node Transfer: Implementation Plan

## End Goal

Enable the nixl uccl backend to automatically route transfers between **two separate nixl agents
on the same physical node** through CUDA IPC (`write_ipc` / `read_ipc` / `writev_ipc` /
`readv_ipc`) instead of RDMA. This applies to the standard two-agent transfer case
(`loadRemoteConnInfo` → `postXfer`) — not to be confused with `supportsLocal()`, which is a
nixl concept for a single agent transferring to itself.

When complete, a nixl agent that connects to a peer on the same machine will transparently use
the IPC data path, with no change required from the caller.

The IPC path (`write_ipc` / `read_ipc`) already exists and is tested at the Python / pybind11
level. This plan wires it into the `uccl_engine` C API and then into the nixl plugin.

---

## Nixl Plugin Code Path (reference)

The relevant flow through `src/plugins/uccl/uccl_backend.cpp`:

```
nixlUcclEngine::getConnInfo()            → uccl_engine_get_metadata()  → returns "ip:port?gpu"
nixlUcclEngine::loadRemoteConnInfo()     → uccl_engine_connect(ip, gpu, port)
nixlUcclEngine::startListener() [thread] → uccl_engine_accept()
nixlUcclEngine::postXfer()               → uccl_engine_read() / uccl_engine_write()
```

Key observations:
- `loadRemoteConnInfo` already has the remote IP (parsed from `remote_conn_info`)
- The local IP is available from `uccl_engine_get_metadata()`
- Comparing these two IPs is the correct and natural place to detect intra-node
- `startListener` stores accepted connections keyed by remote IP (`connected_agents_[ip_buf]`)

---

## Phase 0 — Full Code Flow (reference)

This section traces exactly what happens when a nixl user writes or reads memory between two
agents using the UCCL backend. Memory type (CPU/GPU) is noted where it matters.

Both `DRAM_SEG` (CPU/pinned) and `VRAM_SEG` (GPU) are supported — `getSupportedMems()` returns
both. `registerMem()` and `postXfer()` use the same code path for both; the RDMA engine handles
the rkey/lkey difference internally.

---

### Step 1 — Agent Creation (both sides, independent)

```python
config = nixl_agent_config(backends=["UCCL"])
agent  = nixl_agent("client", config)   # or "server"
```

**C++ (constructor):**
- `uccl_engine_create(num_cpus, in_python)` → allocates `Endpoint`, starts RDMA threads
- Spawns `startListener()` thread → immediately blocks in `uccl_engine_accept()`

---

### Step 2 — Advertise connection info (`getConnInfo`)

```python
local_meta = agent.get_agent_metadata()   # returns bytes
# user sends local_meta to the peer out-of-band (e.g. ZMQ)
```

**C++:**
- `getConnInfo()` → `uccl_engine_get_metadata()` → returns `"<ip>:<port>?<gpu_idx>"`
- This string is the only thing the peer needs to connect.

---

### Step 3 — Memory Registration (both sides, before transfer)

```python
descs          = agent.get_reg_descs(dataset)    # list of (addr, len, devId) descriptors
register_descs = agent.register_memory(descs)    # pins memory with RDMA engine
```

**C++ (`registerMem`):**
- `uccl_engine_reg(engine_, addr, len, &mr_id)` — pins the buffer with the RDMA NIC
  - Works for both CPU (pinned/DRAM) and GPU (VRAM) buffers
- `uccl_engine_prepare_fifo(engine_, mr_id, addr, len, fifo_item[64])` — pre-computes the
  64-byte `FifoItem` (contains rkey + lkey + base address) needed for one-sided RDMA
- Stores `nixlUcclBackendMD{ addr, len, mr_id, fifo_item }` in `mem_reg_info_`

The `fifo_item` is serialised as a hex string in `getPublicData()` and travels with the
descriptor to the peer in Step 4.

---

### Step 4 — Exchange descriptors / connect (out-of-band)

```python
# client side
agent.add_remote_agent(server_meta)           # connect to server
local_xfer_descs  = register_descs.trim()
remote_xfer_descs = agent.deserialize_descs(server_xfer_desc_bytes)
```

**C++ (`loadRemoteConnInfo`):**
- Parses `"ip:port?gpu"` from server metadata
- `uccl_engine_connect(engine_, ip, gpu, port)` — TCP + RDMA handshake to server
- `uccl_engine_start_listener(conn)` — spawns a thread that calls `recv()` to receive
  completion notifications

**C++ (server `startListener` thread):**
- `uccl_engine_accept()` returns a new `uccl_conn_t*`
- `uccl_engine_start_listener(conn)` — same notification receiver thread on server side
- Stores `conn` in `connected_agents_[remote_ip]`

**C++ (`loadRemoteMD`)** — called when deserializing the remote descriptor list:
- Decodes hex → `fifo_item[64]` and stores in a new `nixlUcclBackendMD` (remote side)
- This MD is never registered with the RDMA engine; it is only a container for the remote
  rkey + address needed to address the remote buffer.

---

### Step 5 — Transfer Preparation (`initialize_xfer` / `prepXfer`)

```python
handle = agent.initialize_xfer("WRITE", local_xfer_descs, remote_xfer_descs, "server")
```

**C++ (`prepXfer`):**
- Looks up `uccl_conn_t*` for the remote agent name
- For each iovec pair (local[i], remote[i]):
  - Deserialises remote `fifo_item` → `FifoItem` struct
  - `uccl_engine_update_fifo(fifo_item, remote_addr, size)` — patches the exact sub-buffer
    address and size into the pre-computed FifoItem
- Creates `nixlUcclReqH{ conn, fifo_items[] }`

No data moves here.

---

### Step 6 — Transfer Execution (`transfer` / `postXfer`)

```python
state = agent.transfer(handle)          # returns "INPROG"
while agent.check_xfer_state(handle) != "DONE":
    pass
```

**C++ (`postXfer`):**
- Builds vectors of `(mr_id, local_addr, size)` for each iovec
- **NIXL_WRITE** → `uccl_engine_write_vector(conn, mr_ids, local_addrs, sizes, fifo_items, n)`
  - One-sided RDMA WRITE: local NIC pushes data directly into remote memory using rkey
  - Works for CPU→CPU, GPU→CPU, CPU→GPU, GPU→GPU (all via RDMA)
- **NIXL_READ** → `uccl_engine_read_vector(conn, mr_ids, local_addrs, sizes, fifo_items, n)`
  - One-sided RDMA READ: local NIC pulls data from remote memory using rkey
- Returns `NIXL_IN_PROG`; transfer is async

**C++ (`checkXfer`):**
- `uccl_engine_xfer_status(conn, transfer_id)` — polls NIC completion queue
- When done, sends a completion notification via TCP (`uccl_engine_send_notif`) so the remote
  side knows the transfer finished (important for the WRITE case where the writer is the only
  one that knows when data landed)

---

### Key architectural notes

| Aspect | Current (RDMA) path |
|--------|---------------------|
| Connection | TCP + RDMA QP pair via `uccl_engine_connect` / `uccl_engine_accept` |
| Memory token | 64-byte `FifoItem` with RDMA rkey + base address, pre-computed at `registerMem` |
| Transfer | One-sided RDMA WRITE or READ — remote side does **nothing** during data movement |
| Completion signal | TCP notification from initiator to remote side via `uccl_engine_send_notif` |
| CPU memory | Pinned DRAM, registered with RDMA NIC — same RDMA path as GPU |
| GPU memory | VRAM registered with RDMA NIC (GPUDirect) — same RDMA path as CPU |
| `connected_agents_` key | agent name on connect-side; remote IP on accept-side |

---

## Phases

### Phase 1 — Detection in `uccl_engine`  ✅ Done

`is_intra_node` bool added to `uccl_conn` struct in `uccl_engine.cc`.
Set by comparing remote IP with local IP (via `get_local_ip_from_engine()`) in both
`uccl_engine_connect()` and `uccl_engine_accept()`. A `std::cout` log prints the result.

**Files changed:** `uccl_engine.cc` only — no header or pybind11 changes needed.

#### Phase 1 unit test

Run `benchmark_nixl.py` (UCCL backend) as two separate processes on the **same node**.
The benchmark uses ZMQ for coordination; the two processes call the full nixl plugin code path
(`getConnInfo` → `loadRemoteConnInfo` → `uccl_engine_connect` / `uccl_engine_accept`).

```bash
# Terminal 1 — server (receiver)
cd /home/lirans/uccl/p2p
python benchmarks/benchmark_nixl.py --role server --backend uccl --device cpu

# Terminal 2 — client (sender), pointing to localhost
cd /home/lirans/uccl/p2p
python benchmarks/benchmark_nixl.py --role client --backend uccl --device cpu --remote-ip 127.0.0.1
```

Expected output (from our `std::cout` in `uccl_engine.cc`):
```
uccl_engine_connect: connection to 127.0.0.1 is intra-node
uccl_engine_accept: connection from 127.0.0.1 is intra-node
```

## Status

| Phase | Status |
|-------|--------|
| Phase 1 — `uccl_conn::is_intra_node` in `uccl_engine.cc` | ✅ Done |
| Phase 1 unit test — `benchmark_nixl.py` intra-node log check | 🔲 Next |
