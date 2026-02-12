# Trace: `x` and `rdma_channel_data.send_buffer` (high-throughput dispatch)

## 1. Trace: `x` (original memory — read source)

### With code lines

| 層級 | 變數 / 來源 | File | Line |
|------|-------------|------|------|
| **Python** | 使用者傳入的 `x`（`torch.Tensor`，shape `[num_tokens, hidden]`） | `buffer.py` | 906-907 (參數 `x`) |
| **Python** | `x, x_scales = x if isinstance(x, tuple) else (x, None)` | `buffer.py` | 935 |
| **Python** | `self.runtime.internode_dispatch(x, x_scales, ...)` | `buffer.py` | 954, 1007 |
| **C++** | `Buffer::internode_dispatch(torch::Tensor const& x, ...)` | `uccl_ep.cc` | 401 (宣告), 416 (參數 `x`) |
| **C++** | 傳給 kernel 的 pointer：`x.data_ptr()` | `uccl_ep.cc` | 697 |
| **C++ (host)** | `dispatch(..., void const* x, ...)` | `internode.cu` | 1516 (參數 `x`) |
| **C++ (host)** | launch 時轉型：`reinterpret_cast<int4 const*>(x)` | `internode.cu` | 1565 |
| **C++ (kernel)** | 參數 `int4 const* x` | `internode.cu` | 482 |
| **C++ (kernel)** | 讀取位址 `x + token_idx * hidden_int4` | `internode.cu` | 854 |

### In kernel (device)

| Layer | Variable / Expression | File:Line |
|-------|----------------------|-----------|
| Dispatch kernel param | `x` (type: `int4 const*`) | `internode.cu:482` |
| Copy source | `x + token_idx * hidden_int4` | `internode.cu:854` |

### Host launch (C++)

| Layer | Variable / Expression | File:Line |
|-------|----------------------|-----------|
| Kernel launch | `reinterpret_cast<int4 const*>(x)` | `internode.cu:1565` |
| Host `dispatch()` param | `x` (type: `void const*`) | `internode.cu:1516` |
| Caller | `x.data_ptr()` | `uccl_ep.cc:697` |
| Caller method | `Buffer::internode_dispatch(torch::Tensor const& x, ...)` | `uccl_ep.cc:401, 416` |

### Python

| Layer | Variable / Expression | File:Line |
|-------|----------------------|-----------|
| `Buffer.internode_dispatch` arg | `x` (type: `Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`) | `buffer.py:906-907` |
| Unpack if tuple | `x, x_scales = x if isinstance(x, tuple) else (x, None)` | `buffer.py:935` |
| Passed to C++ | `self.runtime.internode_dispatch(x, x_scales, ...)` | `buffer.py:954, 1007` |

### Chain summary (x)

```
User / caller
  → Python:  x  (torch.Tensor, shape [num_tokens, hidden])
  → buffer.py:  internode_dispatch(self, x, ...)  →  self.runtime.internode_dispatch(x, ...)
  → uccl_ep.cc:  Buffer::internode_dispatch(torch::Tensor const& x, ...)  →  x.data_ptr()
  → internode.cu:  dispatch(..., void const* x, ...)  →  reinterpret_cast<int4 const*>(x)
  → kernel:  int4 const* x  →  read at  x + token_idx * hidden_int4
```

So **`x`** is the **input activation tensor** from the MoE layer (user-provided), in GPU memory; the kernel reads from it.

---

## 2. Trace: `rdma_channel_data.send_buffer` (new contiguous memory — write destination)

### With code lines

| 層級 | 變數 / 來源 | File | Line |
|------|-------------|------|------|
| **Python** | `num_rdma_bytes` 傳入 `Buffer(..., num_rdma_bytes=...)` | `buffer.py` | 59 (參數) |
| **Python** | `self.scratch = torch.zeros(num_rdma_bytes, dtype=torch.uint8, device=...)` | `buffer.py` | 92-94 |
| **Python** | (ROCm) `self.scratch = ep.get_rdma_buffer(num_rdma_bytes, device_index)` | `buffer.py` | 96 |
| **Python** | `rdma_buffer_ptr = self.scratch.data_ptr()` | `buffer.py` | 98 |
| **Python** | `self.runtime.set_rdma_buffer_raw(rdma_buffer_ptr)` | `buffer.py` | 128 |
| **C++** | Python binding 呼叫 `self.set_rdma_buffer_raw(addr)` → `Buffer::set_rdma_buffer_raw(void* ptr)` | `uccl_ep.cc` | 2059-2063 |
| **C++** | `void Buffer::set_rdma_buffer_raw(void* ptr)` → `rdma_buffer_ptr = ptr` | `uccl_ep.cc` | 1865-1869 |
| **C++** | Member：`void* Buffer::rdma_buffer_ptr` | `uccl_ep.cc` | 1912 |
| **C++** | `uccl::internode::dispatch(..., rdma_buffer_ptr, ...)` 傳入的參數 | `uccl_ep.cc` | 710 |
| **C++ (host)** | `void dispatch(..., void* rdma_buffer_ptr, ...)` | `internode.cu` | 1523 (參數) |
| **C++ (host)** | Kernel launch：`dispatch_func(..., rdma_buffer_ptr, ...)` | `internode.cu` | 1569-1570 |
| **C++ (kernel)** | 參數 `void* rdma_buffer_ptr` | `internode.cu` | 494 |
| **C++ (kernel)** | `auto rdma_channel_data = SymBuffer<uint8_t>(rdma_buffer_ptr, ...)` | `internode.cu` | 552-554 |
| **C++ (kernel)** | RDMASender 用：`send_buffer` = `rdma_channel_data.send_buffer(lane_id)` 或 `.recv_buffer(lane_id)` | `internode.cu` | 725-727 |
| **C++ (kernel)** | Coordinator put 用：`rdma_channel_data.send_buffer(dst_rdma_rank)` | `internode.cu` | 1034 |
| **C++ (buffer.cuh)** | `SymBuffer::send_ptr` = `gbl_ptr + per_channel_bytes * sm_id` | `buffer.cuh` | 125 |
| **C++ (buffer.cuh)** | `dtype_t* send_buffer(int idx)` → `send_ptr + num_bytes * idx` | `buffer.cuh` | 132-135 |

### In kernel (device)

| Layer | Variable / Expression | File:Line |
|-------|----------------------|-----------|
| Send buffer (per lane) | `send_buffer` = `rdma_channel_data.recv_buffer(lane_id)` or `.send_buffer(lane_id)` | `internode.cu:725-727` |
| Per-rank send buffer | `rdma_channel_data.send_buffer(dst_rdma_rank)` | `internode.cu:1034` |
| `rdma_channel_data` | `SymBuffer<uint8_t>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels)` | `internode.cu:552-554` |
| Kernel param | `rdma_buffer_ptr` (type: `void*`) | `internode.cu:494` |

### SymBuffer: where `send_buffer` points (buffer.cuh)

| Member | Meaning |
|--------|--------|
| `SymBuffer(void*& gbl_ptr, int num_elems, int num_ranks, int sm_id, int num_sms)` | Carves a region from `gbl_ptr`; **advances** `gbl_ptr` by `total_bytes`. |
| `send_ptr` | `gbl_ptr + per_channel_bytes * sm_id` (per-channel offset). |
| `send_buffer(int idx)` | `send_ptr + num_bytes * idx` → one contiguous region per rank index. |

So **`rdma_channel_data.send_buffer(idx)`** is a **subregion of the memory pointed to by `rdma_buffer_ptr`** when the kernel was launched.

### Host launch (C++)

| Layer | Variable / Expression | File:Line |
|-------|----------------------|-----------|
| Kernel launch | `rdma_buffer_ptr` passed to `dispatch(...)` | `internode.cu:1569` |
| Host `dispatch()` param | `void* rdma_buffer_ptr` | `internode.cu:1523` |
| Caller | `rdma_buffer_ptr` (no cast) | `uccl_ep.cc:710` |
| Source | `Buffer` member: `rdma_buffer_ptr` | `uccl_ep.cc:1912` |

### Where `Buffer::rdma_buffer_ptr` is set

| Layer | Variable / Expression | File:Line |
|-------|----------------------|-----------|
| Setter | `void Buffer::set_rdma_buffer_raw(void* ptr)` → `rdma_buffer_ptr = ptr` | `uccl_ep.cc:1865-1869` |
| Called from | Python binding / C++ init code that holds the actual allocation | `uccl_ep.cc:2060-2064` |

### Python

| Layer | Variable / Expression | File:Line |
|-------|----------------------|-----------|
| Allocation (CUDA) | `self.scratch = torch.zeros(num_rdma_bytes, dtype=torch.uint8, device=f"cuda:{device_index}")` | `buffer.py:92-94` |
| Allocation (ROCm) | `self.scratch = ep.get_rdma_buffer(num_rdma_bytes, device_index)` | `buffer.py:96` |
| Pointer | `rdma_buffer_ptr = self.scratch.data_ptr()` | `buffer.py:98` |
| Passed to C++ | `initialize_uccl(rdma_buffer_ptr, num_rdma_bytes, ...)` and later `self.runtime.set_rdma_buffer_raw(rdma_buffer_ptr)` | `buffer.py:100, 128` |
| `num_rdma_bytes` | Passed into `Buffer.__init__(..., num_rdma_bytes=...)`; typically from `config.get_rdma_buffer_size_hint(...)` at call sites | `buffer.py:59` |

### Chain summary (rdma_channel_data.send_buffer)

```
User / config
  → Python:  num_rdma_bytes  (e.g. from config.get_rdma_buffer_size_hint(hidden_bytes, num_ranks))
  → buffer.py:  Buffer(..., num_rdma_bytes)  →  self.scratch = torch.zeros(num_rdma_bytes, ...)  →  rdma_buffer_ptr = self.scratch.data_ptr()
  → buffer.py:  self.runtime.set_rdma_buffer_raw(rdma_buffer_ptr)
  → uccl_ep.cc:  Buffer::rdma_buffer_ptr  (member)
  → uccl_ep.cc:  internode_dispatch(..., rdma_buffer_ptr, ...)  →  uccl::internode::dispatch(..., rdma_buffer_ptr, ...)
  → internode.cu:  dispatch(..., void* rdma_buffer_ptr, ...)  →  kernel param rdma_buffer_ptr
  → kernel:  rdma_channel_data = SymBuffer<uint8_t>(rdma_buffer_ptr, ...)  →  send_ptr/send_buffer(idx) are offsets into that block
  → rdma_channel_data.send_buffer(dst_rdma_rank)  =  base of the contiguous send region for that rank
```

So **`rdma_channel_data.send_buffer`** is the **contiguous send staging buffer** for each destination rank, carved out of the **RDMA buffer** that was allocated in Python (`self.scratch`) and set on the C++ `Buffer` via **`set_rdma_buffer_raw`**.

---

## 3. Quick reference

| What | Origin |
|------|--------|
| **x** | User’s input tensor (Python `x` → `x.data_ptr()` → kernel `x`). |
| **rdma_channel_data.send_buffer** | A region inside the RDMA buffer whose base pointer is set by Python (`scratch.data_ptr()` → `set_rdma_buffer_raw`) and passed into the kernel as `rdma_buffer_ptr`; layout is built by `SymBuffer` in the kernel. |
