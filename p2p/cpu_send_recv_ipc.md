# CPU support for send_ipc and recv_ipc

## Problem
send_ipc/recv_ipc only work with GPU buffers. gpuIpcGetMemHandle() fails on CPU memory,
gpuIpcOpenMemHandle() has no CPU equivalent, and gpuMemcpyAsync uses DeviceToDevice only.

## Approach: Zero-copy with protocol role reversal
The side with the GPU buffer always creates the IPC handle. The side with the CPU buffer does the copy.

### Combinations supported
| Sender | Receiver | Who creates handle | Copy direction | Messages |
|--------|----------|--------------------|---------------|----------|
| GPU    | GPU      | receiver (current) | D2D           | 2        |
| CPU    | GPU      | receiver           | H2D           | 2        |
| GPU    | CPU      | **sender**         | D2H           | 3        |

CPU-to-CPU is not supported (CHECK/abort).

### Protocol details

**Case 1: CPU sender -> GPU receiver (minimal change)**
- recv_ipc: same as today - gpuIpcGetMemHandle, send IPC_HANDLE{is_host=false}, wait COMPLETION
- send_ipc: receive handle, open it, gpuMemcpyAsync(H2D), send COMPLETION

**Case 2: GPU sender -> CPU receiver (role reversal)**
- recv_ipc: detect CPU buffer, send IPC_HANDLE{is_host=true} (no valid handle)
- send_ipc: see is_host=true, create own IPC handle, send IPC_HANDLE back to receiver
- recv_ipc: receive sender's handle, open it, gpuMemcpyAsync(D2H), close handle, send COMPLETION
- send_ipc: wait for COMPLETION

## Implementation plan

### C++ changes (engine.h, engine.cc)
1. Add `bool is_host` field to IpcTransferInfo
2. In recv_ipc: detect memory type via get_dev_idx(data), branch on GPU vs CPU
3. In send_ipc: after receiving first msg, branch on info.is_host
4. Use ipc_streams_[gpu_idx] for stream selection regardless of which side copies

### Benchmark changes (benchmark_uccl.py)
1. Add --sender-device and --receiver-device flags (default: gpu)
2. rank 0 (client/sender) uses --sender-device, rank 1 (server/receiver) uses --receiver-device
3. Test combinations: gpu/gpu, cpu/gpu, gpu/cpu
