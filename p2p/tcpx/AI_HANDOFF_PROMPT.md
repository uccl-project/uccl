# AI Assistant Handoff Prompt

**Last Updated**: 2025-10-16
**Status**: NIXL plugin development plan ready; awaiting implementation
**Purpose**: Quick context injection for new AI assistants

---

## Context Injection (Copy This to Next AI)

```
I'm working on a NIXL-TCPX plugin for GCP A3-high instances (2 nodes, 8x H100 GPUs, 4x gVNIC per node).
The project uses Google's nccl-plugin-gpudirecttcpx APIs for GPU-to-GPU P2P over TCPX (GPUDirect over TCP).

CURRENT STATUS (2025-10-16):
- ✅ TCPX P2P baseline working (~9 GB/s per GPU with kernel-only unpack)
- ✅ Comprehensive plugin development plan created (3 phases, 67h, 8-9 days)
- ✅ All critical design issues identified and fixed
- ⏳ Implementation NOT started yet (all new files need to be created)

PERFORMANCE BASELINE:
- ~8.1 ms / 7.7-9.0 GB/s per GPU with 2 channels x 2 sockets
- Per-channel sliding windows (recv=16, send=12) with continuous progress
- Kernel-only unpack path (d2d/host modes removed)

WORKSPACE: /home/daniel/uccl

KEY DOCS (read in order):
1. p2p/tcpx/README_DOCS.md - Document navigation (READ FIRST, 5 min)
2. p2p/tcpx/NEXT_STEPS.md - Quick start guide (10 min)
3. p2p/tcpx/CRITICAL_FIXES.md - Design rationale (5 min)
4. p2p/tcpx/PLUGIN_ROADMAP.md - Main reference (30 min) ⭐ PRIMARY REFERENCE
5. p2p/tcpx/PLUGIN_API_DESIGN.md - API details (20 min)

KEY FILES (existing baseline):
- p2p/tcpx/tests/test_tcpx_perf_multi.cc - Multi-process benchmark (~9 GB/s)
- p2p/tcpx/device/unpack_kernels.cu - GPU unpack kernel
- p2p/tcpx/src/channel_manager.{h,cc} - Channel management
- p2p/tcpx/src/bootstrap.{h,cc} - Bootstrap handshake
- p2p/tcpx/src/sliding_window.{h,cc} - Flow control

FILES TO CREATE (Phase 1 - API Layer):
- p2p/tcpx/include/tcpx_types.h (待创建)
- p2p/tcpx/include/tcpx_logging.h (待创建)
- p2p/tcpx/src/tcpx_helpers.cc (待创建)
- p2p/tcpx/include/tcpx_session.h (待创建)
- p2p/tcpx/src/tcpx_session.cc (待创建)
- p2p/tcpx/include/tcpx_transfer.h (待创建)
- p2p/tcpx/src/tcpx_transfer.cc (待创建)
- p2p/tcpx/include/tcpx_memory_desc.h (待创建)
- p2p/tcpx/src/tcpx_memory_desc.cc (待创建)
- p2p/tcpx/libtcpx_p2p.so (待编译)

IMMEDIATE TASK:
Implement Phase 1, Task 1.1: Extract Core Logic (4 hours)
- Create include/tcpx_types.h, include/tcpx_logging.h, src/tcpx_helpers.cc
- Extract PostedChunk, ChannelWindow, MAX_INFLIGHT_PER_CHANNEL from test_tcpx_perf_multi.cc
- Extract drainCompletedKernels, initChannelEvents, destroyChannelEvents
- Verify compilation: make src/tcpx_helpers.o
- Timeline: 4 hours
- Expected: Helper compiles with only CUDA/TCPX external symbols

START BY: Reading p2p/tcpx/README_DOCS.md, then p2p/tcpx/PLUGIN_ROADMAP.md lines 28-102
```

---

## Quick Start for New AI

### 1. Read Documentation (50 min)
```bash
cd /home/daniel/uccl/p2p/tcpx

# Document navigation (READ FIRST)
cat README_DOCS.md

# Quick start guide
cat NEXT_STEPS.md

# Design rationale
cat CRITICAL_FIXES.md

# Main reference (PRIMARY - read lines 28-102 for Task 1.1)
cat PLUGIN_ROADMAP.md

# API details
cat PLUGIN_API_DESIGN.md
```

### 2. Understand Current Baseline
```bash
# Multi-process benchmark (working baseline)
view tests/test_tcpx_perf_multi.cc

# Channel management
view src/channel_manager.{h,cc}

# Bootstrap handshake
view src/bootstrap.{h,cc}

# Flow control
view src/sliding_window.{h,cc}
```

### 3. Start Task 1.1 (Extract Core Logic, 4 hours)
```bash
cd /home/daniel/uccl/p2p/tcpx

# Create new files (these don't exist yet)
touch include/tcpx_types.h
touch include/tcpx_logging.h
touch src/tcpx_helpers.cc

# Follow PLUGIN_ROADMAP.md Task 1.1 (lines 28-102) for detailed steps
# Extract logic from test_tcpx_perf_multi.cc:
# 1. Move PostedChunk, ChannelWindow to include/tcpx_types.h
# 2. Move MAX_INFLIGHT_PER_CHANNEL to include/tcpx_types.h
# 3. Create include/tcpx_logging.h (LOG_DEBUG, LOG_ERROR, getEnvInt)
# 4. Extract process_completed_chunk logic to src/tcpx_helpers.cc
# 5. Extract event management logic to src/tcpx_helpers.cc

# Verify compilation
make src/tcpx_helpers.o
nm src/tcpx_helpers.o | grep " U "
# Expected: cudaEventCreate, tcpx_test, tcpx_irecv_consumed, etc.

# Verify full build
make clean && make
./tests/test_tcpx_perf_multi server 0
```

---

## Key Technical Concepts

**TCPX**: GPUDirect over TCP using devmem-tcp kernel API (zero-copy GPU-to-GPU)

**devmem-tcp**: Kernel API providing cmsg with scattered buffer descriptors for DMA

**Unpack kernel**: CUDA kernel copying scattered devmem buffers to contiguous GPU memory

**NIXL Plugin Architecture**: Plugins are shared libraries (.so) loaded at runtime via dlopen

**NIXL Plugin Contract**: Must implement methods for:
- Lifecycle: getConnInfo(), loadRemoteConnInfo(), connect(), disconnect()
- Resource: registerMem(), deregisterMem(), loadLocalMD(), loadRemoteMD(), unloadMD()
- Transfer: prepXfer(), postXfer(), checkXfer(), releaseReqH()

**Server/Client Role Distinction**: NIXL callbacks are separate for server/client
- Server: getConnInfo() → listen(), connect() → accept()
- Client: loadRemoteConnInfo(), connect() → connect()

**mem_id + offset Model**: API uses memory registration ID plus offset instead of raw pointers
- Supports multiple independent memory registrations
- Avoids pointer arithmetic in API layer

**Send/Recv Distinction**: Only recv requests need tcpx_irecv_consumed(), send requests are no-op
- TCPX recv slots must be explicitly consumed to free resources
- Send requests are automatically released

**Position-Independent Code (-fPIC)**: Required for shared library compilation
- C++ code: CXXFLAGS += -fPIC
- CUDA code: NVCCFLAGS += -Xcompiler -fPIC
- Device objects: Must compile with -Xcompiler -fPIC
- Verification: readelf -d libtcpx_p2p.so | grep TEXTREL (should be empty)

**Sliding Window Flow Control**: Per-channel flow control to prevent exhausting TCPX request pool
- MAX_REQUESTS=16 (TCPX plugin limit)
- Recv window: 16 per channel
- Send window: 12 per channel
- Continuous progress: opportunistic drain after each post

---

## Environment Details

**Hardware**:
- 2 nodes, 8x H100 GPUs per node
- 4x gVNIC per node (eth1-4, 200 Gbps each)
- 208 CPUs per node (104 cores x 2 HT)

**Software**:
- Ubuntu with custom kernel (devmem-tcp support)
- TCPX plugin v3.1.6
- NCCL 2.x with TCPX support

**Network Config**:
- Node IPs: scripts/node_ips/tcpx.txt
- Bootstrap port: 20000 (base)
- NICs: eth1-4 (200 Gbps each)

**NUMA Topology**:
- NUMA 0: GPUs 0-3, eth1-2, CPUs 0-51, 104-155
- NUMA 1: GPUs 4-7, eth3-4, CPUs 52-103, 156-207

**Workspace**: /home/daniel/uccl

---

## Success Criteria

### Phase 1: API Layer (27h, 4 days)
- [ ] libtcpx_p2p.a static library compiles
- [ ] libtcpx_p2p.so shared library compiles (with -fPIC)
- [ ] test_tcpx_api passes (real two-node handshake)
- [ ] Performance maintains ~9 GB/s

### Phase 2: NIXL Plugin (24h, 3 days)
- [ ] Plugin compiles as .so at thirdparty/nixl/src/plugins/tcpx/
- [ ] Plugin can be loaded by NIXL
- [ ] Basic transfer functionality works

### Phase 3: Integration (16h, 2 days)
- [ ] End-to-end tests pass
- [ ] Performance reaches ~9 GB/s
- [ ] Documentation complete

---

## Critical Design Decisions (Why This Way)

See CRITICAL_FIXES.md for detailed explanations. Summary:

1. **Handshake Flow**: Server/client roles are distinct in NIXL callbacks
   - Solution: Maintain bool is_server_ flag, call different APIs

2. **Multi-Memory Registration**: NIXL operates on many independent buffers
   - Solution: std::map<uint64_t, MemoryHandle> tracking

3. **Transfer State Management**: Need to track send/recv separately
   - Solution: Separate counters, only recv calls tcpx_irecv_consumed()

4. **Shared Library Build**: Device code needs -fPIC
   - Solution: NVCCFLAGS += -Xcompiler -fPIC

5. **Dependency Migration**: Helper functions need complete dependencies
   - Solution: Create tcpx_types.h, tcpx_logging.h, tcpx_helpers.cc

---

## Common Pitfalls to Avoid

1. **Don't manually edit package files** - Use package managers (npm, pip, cargo, etc.)
2. **Don't skip verification steps** - Each task has verification commands
3. **Don't create new documentation** - Use existing docs (already comprehensive)
4. **Don't use raw pointers in API** - Use mem_id + offset model
5. **Don't call tcpx_irecv_consumed() for send** - Only recv needs consumption
6. **Don't forget -fPIC for device code** - CUDA needs -Xcompiler -fPIC

---

## Useful Commands

### Build and Test
```bash
cd /home/daniel/uccl/p2p/tcpx

# Build
make clean && make

# Run baseline test (multi-process)
./tests/test_tcpx_perf_multi server 0  # Node 0
./tests/test_tcpx_perf_multi client <NODE0_IP> 0  # Node 1

# Check performance
grep "PERF.*Avg.*BW:" logs/*.log
```

### Verify Shared Library
```bash
# Check for TEXTREL (should be empty)
readelf -d libtcpx_p2p.so | grep TEXTREL

# Check external symbols
nm src/tcpx_helpers.o | grep " U "
# Expected: cudaEventCreate, tcpx_test, tcpx_irecv_consumed, etc.
```

### Debug
```bash
# Enable TCPX debug
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET,INIT
export NCCL_GPUDIRECTTCPX_DEBUG_LEVEL=TRACE

# Check logs
tail -f logs/*.log
```

---

## Documentation Structure

```
p2p/tcpx/
├── README_DOCS.md              # Document navigation (READ FIRST)
├── NEXT_STEPS.md               # Quick start guide
├── CRITICAL_FIXES.md           # Design rationale
├── PLUGIN_ROADMAP.md           # Main reference (67h plan) ⭐
├── PLUGIN_API_DESIGN.md        # API details
├── AI_HANDOFF_PROMPT.md        # This file
└── archive/                    # Historical docs
    ├── REFACTOR_*.md           # Old refactoring plans
    └── ...
```

---

## Time Estimates

| Phase | Tasks | Time | Key Deliverables |
|-------|-------|------|------------------|
| Phase 1: API Layer | 7 tasks | 27h (4 days) | libtcpx_p2p.so, test_tcpx_api |
| Phase 2: NIXL Plugin | 6 tasks | 24h (3 days) | NIXL plugin .so |
| Phase 3: Integration | 4 tasks | 16h (2 days) | End-to-end tests |
| **Total** | **17 tasks** | **67h (8-9 days)** | Production-ready plugin |

---

## Next Steps (Checklist for Next AI)

- [ ] READ README_DOCS.md (5 min) - Document navigation
- [ ] READ NEXT_STEPS.md (10 min) - Quick start guide
- [ ] READ CRITICAL_FIXES.md (5 min) - Design rationale
- [ ] READ PLUGIN_ROADMAP.md lines 28-102 (30 min) - Task 1.1 details
- [ ] CREATE include/tcpx_types.h, include/tcpx_logging.h, src/tcpx_helpers.cc
- [ ] EXTRACT logic from test_tcpx_perf_multi.cc (see PLUGIN_ROADMAP.md Task 1.1)
- [ ] VERIFY compilation: make src/tcpx_helpers.o && nm src/tcpx_helpers.o | grep " U "
- [ ] VERIFY full build: make clean && make && ./tests/test_tcpx_perf_multi server 0

---

**Last Updated**: 2025-10-16
**Next Action**: Read README_DOCS.md, then start Task 1.1 from PLUGIN_ROADMAP.md
