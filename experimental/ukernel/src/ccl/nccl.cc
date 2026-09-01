#include "nccl.h"
#include "coll_config.h"
#include "coll_types.h"
#include "executor.h"
#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <ifaddrs.h>
#include <memory>
#include <netdb.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

using namespace UKernel::CCL;

struct ncclComm {
  std::unique_ptr<SprayExecutor> executor;
  int rank = 0;
  int nranks = 1;
  bool aborted = false;
  bool finalized = false;
  // Async completion tracking: submitted handles stay in pending until
  // reaped (reap runs at every collective entry and at destroy).
  // async_error latches the first failure observed while reaping —
  // reported once by ncclCommGetAsyncError, then cleared (NCCL
  // semantics).
  std::vector<CollectiveOpHandle> pending;
  ncclResult_t async_error = ncclSuccess;
};

static void reap_pending(ncclComm_t comm);  // defined next to run_coll

/* --- Unique ID format (wire-compatible layout) ---
 * bytes 0-3:   magic 0x554B4343 ("UKCC")
 * bytes 4-7:   exchanger port (uint32_t, network order)
 * bytes 8-11:  world size (uint32_t, network order)
 * bytes 12-75: exchanger IP (null-terminated, up to 64 bytes)
 * bytes 76-127: reserved (zero)
 */

static constexpr uint32_t kUniqueIdMagic = 0x554B4343;

static void pack_unique_id(ncclUniqueId* id, const char* ip, int port,
                           int nranks) {
  std::memset(id, 0, sizeof(*id));
  uint32_t* p = reinterpret_cast<uint32_t*>(id->internal);
  p[0] = htonl(kUniqueIdMagic);
  p[1] = htonl(static_cast<uint32_t>(port));
  p[2] = htonl(static_cast<uint32_t>(nranks));
  std::strncpy(id->internal + 12, ip ? ip : "", 63);
}

static bool unpack_unique_id(const ncclUniqueId* id, std::string& ip, int& port,
                              int& nranks) {
  const uint32_t* p = reinterpret_cast<const uint32_t*>(id->internal);
  if (ntohl(p[0]) != kUniqueIdMagic) return false;
  port = static_cast<int>(ntohl(p[1]));
  nranks = static_cast<int>(ntohl(p[2]));
  ip = std::string(id->internal + 12, 63);
  ip = ip.c_str();
  return port > 0 && port < 65536 && nranks >= 0;
}

static std::string detect_host_ip() {
  struct ifaddrs *ifaddr = nullptr, *ifa;
  if (getifaddrs(&ifaddr) != 0) return "127.0.0.1";
  std::string ip = "127.0.0.1";
  const char* ifname = std::getenv("NCCL_SOCKET_IFNAME");
  bool found = false;
  size_t ifname_len = ifname ? std::strlen(ifname) : 0;
  for (ifa = ifaddr; ifa; ifa = ifa->ifa_next) {
    if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
    if (std::strcmp(ifa->ifa_name, "lo") == 0) continue;
    if (ifname && std::strncmp(ifa->ifa_name, ifname, ifname_len) != 0) continue;
    found = true;
    char buf[INET_ADDRSTRLEN];
    inet_ntop(AF_INET,
              &reinterpret_cast<sockaddr_in*>(ifa->ifa_addr)->sin_addr, buf,
              sizeof(buf));
    ip = buf;
    break;
  }
  if (ifname && !found)
    std::fprintf(stderr, "[nccl] NCCL_SOCKET_IFNAME=%s matched no interface\n", ifname);
  freeifaddrs(ifaddr);
  return ip;
}

static int alloc_port() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return 17000 + (rand() % 1000);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = 0;
  int port = 17000 + (rand() % 1000);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
    socklen_t len = sizeof(addr);
    getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    port = ntohs(addr.sin_port);
  }
  close(fd);
  return port;
}

static ScalarType to_scalar(ncclDataType_t dt) {
  switch (dt) {
    case ncclInt8:    return ScalarType::Int8;
    case ncclUint8:   return ScalarType::UInt8;
    case ncclFloat16: return ScalarType::Float16;
    case ncclFloat32: return ScalarType::Float32;
    case ncclFloat64: return ScalarType::Float64;
    case ncclInt32:   return ScalarType::Int32;
    case ncclUint32:  return ScalarType::Int32;
    case ncclInt64:   return ScalarType::Int64;
    case ncclUint64:  return ScalarType::Int64;
    case ncclBfloat16: return ScalarType::BFloat16;
    default:          return ScalarType::Float32;
  }
}

static ReductionKind to_redop(ncclRedOp_t op) {
  switch (op) {
    case ncclSum:  return ReductionKind::Sum;
    case ncclProd: return ReductionKind::Prod;
    case ncclMax:  return ReductionKind::Max;
    case ncclMin:  return ReductionKind::Min;
    default:       return ReductionKind::Sum;
  }
}

// The device kernel has no unsigned reduce type; UInt32/UInt64 map to
// the signed bit pattern. Sum/Prod are then bit-exact, but Max/Min
// compare signed and silently produce wrong results on values with the
// high bit set — reject that combination honestly.
static bool unsupported_uint_redop(ncclDataType_t dt, ncclRedOp_t op) {
  return (dt == ncclUint32 || dt == ncclUint64) &&
         (op == ncclMax || op == ncclMin);
}

static const char* errstr(ncclResult_t r) {
  switch (r) {
    case ncclSuccess:            return "no error";
    case ncclUnhandledCudaError: return "unhandled cuda error";
    case ncclSystemError:        return "system error";
    case ncclInternalError:      return "internal error";
    case ncclInvalidArgument:    return "invalid argument";
    case ncclInvalidUsage:       return "invalid usage";
    case ncclRemoteError:        return "remote error";
    default:                     return "unknown";
  }
}

static bool s_rand_seeded = false;

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
  if (!uniqueId) return ncclInvalidArgument;
  if (!s_rand_seeded) { srand(static_cast<unsigned>(time(nullptr))); s_rand_seeded = true; }
  std::string ip = detect_host_ip();
  int port = alloc_port();
  pack_unique_id(uniqueId, ip.c_str(), port, 0);
  return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                              ncclUniqueId uniqueId, int rank) {
  if (!comm || nranks < 1 || rank < 0 || rank >= nranks)
    return ncclInvalidArgument;
  std::string ip;
  int port = 0, id_nranks = 0;
  if (!unpack_unique_id(&uniqueId, ip, port, id_nranks))
    return ncclInvalidArgument;
  if (id_nranks != 0 && id_nranks != nranks)
    return ncclInvalidArgument;
  auto c = std::make_unique<ncclComm>();
  c->rank = rank;
  c->nranks = nranks;
  int gpu_id = 0;
  if (gpuGetDevice(&gpu_id) != gpuSuccess) {
    std::fprintf(stderr, "[nccl] init r%d: gpuGetDevice failed\n", rank);
    return ncclUnhandledCudaError;
  }
  SprayExecutorConfig cfg;
  cfg.gpu_id = gpu_id;
  cfg.rank = rank;
  cfg.world_size = nranks;
  cfg.exchanger_ip = ip;
  cfg.exchanger_port = port;
  cfg.local_id = gpu_id;
  cfg.max_concurrent_runs = 16;
  cfg.device_idle_exit_us = 500;  // allow persistent kernel to exit when idle
  try {
    c->executor = SprayExecutor::create(cfg);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "[nccl] init r%d: %s\n", rank, e.what());
    return ncclSystemError;
  }
  *comm = c.release();
  return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm) {
    // Best-effort reap; handles still Running (or Failed-but-unquiesced)
    // stay with the executor, whose teardown stops the drain threads.
    reap_pending(comm);
    delete comm;
  }
  return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
  // NCCL semantics: abort only flags the communicator (subsequent
  // collectives fail fast in run_coll); teardown still belongs to
  // ncclCommDestroy, which apps call after abort.
  if (comm) {
    comm->aborted = true;
  }
  return ncclSuccess;
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  if (comm) comm->finalized = true;
  return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  if (!comm || !count) return ncclInvalidArgument;
  *count = comm->nranks;
  return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  if (!comm || !rank) return ncclInvalidArgument;
  *rank = comm->rank;
  return ncclSuccess;
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
  if (!comm || !asyncError) return ncclInvalidArgument;
  reap_pending(comm);
  *asyncError = comm->async_error;
  comm->async_error = ncclSuccess;  // reported once, then cleared
  return ncclSuccess;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  if (!comms || ndev < 1 || !devlist) return ncclInvalidArgument;
  ncclUniqueId id;
  ncclResult_t r = ncclGetUniqueId(&id);
  if (r != ncclSuccess) return r;
  pack_unique_id(&id, "127.0.0.1",
                 ntohl(reinterpret_cast<uint32_t*>(id.internal)[1]), ndev);
  std::vector<std::thread> threads;
  threads.reserve(ndev);
  std::vector<ncclResult_t> results(ndev, ncclSuccess);
  for (int i = 0; i < ndev; ++i) {
    threads.emplace_back([&, i] {
      gpuSetDevice(devlist[i]);
      results[i] = ncclCommInitRank(&comms[i], ndev, id, i);
    });
  }
  for (auto& t : threads) t.join();
  for (int i = 0; i < ndev; ++i) {
    if (results[i] != ncclSuccess) {
      for (int j = 0; j < ndev; ++j)
        if (results[j] == ncclSuccess) ncclCommDestroy(comms[j]);
      return results[i];
    }
  }
  return ncclSuccess;
}

// Reap finished handles in comm->pending: Completed → release; Failed →
// latch async_error (status() on a released handle reports Completed,
// so the message must be captured BEFORE release). A Failed run whose
// ops are still in flight cannot be released yet (the executor throws);
// it stays in pending and is retried on the next reap.
static void reap_pending(ncclComm_t comm) {
  auto& pending = comm->pending;
  for (size_t i = 0; i < pending.size();) {
    CollectiveOpHandle h = pending[i];
    auto st = comm->executor->status(h);
    if (st == CollectiveOpStatus::Running) {
      ++i;
      continue;
    }
    try {
      if (st == CollectiveOpStatus::Failed) {
        std::string msg = comm->executor->error_message(h);
        std::fprintf(stderr, "[nccl] r%d async error: %s\n", comm->rank,
                     msg.c_str());
        if (comm->async_error == ncclSuccess)
          comm->async_error = ncclRemoteError;
      }
      comm->executor->release(h);
    } catch (std::exception const&) {
      ++i;  // failed run not quiesced yet; retry on the next reap
      continue;
    }
    pending.erase(pending.begin() + static_cast<long>(i));
  }
}

static ncclResult_t run_coll(ncclComm_t comm, CollectiveConfig& cfg,
                              void* input, void* output, gpuStream_t stream) {
  if (comm->aborted) return ncclInvalidUsage;
  reap_pending(comm);
  // Signal aggregation: one Signal/WaitSignal pair per this many tiles
  // of a chunk pair (1 = per-tile). Fewer signals cut host dispatch and
  // drain cost; larger groups coarsen pipelining at group boundaries.
  static uint32_t const kSigGroupTiles = [] {
    char const* e = std::getenv("UK_CCL_SIG_GROUP_TILES");
    return e ? static_cast<uint32_t>(std::max(1L, std::stol(e))) : 1u;
  }();
  cfg.signal_group_tiles = kSigGroupTiles;
  // Fused reduce+copy: the RS RecvReduce task also forwards the reduced
  // shard to the next rank (device copy + device signal). With device
  // flags the per-tile signals are counted waits (any G); without them
  // the host-signal fallback fires one ring signal per tile, which is
  // only safe at G=1 (a plain wait completes on the first arrival).
  static bool const kFuseReduceCopy = [] {
    char const* e = std::getenv("UK_CCL_FUSE_REDUCE_COPY");
    return e && std::string(e) != "0";
  }();
  cfg.fuse_reduce_copy = kFuseReduceCopy;
  // Fused AG copy: the AG forward is a device copy task with an inline
  // completion flag (no CE, no host signal per hop).
  static bool const kFuseAgCopy = [] {
    char const* e = std::getenv("UK_CCL_FUSE_AG_COPY");
    return e && std::string(e) != "0";
  }();
  cfg.fuse_ag_copy = kFuseAgCopy;
  static bool const kA2aHybrid = [] {
    char const* e = std::getenv("UK_CCL_A2A_HYBRID");
    return e && std::string(e) != "0";
  }();
  cfg.a2a_hybrid = kA2aHybrid;
  static uint32_t const kA2aHybridCePct = [] {
    char const* e = std::getenv("UK_CCL_A2A_HYBRID_CE_PCT");
    uint32_t v = e ? static_cast<uint32_t>(std::max(0L, std::stol(e))) : 50u;
    return std::min(100u, v);
  }();
  cfg.a2a_hybrid_ce_pct = kA2aHybridCePct;
  // AllToAll per-peer send rotation (Latin square). Default on; 0
  // restores the ascending peer order (incast A/B on NVSwitch).
  static bool const kA2aRotate = [] {
    char const* e = std::getenv("UK_CCL_A2A_ROTATE");
    return !e || std::string(e) != "0";
  }();
  cfg.a2a_rotate_order = kA2aRotate;
  // Device-completion flags for fused tasks (default on; the per-slot
  // plain-store protocol needs no host-native atomics). Only meaningful
  // with a fused mode (the wait/flag pairing lives there).
  static bool const kDeviceFlags = [] {
    char const* e = std::getenv("UK_CCL_DEVICE_FLAGS");
    return !e || std::string(e) != "0";
  }();
  cfg.device_flags = kDeviceFlags && (kFuseReduceCopy || kFuseAgCopy);
  if ((kFuseReduceCopy || kFuseAgCopy) && !kDeviceFlags)
    cfg.signal_group_tiles = 1;
  CollectiveOpHandle h = 0;
  try {
    // prepare() is idempotent (deduped on shape + allocations) and
    // thread-safe inside the executor; the single-rank case is handled
    // there too. Call it unconditionally before every submit.
    comm->executor->prepare(cfg, input, output);
    h = comm->executor->submit(cfg, input, output, stream);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "[nccl] submit r%d: %s\n", comm->rank, e.what());
    return ncclInternalError;
  }
  if (h == kInvalidHandle) return ncclInternalError;
  // Truly async (NCCL semantics): completion is signaled on the stream
  // by the executor's WaitValue gate; the handle is reaped lazily at
  // the next collective entry / ncclCommDestroy.
  comm->pending.push_back(h);
  return ncclSuccess;
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, gpuStream_t stream) {
  if (!comm || !comm->executor || count == 0) return ncclInvalidArgument;
  if (unsupported_uint_redop(datatype, op)) {
    std::fprintf(stderr, "[nccl] r%d: unsigned Max/Min not supported\n",
                 comm->rank);
    return ncclInvalidUsage;
  }
  size_t elem_sz = scalar_type_size(to_scalar(datatype));
  if (elem_sz == 0) return ncclInvalidArgument;
  void* input = const_cast<void*>(sendbuff);
  if (count > SIZE_MAX / elem_sz) return ncclInvalidArgument;
  size_t nbytes = count * elem_sz;
  CollectiveConfig cfg;
  // Opt-in binary-tree allreduce via UK_CCL_TREE_THRESHOLD_BYTES
  // (default 0 = never; tree kicks in at nbytes >= threshold when set).
  // With nranks == 2 the tree degenerates to the ring's shape, so this
  // machine cannot tune the crossover — threshold calibration is left
  // to a larger-rank environment.
  static const size_t kTreeThreshold = []() {
    char const* env = std::getenv("UK_CCL_TREE_THRESHOLD_BYTES");
    return env ? std::stoull(env) : 0UL;
  }();
  // In-place stays on the ring — the tree has no snapshot support yet.
  cfg.kind = (kTreeThreshold > 0 && nbytes >= kTreeThreshold &&
              input != recvbuff)
                 ? CollKind::AllReduceTree
                 : CollKind::AllReduceRing;
  cfg.nranks = comm->nranks;
  cfg.rank = comm->rank;
  cfg.input_bytes = nbytes;
  cfg.output_bytes = nbytes;
  cfg.tile_bytes = adaptive_tile_bytes(nbytes);
  cfg.dtype = to_scalar(datatype);
  cfg.reduction = to_redop(op);
  return run_coll(comm, cfg, input, recvbuff, stream);
}

ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
                          ncclDataType_t datatype, ncclComm_t comm,
                          gpuStream_t stream) {
  if (!comm || !comm->executor || count == 0) return ncclInvalidArgument;
  size_t elem_sz = scalar_type_size(to_scalar(datatype));
  if (elem_sz == 0) return ncclInvalidArgument;
  // NCCL semantics: count is the element count per rank PAIR — the
  // total tensor per rank is nranks * count elements.
  if (count > SIZE_MAX / (static_cast<size_t>(comm->nranks) * elem_sz))
    return ncclInvalidArgument;
  size_t total_bytes = count * elem_sz * static_cast<size_t>(comm->nranks);
  CollectiveConfig cfg;
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.nranks = comm->nranks;
  cfg.rank = comm->rank;
  cfg.input_bytes = total_bytes;
  cfg.output_bytes = total_bytes;
  cfg.tile_bytes = adaptive_tile_bytes(total_bytes);
  cfg.dtype = to_scalar(datatype);
  // Out-of-place preferred (native semantics, no staging); in-place is
  // still accepted and runs the staged variant.
  cfg.inplace = (sendbuff == recvbuff);
  if (!cfg.inplace) {
    // Self-slice: my own partition lands in recvbuff via the copy
    // engine on the user stream (native sends partition r to itself).
    // Doing it here keeps the persistent worker out of AllToAll, so
    // BLK can stay 1 and every byte moves through IPC/copy-engine DMA.
    size_t off = static_cast<size_t>(comm->rank) * count * elem_sz;
    if (gpuMemcpyAsync(static_cast<char*>(recvbuff) + off,
                       static_cast<const char*>(sendbuff) + off,
                       count * elem_sz, gpuMemcpyDeviceToDevice, stream) !=
        gpuSuccess)
      return ncclUnhandledCudaError;
    cfg.external_self_slice = true;
  }
  return run_coll(comm, cfg, const_cast<void*>(sendbuff), recvbuff, stream);
}

// Compatibility alias under the historical shim-extension name.
ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
                          ncclDataType_t datatype, ncclComm_t comm,
                          gpuStream_t stream) {
  return ncclAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
}

static ncclResult_t unsupported(const char* fn) {
  std::fprintf(stderr, "[nccl] %s: not implemented\n", fn);
  return ncclInvalidUsage;
}

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclComm_t comm,
                           gpuStream_t stream) {
  if (!comm || !comm->executor || count == 0) return ncclInvalidArgument;
  size_t elem_sz = scalar_type_size(to_scalar(datatype));
  if (elem_sz == 0) return ncclInvalidArgument;
  // In-place (NCCL form): sendbuff points at the rank's own shard inside
  // recvbuff (sendbuff == recvbuff + rank*sendcount). Detect the overlap
  // and run the in-place algorithm variant.
  const char* sendp = static_cast<const char*>(sendbuff);
  const char* recvp = static_cast<const char*>(recvbuff);
  if (count > SIZE_MAX / (static_cast<size_t>(comm->nranks) * elem_sz))
    return ncclInvalidArgument;
  size_t out_bytes = count * elem_sz * static_cast<size_t>(comm->nranks);
  bool const inplace = sendp >= recvp && sendp < recvp + out_bytes;
  CollectiveConfig cfg;
  cfg.kind = CollKind::AllGatherRing;
  cfg.inplace = inplace;
  cfg.nranks = comm->nranks;
  cfg.rank = comm->rank;
  cfg.input_bytes = count * elem_sz;
  cfg.output_bytes = out_bytes;
  cfg.tile_bytes = adaptive_tile_bytes(out_bytes);
  cfg.dtype = to_scalar(datatype);
  return run_coll(comm, cfg, const_cast<void*>(sendbuff), recvbuff, stream);
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t count, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               gpuStream_t stream) {
  if (!comm || !comm->executor || count == 0) return ncclInvalidArgument;
  if (unsupported_uint_redop(datatype, op)) {
    std::fprintf(stderr, "[nccl] r%d: unsigned Max/Min not supported\n",
                 comm->rank);
    return ncclInvalidUsage;
  }
  size_t elem_sz = scalar_type_size(to_scalar(datatype));
  if (elem_sz == 0) return ncclInvalidArgument;
  // In-place (NCCL form): recvbuff points at the rank's own shard inside
  // sendbuff (recvbuff == sendbuff + rank*recvcount). The algorithm needs
  // no change (partials accumulate in Tmp; the final Tmp->Output[0] copy
  // lands on Input[offset(rank)] via the allocation-scoped offset), but
  // the executor must know not to treat the two distinct overlapping
  // pointers as out-of-place buffers.
  const char* sendp = static_cast<const char*>(sendbuff);
  const char* recvp = static_cast<const char*>(recvbuff);
  if (count > SIZE_MAX / (static_cast<size_t>(comm->nranks) * elem_sz))
    return ncclInvalidArgument;
  size_t in_bytes = count * elem_sz * static_cast<size_t>(comm->nranks);
  bool const inplace = recvp >= sendp && recvp < sendp + in_bytes;
  CollectiveConfig cfg;
  cfg.kind = CollKind::ReduceScatterRing;
  cfg.inplace = inplace;
  cfg.nranks = comm->nranks;
  cfg.rank = comm->rank;
  cfg.input_bytes = in_bytes;
  cfg.output_bytes = count * elem_sz;
  cfg.tile_bytes = adaptive_tile_bytes(in_bytes);
  cfg.dtype = to_scalar(datatype);
  cfg.reduction = to_redop(op);
  return run_coll(comm, cfg, const_cast<void*>(sendbuff), recvbuff, stream);
}

ncclResult_t ncclBroadcast(const void*, void*, size_t, ncclDataType_t, int,
                           ncclComm_t, gpuStream_t)
{ return unsupported("ncclBroadcast"); }
ncclResult_t ncclReduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t,
                        int, ncclComm_t, gpuStream_t)
{ return unsupported("ncclReduce"); }
ncclResult_t ncclSend(const void*, size_t, ncclDataType_t, int, ncclComm_t,
                      gpuStream_t)
{ return unsupported("ncclSend"); }
ncclResult_t ncclRecv(void*, size_t, ncclDataType_t, int, ncclComm_t,
                      gpuStream_t)
{ return unsupported("ncclRecv"); }
ncclResult_t ncclGroupStart(void) { return ncclSuccess; }
ncclResult_t ncclGroupEnd(void) { return ncclSuccess; }

ncclResult_t ncclMemAlloc(void** ptr, size_t size) {
  if (!ptr) return ncclInvalidArgument;
  return gpuMalloc(ptr, size) == gpuSuccess ? ncclSuccess : ncclSystemError;
}

ncclResult_t ncclMemFree(void* ptr) {
  return gpuFree(ptr) == gpuSuccess ? ncclSuccess : ncclSystemError;
}

ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size,
                              void** handle) {
  (void)comm; (void)buff; (void)size;
  if (!handle) return ncclInvalidArgument;
  *handle = buff;  // no-op registration; the shim never moves buffers
  return ncclSuccess;
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
  (void)comm; (void)handle;
  return ncclSuccess;
}

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar,
                                      ncclDataType_t datatype,
                                      ncclScalarResidence_t residence,
                                      ncclComm_t comm) {
  (void)op; (void)scalar; (void)datatype; (void)residence; (void)comm;
  return unsupported("ncclRedOpCreatePreMulSum");
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  (void)op; (void)comm;
  return ncclSuccess;
}

const char* ncclGetErrorString(ncclResult_t result) { return errstr(result); }

void ncclGetVersion(int* version) {
  if (version) *version = NCCL_VERSION_CODE;
}
