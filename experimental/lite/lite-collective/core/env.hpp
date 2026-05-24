// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ENV_HPP_
#define MSCCLPP_ENV_HPP_

#include <memory>
#include <string>

namespace mscclpp {

class Env;

/// Get the MSCCL++ environment.
/// @return A reference to the global environment object.
std::shared_ptr<Env> env();

/// The MSCCL++ environment. The constructor reads environment variables and
/// sets the corresponding fields. Use the env() function to get the environment
/// object.
class Env {
 public:
  /// Env name: `MSCCLPP_DEBUG`. The debug flag, one of VERSION, WARN, INFO,
  /// ABORT, or TRACE. Unset by default.
  const std::string debug;

  /// Env name: `MSCCLPP_DEBUG_SUBSYS`. The debug subsystem, a comma-separated
  /// list of subsystems to enable debug logging for. If the first character is
  /// '^', it inverts the mask, i.e., enables all subsystems except those
  /// specified. Possible values are INIT, COLL, P2P, SHM, NET, GRAPH, TUNING,
  /// ENV, ALLOC, CALL, MSCCLPP_EXECUTOR, MSCCLPP_NCCL, ALL. Unset by default.
  const std::string debugSubsys;

  /// Env name: `MSCCLPP_DEBUG_FILE`. A file path to write debug logs to. Unset
  /// by default.
  const std::string debugFile;

  /// Env name: `MSCCLPP_LOG_LEVEL`. One of DEBUG, INFO, WARN, or ERROR, in the
  /// order of severity (lower to higher level). A lower level is a superset of
  /// a higher level. Default is ERROR.
  const std::string logLevel;

  /// Env name: `MSCCLPP_LOG_SUBSYS`. The log subsystem, a comma-separated list
  /// of subsystems to enable logging for. Possible values are ENV, GPU, NET,
  /// CONN, EXEC, NCCL, ALL (default). If the first character is '^', it inverts
  /// the mask, i.e., enables all subsystems except those specified. For
  /// example, "^NET,CONN" enables all subsystems except NET and CONN.
  const std::string logSubsys;

  /// Env name: `MSCCLPP_LOG_FILE`. A file path to write log messages to. Unset
  /// by default.
  const std::string logFile;

  /// Env name: `MSCCLPP_HCA_DEVICES`. A comma-separated list of HCA devices to
  /// use for IB transport. i-th device in the list will be used for the i-th
  /// GPU in the system. If unset, it will use ibverbs APIs to find the devices
  /// automatically.
  const std::string hcaDevices;

  /// Env name: `MSCCLPP_IBV_SO`. The path to the libibverbs shared library to
  /// use. If unset, it will use the default libibverbs library found in the
  /// system.
  const std::string ibvSo;

  /// Env name: `MSCCLPP_IBV_MODE`. Selects the IB stack implementation for
  /// PortChannel. Allowed values:
  /// - "host": use the host stack with RDMA atomics (default).
  /// - "host-no-atomic": use the host stack with write-with-immediate signaling
  /// (no RDMA atomics).
  const std::string ibvMode;

  /// Env name: `MSCCLPP_HOSTID`. A string that uniquely identifies the host. If
  /// unset, it will use the hostname. This is used to determine whether the
  /// host is the same across different processes.
  const std::string hostid;

  /// Env name: `MSCCLPP_SOCKET_FAMILY`. The socket family to use for TCP
  /// sockets (used by TcpBootstrap and the Ethernet transport). Possible values
  /// are `AF_INET` (IPv4) and `AF_INET6` (IPv6). If unset, it will not force
  /// any family and will use the first one found.
  const std::string socketFamily;

  /// Env name: `MSCCLPP_SOCKET_IFNAME`. The interface name to use for TCP
  /// sockets (used by TcpBootstrap and the Ethernet transport). If unset, it
  /// will use the first interface found that matches the socket family.
  const std::string socketIfname;

  /// Env name: `MSCCLPP_COMM_ID`. To be deprecated; don't use this.
  const std::string commId;

  /// Env name: `MSCCLPP_CACHE_DIR`. The directory to use for caching execution
  /// plans and other temporary files. If unset, it defaults to
  /// `~/.cache/mscclpp`.
  const std::string cacheDir;

  /// Env name: `MSCCLPP_NPKIT_DUMP_DIR`. The directory to dump NPKIT traces to.
  /// If this is set, NPKIT will be enabled and will dump traces to this
  /// directory. Unset by default.
  const std::string npkitDumpDir;

  /// Env name: `MSCCLPP_CUDAIPC_USE_DEFAULT_STREAM`. If set to true, the CUDA
  /// IPC transport will use the default stream for all operations. If set to
  /// false, it will use a separate stream for each operation. This is an
  /// experimental feature and should be false in most cases. Default is false.
  bool const cudaIpcUseDefaultStream;

  /// Env name: `MSCCLPP_NCCL_LIB_PATH`. The path to the original NCCL/RCCL
  /// shared library. If set, it will be used as a fallback for NCCL operations
  /// in cases where the MSCCL++ NCCL cannot work.
  const std::string ncclSharedLibPath;

  /// Env name: `MSCCLPP_FORCE_NCCL_FALLBACK_OPERATION`. A comma-separated list
  /// of NCCL operations that should always use the fallback implementation,
  /// even if the MSCCL++ NCCL can handle them. This is useful for debugging
  /// purposes. Currently supports `all`, `broadcast`, `allreduce`,
  /// `reducescatter`, and `allgather`.
  const std::string forceNcclFallbackOperation;

  /// Env name: `MSCCLPP_NCCL_SYMMETRIC_MEMORY`. If set to true, it indicates
  /// that the application uses symmetric memory allocation across all ranks,
  /// making it safe to cache memory handles for all NCCL algorithms. If set to
  /// false, the system will either use non-zero-copy algorithms (when CUDA/HIP
  /// graphs are not enabled) or set up new connections every time (when
  /// CUDA/HIP graphs are enabled). This should be set to false if the
  /// application may call NCCL APIs on the same local buffer with different
  /// remote buffers, e.g., in the case of a dynamic communicator. Default is
  /// false.
  bool const ncclSymmetricMemory;

  /// Env name: `MSCCLPP_NCCL_SENDRECV_STAGING_BYTES`. The size in bytes of the
  /// pinned host staging buffers used by the NCCL compatibility layer for
  /// `ncclSend`/`ncclRecv` over PortChannel RDMA. Default is 268435456
  /// (256 MiB).
  int const ncclSendRecvStagingBytes;

  /// Env name: `MSCCLPP_FORCE_DISABLE_NVLS`. If set to true, it will disable
  /// the NVLS support in MSCCL++. Default is false.
  bool const forceDisableNvls;

  /// Env name: `MSCCLPP_DISABLE_CHANNEL_CACHE`. If set to true, NCCL-app style
  /// algorithms disable channel caching (legacy apps/nccl behavior). Default
  /// is false.
  bool const disableChannelCache;

 private:
  Env();

  friend std::shared_ptr<Env> env();
};

}  // namespace mscclpp

#endif  // MSCCLPP_ENV_HPP_
