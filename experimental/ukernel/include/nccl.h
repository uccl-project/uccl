/*
 * NCCL-compatible API for ukernel CCL.
 * Drop-in replacement header — recompile NCCL-based applications against
 * this header and link with libukernel_nccl to use ukernel CCL instead.
 */
#ifndef UKERNEL_NCCL_H_
#define UKERNEL_NCCL_H_

#include "gpu_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque communicator handle. */
typedef struct ncclComm* ncclComm_t;

/* Return codes. */
typedef enum {
  ncclSuccess       = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError   = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage  = 5,
  ncclRemoteError   = 6,
  ncclInProgress    = 7,
  ncclNumResults    = 8,
} ncclResult_t;

/* Data types (subset matching NCCL). */
typedef enum {
  ncclInt8    = 0,
  ncclChar    = 0,
  ncclUint8   = 1,
  ncclInt32   = 2,
  ncclInt     = 2,
  ncclUint32  = 3,
  ncclInt64   = 4,
  ncclUint64  = 5,
  ncclFloat16 = 6,
  ncclHalf    = 6,
  ncclFloat32 = 7,
  ncclFloat   = 7,
  ncclFloat64 = 8,
  ncclDouble  = 8,
  ncclBfloat16 = 9,
  ncclNumTypes = 10,
} ncclDataType_t;

/* Reduction operations. */
typedef enum {
  ncclSum  = 0,
  ncclProd = 1,
  ncclMax  = 2,
  ncclMin  = 3,
  ncclAvg  = 4,  /* Sum then divide by world_size (ukernel extension). */
  ncclNumOps = 5,
} ncclRedOp_t;

/* Scalar residence for custom reduction ops (NCCL >= 2.29 ABI). */
typedef enum {
  ncclScalarDevice = 0,
  ncclScalarHostImmediate = 1
} ncclScalarResidence_t;

/* Bootstrap unique ID (128 bytes, matches NCCL wire format). */
#define NCCL_UNIQUE_ID_BYTES 128
typedef struct {
  char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

/* Versioning (compatible with nccl-tests preprocessor checks). */
#define NCCL_MAJOR 2
#define NCCL_MINOR 9
#define NCCL_PATCH 0
#define NCCL_SUFFIX "+ukernel"
#define NCCL_VERSION_CODE 2900
#define NCCL_VERSION(major, minor, patch) \
  (((major) * 1000) + ((minor) * 100) + (patch))

/* --- Communicator lifecycle --- */

/* Generate a unique ID for bootstrap. The caller (typically rank 0)
 * distributes this to all ranks out-of-band (MPI, file, env, etc.). */
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);

/* Initialize a communicator. All ranks must call this with the same
 * uniqueId and nranks. */
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                              ncclUniqueId uniqueId, int rank);

/* Destroy a communicator and free all associated resources. */
ncclResult_t ncclCommDestroy(ncclComm_t comm);

/* Abort a communicator (best-effort teardown for error paths). */
ncclResult_t ncclCommAbort(ncclComm_t comm);

/* Finalize the NCCL library (no-op for ukernel). */
ncclResult_t ncclCommFinalize(ncclComm_t comm);

/* Query communicator attributes. */
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);

/* Deprecated: init all communicators at once. */
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist);

/* --- Collective operations --- */

/* All-reduce: element-wise reduction across all ranks.
 * In-place when sendbuff == recvbuff. */
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, gpuStream_t stream);

/* All-to-all: equal-split scatter/gather across all ranks.
 * In-place only (sendbuff == recvbuff). */
ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
                          ncclDataType_t datatype, ncclComm_t comm,
                          gpuStream_t stream);

/* Barrier: collective synchronization point. */
ncclResult_t ncclBarrier(ncclComm_t comm, gpuStream_t stream);

/* Point-to-point (supported via GroupStart/GroupEnd). */
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, gpuStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, gpuStream_t stream);

/* Group operations for fused collectives. */
ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);

/* Additional collectives — unsupported, returns ncclInternalError. */
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           gpuStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, gpuStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm,
                           gpuStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               gpuStream_t stream);

/* --- Memory, buffer registration and custom ops (NCCL 2.19+ ABI) ---
 * These exist so binaries compiled against a modern NCCL header load and
 * run against the shim: ncclMemAlloc/Free wrap cudaMalloc/cudaFree,
 * registration is a no-op (the shim never moves buffers), and custom
 * pre-mul-sum ops are unsupported. */
ncclResult_t ncclMemAlloc(void** ptr, size_t size);
ncclResult_t ncclMemFree(void* ptr);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size,
                              void** handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle);
ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar,
                                      ncclDataType_t datatype,
                                      ncclScalarResidence_t residence,
                                      ncclComm_t comm);
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);

/* --- Query --- */

/* Return a human-readable string for a result code. */
const char* ncclGetErrorString(ncclResult_t result);

/* Return the version of the ukernel NCCL compat layer. */
void ncclGetVersion(int* version);

#ifdef __cplusplus
}
#endif

#endif /* UKERNEL_NCCL_H_ */
