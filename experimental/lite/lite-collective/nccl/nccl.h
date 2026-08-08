// Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
// Modifications Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef NCCL_H_
#define NCCL_H_

#include "gpu.hpp"
#include "gpu_data_types.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include <limits.h>
/* Opaque handle to communicator */
typedef struct ncclComm* ncclComm_t;
typedef struct ncclWindow* ncclWindow_t;
#define NCCL_COMM_NULL NULL

#define NCCL_UNIQUE_ID_BYTES 128
typedef struct {
  char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

/* Error type */
typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6,
  ncclInProgress = 7,
  ncclNumResults = 8
} ncclResult_t;

#define NCCL_CONFIG_UNDEF_INT INT_MIN
#define NCCL_CONFIG_UNDEF_PTR NULL
#define NCCL_SPLIT_NOCOLOR -1

/* Communicator configuration. Users can assign value to attributes to specify
 * the behavior of a communicator. */
typedef struct ncclConfig_v21700 {
  /* attributes that users should never touch. */
  size_t size;
  unsigned int magic;
  unsigned int version;
  /* attributes that users are able to customize. */
  int blocking;
  int cgaClusterSize;
  int minCTAs;
  int maxCTAs;
  char const* netName;
  int splitShare;
} ncclConfig_t;

/* Config initializer must be assigned to initialize config structure when it is
 * created. Not initialized config will result in NCCL error. */
#define NCCL_CONFIG_INITIALIZER                                                \
  {                                                                            \
    sizeof(ncclConfig_t),                                 /* size */           \
        0xcafebeef,                                       /* magic */          \
        NCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH), /* version */        \
        NCCL_CONFIG_UNDEF_INT,                            /* blocking */       \
        NCCL_CONFIG_UNDEF_INT,                            /* cgaClusterSize */ \
        NCCL_CONFIG_UNDEF_INT,                            /* minCTAs */        \
        NCCL_CONFIG_UNDEF_INT,                            /* maxCTAs */        \
        NCCL_CONFIG_UNDEF_PTR,                            /* netName */        \
        NCCL_CONFIG_UNDEF_INT                             /* splitShare */     \
  }

/* This struct will be used by ncclGroupSimulateEnd() API to query information
 * about simulation. */
typedef struct ncclSimInfo_v22200 {
  size_t size;
  unsigned int magic;
  unsigned int version;
  float estimatedTime;
} ncclSimInfo_t;

/* NCCL malloc and free function for all types of NCCL optimizations
 * (e.g. user buffer registration). The actual allocated size might
 * be larger than requested due to granularity requirement. */
ncclResult_t ncclMemAlloc(void** ptr, size_t size);
ncclResult_t pncclMemAlloc(void** ptr, size_t size);

ncclResult_t ncclMemFree(void* ptr);
ncclResult_t pncclMemFree(void* ptr);

/* Return the NCCL_VERSION_CODE of the NCCL library in the supplied integer.
 * This integer is coded with the MAJOR, MINOR and PATCH level of the
 * NCCL library
 */
ncclResult_t ncclGetVersion(int* version);
ncclResult_t pncclGetVersion(int* version);

/* Generates an Id to be used in ncclCommInitRank. ncclGetUniqueId should be
 * called once and the Id should be distributed to all ranks in the
 * communicator before calling ncclCommInitRank. */
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t pncclGetUniqueId(ncclUniqueId* uniqueId);

/* Create a new communicator (multi thread/process version) with a configuration
 * set by users. */
ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks,
                                    ncclUniqueId commId, int rank,
                                    ncclConfig_t* config);
ncclResult_t pncclCommInitRankConfig(ncclComm_t* comm, int nranks,
                                     ncclUniqueId commId, int rank,
                                     ncclConfig_t* config);

/* Creates a new communicator (multi thread/process version).
 * rank must be between 0 and nranks-1 and unique within a communicator clique.
 * Each rank is associated to a CUDA device, which has to be set before calling
 * ncclCommInitRank.
 * ncclCommInitRank implicitly syncronizes with other ranks, so it must be
 * called by different threads/processes or use ncclGroupStart/ncclGroupEnd. */
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId,
                              int rank);
ncclResult_t pncclCommInitRank(ncclComm_t* comm, int nranks,
                               ncclUniqueId commId, int rank);

/* Creates a clique of communicators (single process version).
 * This is a convenience function to create a single-process communicator
 * clique. Returns an array of ndev newly initialized communicators in comm.
 * comm should be pre-allocated with size at least ndev*sizeof(ncclComm_t).
 * If devlist is NULL, the first ndev CUDA devices are used.
 * Order of devlist defines user-order of processors within the communicator. */
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, int const* devlist);
ncclResult_t pncclCommInitAll(ncclComm_t* comm, int ndev, int const* devlist);

/* Finalize a communicator. ncclCommFinalize flushes all issued communications,
 * and marks communicator state as ncclInProgress. The state will change to
 * ncclSuccess when the communicator is globally quiescent and related resources
 * are freed; then, calling ncclCommDestroy can locally free the rest of the
 * resources (e.g. communicator itself) without blocking. */
ncclResult_t ncclCommFinalize(ncclComm_t comm);
ncclResult_t pncclCommFinalize(ncclComm_t comm);

/* Frees local resources associated with communicator object. */
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t pncclCommDestroy(ncclComm_t comm);

/* Frees resources associated with communicator object and aborts any operations
 * that might still be running on the device. */
ncclResult_t ncclCommAbort(ncclComm_t comm);
ncclResult_t pncclCommAbort(ncclComm_t comm);

/* Creates one or more communicators from an existing one.
 * Ranks with the same color will end up in the same communicator.
 * Within the new communicator, key will be used to order ranks.
 * NCCL_SPLIT_NOCOLOR as color will indicate the rank will not be part of any
 * group and will therefore return a NULL communicator. If config is NULL, the
 * new communicator will inherit the original communicator's configuration*/
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key,
                           ncclComm_t* newcomm, ncclConfig_t* config);
ncclResult_t pncclCommSplit(ncclComm_t comm, int color, int key,
                            ncclComm_t* newcomm, ncclConfig_t* config);

/* Shrink existing communicator.
 * Ranks in excludeRanksList will be removed form the existing communicator.
 * Within the new communicator, ranks will be re-ordered to fill the gap of
 * removed ones. If config is NULL, the new communicator will inherit the
 * original communicator's configuration The flag enables NCCL to adapt to
 * various states of the parent communicator, see NCCL_SHRINK flags.*/
ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList,
                            int excludeRanksCount, ncclComm_t* newcomm,
                            ncclConfig_t* config, int shrinkFlags);
ncclResult_t pncclCommShrink(ncclComm_t comm, int* excludeRanksList,
                             int excludeRanksCount, ncclComm_t* newcomm,
                             ncclConfig_t* config, int shrinkFlags);

/* Creates a new communicator (multi thread/process version), similar to
 * ncclCommInitRankConfig. Allows to use more than one ncclUniqueId (up to one
 * per rank), indicated by nId, to accelerate the init operation. The number of
 * ncclUniqueIds and their order must be the same for every rank.
 */
ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks,
                                      int myrank, int nId,
                                      ncclUniqueId* commIds,
                                      ncclConfig_t* config);
ncclResult_t pncclCommInitRankScalable(ncclComm_t* newcomm, int nranks,
                                       int myrank, int nId,
                                       ncclUniqueId* commIds,
                                       ncclConfig_t* config);

/* Returns a string for each error code. */
char const* ncclGetErrorString(ncclResult_t result);
char const* pncclGetErrorString(ncclResult_t result);

/* Returns a human-readable message of the last error that occurred.
 * comm is currently unused and can be set to NULL
 */
char const* ncclGetLastError(ncclComm_t comm);
char const* pncclGetLastError(ncclComm_t comm);

/* Checks whether the comm has encountered any asynchronous errors */
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);
ncclResult_t pncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);

/* Gets the number of ranks in the communicator clique. */
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count);
ncclResult_t pncclCommCount(const ncclComm_t comm, int* count);

/* Returns the cuda device number associated with the communicator. */
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device);
ncclResult_t pncclCommCuDevice(const ncclComm_t comm, int* device);

/* Returns the user-ordered "rank" associated with the communicator. */
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank);
ncclResult_t pncclCommUserRank(const ncclComm_t comm, int* rank);

/* Register CUDA buffer for zero-copy operation */
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size,
                              void** handle);
ncclResult_t pncclCommRegister(const ncclComm_t comm, void* buff, size_t size,
                               void** handle);

/* Deregister CUDA buffer */
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle);
ncclResult_t pncclCommDeregister(const ncclComm_t comm, void* handle);

/* Register memory window  */
ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size,
                                    ncclWindow_t* win, int winFlags);
ncclResult_t pncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size,
                                     ncclWindow_t* win, int winFlags);

/* Deregister symmetric memory */
ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win);
ncclResult_t pncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win);

/* Reduction operation selector */
typedef enum { ncclNumOps_dummy = 5 } ncclRedOp_dummy_t;
typedef enum {
  ncclSum = 0,
  ncclProd = 1,
  ncclMax = 2,
  ncclMin = 3,
  ncclAvg = 4,
  /* ncclNumOps: The number of built-in ncclRedOp_t values. Also
   * serves as the least possible value for dynamic ncclRedOp_t's
   * as constructed by ncclRedOpCreate*** functions. */
  ncclNumOps = 5,
  /* ncclMaxRedOp: The largest valid value for ncclRedOp_t.
   * It is defined to be the largest signed value (since compilers
   * are permitted to use signed enums) that won't grow
   * sizeof(ncclRedOp_t) when compared to previous NCCL versions to
   * maintain ABI compatibility. */
  ncclMaxRedOp = 0x7fffffff >> (32 - 8 * sizeof(ncclRedOp_dummy_t))
} ncclRedOp_t;

/* Data types */
typedef enum {
  ncclInt8 = 0,
  ncclChar = 0,
  ncclUint8 = 1,
  ncclInt32 = 2,
  ncclInt = 2,
  ncclUint32 = 3,
  ncclInt64 = 4,
  ncclUint64 = 5,
  ncclFloat16 = 6,
  ncclHalf = 6,
  ncclFloat32 = 7,
  ncclFloat = 7,
  ncclFloat64 = 8,
  ncclDouble = 8,
  ncclBfloat16 = 9,
  ncclFloat8e4m3 = 10,
  ncclFloat8e5m2 = 11,
  ncclNumTypes = 12
} ncclDataType_t;

static inline size_t ncclTypeSize(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
      return 2;
    case ncclInt32:
    case ncclUint32:
      return 4;
    case ncclInt64:
    case ncclUint64:
      return 8;
    case ncclFloat32:
      return 4;
    case ncclFloat64:
      return 8;
    case ncclBfloat16:
      return 2;
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
      return 1;
    case ncclNumTypes:
      return 0;
  }
  return 0;
}

/* ncclScalarResidence_t: Location and dereferencing logic for scalar arguments.
 */
typedef enum {
  /* ncclScalarDevice: The scalar is in device-visible memory and will be
   * dereferenced while the collective is running. */
  ncclScalarDevice = 0,

  /* ncclScalarHostImmediate: The scalar is in host-visible memory and will be
   * dereferenced before the ncclRedOpCreate***() function returns. */
  ncclScalarHostImmediate = 1
} ncclScalarResidence_t;

/*
 * ncclRedOpCreatePreMulSum
 *
 * Creates a new reduction operator which pre-multiplies input values by a given
 * scalar locally before reducing them with peer values via summation. For use
 * only with collectives launched against *comm* and *datatype*. The
 * *residence* argument indicates how/when the memory pointed to by *scalar*
 * will be dereferenced. Upon return, the newly created operator's handle
 * is stored in *op*.
 */
ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar,
                                      ncclDataType_t datatype,
                                      ncclScalarResidence_t residence,
                                      ncclComm_t comm);
ncclResult_t pncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar,
                                       ncclDataType_t datatype,
                                       ncclScalarResidence_t residence,
                                       ncclComm_t comm);

/*
 * ncclRedOpDestroy
 *
 * Destroys the reduction operator *op*. The operator must have been created by
 * ncclRedOpCreatePreMul with the matching communicator *comm*. An operator may
 * be destroyed as soon as the last NCCL function which is given that operator
 * returns.
 */
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);
ncclResult_t pncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);

/*
 * Collective communication operations
 *
 * Collective communication operations must be called separately for each
 * communicator in a communicator clique.
 *
 * They return when operations have been enqueued on the CUDA stream.
 *
 * Since they may perform inter-CPU synchronization, each call has to be done
 * from a different thread or process, or need to use Group Semantics (see
 * below).
 */

/*
 * Reduce
 *
 * Reduces data arrays of length count in sendbuff into recvbuff using op
 * operation.
 * recvbuff may be NULL on all calls except for root device.
 * root is the rank (not the CUDA device) where data will reside after the
 * operation is complete.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
ncclResult_t ncclReduce(void const* sendbuff, void* recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclReduce(void const* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op, int root,
                         ncclComm_t comm, cudaStream_t stream);

/*
 * (deprecated) Broadcast (in-place)
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the CUDA device) where data resides before the
 * operation is started.
 *
 * This operation is implicitly in place.
 */
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype,
                       int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclBcast(void* buff, size_t count, ncclDataType_t datatype,
                        int root, ncclComm_t comm, cudaStream_t stream);

/*
 * Broadcast
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the CUDA device) where data resides before the
 * operation is started.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
ncclResult_t ncclBroadcast(void const* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream);
ncclResult_t pncclBroadcast(void const* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, int root, ncclComm_t comm,
                            cudaStream_t stream);

/*
 * All-Reduce
 *
 * Reduces data arrays of length count in sendbuff using op operation, and
 * leaves identical copies of result on each recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
ncclResult_t ncclAllReduce(void const* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclAllReduce(void const* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op,
                            ncclComm_t comm, cudaStream_t stream);

/*
 * Reduce-Scatter
 *
 * Reduces data in sendbuff using op operation and leaves reduced result
 * scattered over the devices so that recvbuff on rank i will contain the i-th
 * block of the result.
 * Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
 * should have a size of at least nranks*recvcount elements.
 *
 * In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
 */
ncclResult_t ncclReduceScatter(void const* sendbuff, void* recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream);
ncclResult_t pncclReduceScatter(void const* sendbuff, void* recvbuff,
                                size_t recvcount, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm_t comm,
                                cudaStream_t stream);

/*
 * All-Gather
 *
 * Each device gathers sendcount values from other GPUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
ncclResult_t ncclAllGather(void const* sendbuff, void* recvbuff,
                           size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclAllGather(void const* sendbuff, void* recvbuff,
                            size_t sendcount, ncclDataType_t datatype,
                            ncclComm_t comm, cudaStream_t stream);

/*
 * Send
 *
 * Send data from sendbuff to rank peer.
 *
 * Rank peer needs to call ncclRecv with the same datatype and the same count
 * from this rank.
 *
 * This operation is blocking for the GPU. If multiple ncclSend and ncclRecv
 * operations need to progress concurrently to complete, they must be fused
 * within a ncclGroupStart/ ncclGroupEnd section.
 */
ncclResult_t ncclSend(void const* sendbuff, size_t count,
                      ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream);
ncclResult_t pncclSend(void const* sendbuff, size_t count,
                       ncclDataType_t datatype, int peer, ncclComm_t comm,
                       cudaStream_t stream);

/*
 * Receive
 *
 * Receive data from rank peer into recvbuff.
 *
 * Rank peer needs to call ncclSend with the same datatype and the same count to
 * this rank.
 *
 * This operation is blocking for the GPU. If multiple ncclSend and ncclRecv
 * operations need to progress concurrently to complete, they must be fused
 * within a ncclGroupStart/ ncclGroupEnd section.
 */
ncclResult_t pncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                       int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);

/* All-To-All
 *
 * Device (i) send (j)th block of data to device (j) and be placed as (i)th
 * block. Each block for sending/receiving has count elements, which means
 * that recvbuff and sendbuff should have a size of nranks*count elements.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
ncclResult_t ncclAllToAll(void const* sendbuff, void* recvbuff, size_t count,
                          ncclDataType_t datatype, ncclComm_t comm,
                          cudaStream_t stream);
ncclResult_t pncclAllToAll(void const* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclComm_t comm,
                           cudaStream_t stream);

/*!All-To-Allv
 * Device (i) sends sendcounts[j] of data from offset sdispls[j]
 * to device (j). At the same time, device (i) receives recvcounts[j] of data
 * from device (j) to be placed at rdispls[j].
 * sendcounts, sdispls, recvcounts and rdispls are all measured in the units
 * In-place operation will happen if sendbuff == recvbuff.
 */
ncclResult_t ncclAllToAllv(void const* sendbuff, const size_t sendcounts[],
                           const size_t sdispls[], void* recvbuff,
                           const size_t recvcounts[], const size_t rdispls[],
                           ncclDataType_t datatype, ncclComm_t comm,
                           cudaStream_t stream);
ncclResult_t pncclAllToAllv(void const* sendbuff, const size_t sendcounts[],
                            const size_t sdispls[], void* recvbuff,
                            const size_t recvcounts[], const size_t rdispls[],
                            ncclDataType_t datatype, ncclComm_t comm,
                            cudaStream_t stream);
/*! @brief Opaque handle to MSCCL algorithm */
typedef int mscclAlgoHandle_t;

/*! @brief MSCCL Load Algorithm
 *
 * @details Load MSCCL algorithm file specified in mscclAlgoFilePath and return
 * its handle via mscclAlgoHandle. This API is expected to be called by MSCCL
 * scheduler instead of end users.
 */
ncclResult_t mscclLoadAlgo(char const* mscclAlgoFilePath,
                           mscclAlgoHandle_t* mscclAlgoHandle, int rank);
ncclResult_t pmscclLoadAlgo(char const* mscclAlgoFilePath,
                            mscclAlgoHandle_t* mscclAlgoHandle, int rank);

/*! @brief MSCCL Run Algorithm
 *
 * @details Run MSCCL algorithm specified by mscclAlgoHandle. The parameter
 * list merges all possible parameters required by different operations as this
 * is a general-purposed API. This API is expected to be called by MSCCL
 * scheduler instead of end users.
 */
ncclResult_t mscclRunAlgo(void const* sendBuff, const size_t sendCounts[],
                          const size_t sDisPls[], void* recvBuff,
                          const size_t recvCounts[], const size_t rDisPls[],
                          size_t count, ncclDataType_t dataType, int root,
                          int peer, ncclRedOp_t op,
                          mscclAlgoHandle_t mscclAlgoHandle, ncclComm_t comm,
                          cudaStream_t stream);
ncclResult_t pmscclRunAlgo(void const* sendBuff, const size_t sendCounts[],
                           const size_t sDisPls[], void* recvBuff,
                           const size_t recvCounts[], const size_t rDisPls[],
                           size_t count, ncclDataType_t dataType, int root,
                           int peer, ncclRedOp_t op,
                           mscclAlgoHandle_t mscclAlgoHandle, ncclComm_t comm,
                           cudaStream_t stream);

/*! @brief MSCCL Load Algorithm
 *
 * @details Unload MSCCL algorithm previous loaded using its handle. This API
 * is expected to be called by MSCCL scheduler instead of end users.
 */
ncclResult_t mscclUnloadAlgo(mscclAlgoHandle_t mscclAlgoHandle);
ncclResult_t pmscclUnloadAlgo(mscclAlgoHandle_t mscclAlgoHandle);

/*
 * Group semantics
 *
 * When managing multiple GPUs from a single thread, and since NCCL collective
 * calls may perform inter-CPU synchronization, we need to "group" calls for
 * different ranks/devices into a single call.
 *
 * Grouping NCCL calls as being part of the same collective operation is done
 * using ncclGroupStart and ncclGroupEnd. ncclGroupStart will enqueue all
 * collective calls until the ncclGroupEnd call, which will wait for all calls
 * to be complete. Note that for collective communication, ncclGroupEnd only
 * guarantees that the operations are enqueued on the streams, not that
 * the operation is effectively done.
 *
 * Both collective communication and ncclCommInitRank can be used in conjunction
 * of ncclGroupStart/ncclGroupEnd, but not together.
 *
 * Group semantics also allow to fuse multiple operations on the same device
 * to improve performance (for aggregated collective calls), or to permit
 * concurrent progress of multiple send/receive operations.
 */

/*
 * Group Start
 *
 * Start a group call. All calls to NCCL until ncclGroupEnd will be fused into
 * a single NCCL operation. Nothing will be started on the CUDA stream until
 * ncclGroupEnd.
 */
ncclResult_t ncclGroupStart();
ncclResult_t pncclGroupStart();

/*
 * Group End
 *
 * End a group call. Start a fused NCCL operation consisting of all calls since
 * ncclGroupStart. Operations on the CUDA stream depending on the NCCL
 * operations need to be called after ncclGroupEnd.
 */
ncclResult_t ncclGroupEnd();
ncclResult_t pncclGroupEnd();

/*
 * Group Simulate End
 *
 * Simulate a ncclGroupEnd() call and return NCCL's simulation info in a struct.
 */
ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* simInfo);
ncclResult_t pncclGroupSimulateEnd(ncclSimInfo_t* simInfo);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // end include guard
