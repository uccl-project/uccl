#pragma once
// Minimal NCCL type definitions for compile-time use.
// This allows building without nccl.h installed. The actual NCCL library
// (libnccl.so or librccl.so) is loaded at runtime via dlopen/dlsym.
//
// These definitions must match the NCCL ABI. They are stable across
// NCCL 2.x versions.

#include <cuda_runtime_api.h>

#define NCCL_UNIQUE_ID_BYTES 128

typedef struct ncclComm* ncclComm_t;

typedef struct {
  char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6,
  ncclInProgress = 7,
  ncclNumResults = 8,
} ncclResult_t;

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
  ncclNumTypes = 10,
} ncclDataType_t;

// Function declarations (resolved at runtime via dlsym in nccl_dl.cc)
#ifdef __cplusplus
extern "C" {
#endif

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                              ncclUniqueId commId, int rank);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclSend(const void* sendbuff, size_t count,
                      ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count,
                      ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream);
const char* ncclGetErrorString(ncclResult_t result);

#ifdef __cplusplus
}
#endif
