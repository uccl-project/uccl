// Linker-level dlsym wrappers for libnccl functions.
// Provides nccl* symbols via dlopen/dlsym instead of linking -lnccl.

#include "include/nccl_wrapper.h"

static void* nccl_resolve(char const* name) {
  return uccl::nccl_dl::resolve(name);
}

extern "C" {

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
  using FnType = ncclResult_t (*)(ncclUniqueId*);
  static FnType fn = reinterpret_cast<FnType>(nccl_resolve("ncclGetUniqueId"));
  return fn(uniqueId);
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId,
                              int rank) {
  using FnType = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int);
  static FnType fn = reinterpret_cast<FnType>(nccl_resolve("ncclCommInitRank"));
  return fn(comm, nranks, commId, rank);
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  using FnType = ncclResult_t (*)(ncclComm_t);
  static FnType fn = reinterpret_cast<FnType>(nccl_resolve("ncclCommDestroy"));
  return fn(comm);
}

ncclResult_t ncclSend(void const* sendbuff, size_t count,
                      ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
  using FnType = ncclResult_t (*)(void const*, size_t, ncclDataType_t, int,
                                  ncclComm_t, cudaStream_t);
  static FnType fn = reinterpret_cast<FnType>(nccl_resolve("ncclSend"));
  return fn(sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  using FnType = ncclResult_t (*)(void*, size_t, ncclDataType_t, int,
                                  ncclComm_t, cudaStream_t);
  static FnType fn = reinterpret_cast<FnType>(nccl_resolve("ncclRecv"));
  return fn(recvbuff, count, datatype, peer, comm, stream);
}

char const* ncclGetErrorString(ncclResult_t result) {
  using FnType = char const* (*)(ncclResult_t);
  static FnType fn =
      reinterpret_cast<FnType>(nccl_resolve("ncclGetErrorString"));
  return fn(result);
}

}  // extern "C"
