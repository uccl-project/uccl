#include "prepared_abi.h"
ncclResult_t uccl_lite_c_abi_prepare(ncclComm_t comm,
                                     ucclLiteUniqueId const* id,
                                     ucclLiteOptions const* options,
                                     ucclLiteRequest const* requests,
                                     size_t count) {
  return ucclLiteNcclPrepare(comm, id, options, requests, count);
}
