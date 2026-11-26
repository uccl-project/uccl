#ifndef UCCL_LITE_PREPARED_ABI_H
#define UCCL_LITE_PREPARED_ABI_H
#include "prepared_api.h"
#ifdef __cplusplus
extern "C" {
#endif
// Native communicator, one process/GPU and one active prepared communicator
// per process. Collective idle preparation, closed groups, outside capture.
// ID comes from ucclLiteGetUniqueId and is shared by the application.
// Subsequent ordinary nccl* collectives may use qualified UCCL executors.
// Unsupported collective requests, groups, capture and P2P transfer ownership
// permanently to native NCCL. Do not mix this API with ucclLiteAttach on the
// same comm. Opt-in UCCL_LITE_AUTO_INIT performs preparation at native
// communicator initialization from UCCL_LITE_AUTO_MANIFEST; see
// docs/automatic-nccl-init.md for the collective deployment contract.
ncclResult_t ucclLiteNcclPrepare(ncclComm_t comm, ucclLiteUniqueId const* id,
                                 ucclLiteOptions const* options,
                                 ucclLiteRequest const* requests, size_t count);
// Collective, outside capture/groups. Retires UCCL resources but keeps native
// communicator and completion-event leases alive for native graph replay.
// State is removed by ncclCommDestroy/ncclCommAbort, not by this call.
// Revoke/shrink require this release first. Aborting active UCCL execution
// retains its resources until process exit; it does not cancel UCCL kernels.
ncclResult_t ucclLiteNcclRelease(ncclComm_t comm);
ncclResult_t ucclLiteNcclReport(ncclComm_t comm, char* buffer, size_t* bytes);
#ifdef __cplusplus
}
#endif
#endif
