#ifndef UCCL_LITE_PREPARED_API_H
#define UCCL_LITE_PREPARED_API_H
#include <nccl.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Explicit preparation extension; this library exports no nccl* replacements.
// One attached session/process, one process/GPU, 1..32 ranks. All host calls
// must be serialized. Borrowed NCCL comm must be ready, blocking and kept alive
// until detach. Close NCCL groups and finish external capture before any call.
typedef uint64_t ucclLiteSession;
typedef uint64_t ucclLitePlan;
typedef struct {
  unsigned char bytes[128];
} ucclLiteUniqueId;
typedef enum {
  ucclLiteAllGather = 1,
  ucclLiteReduceScatter = 2,
  ucclLiteAllReduce = 3
} ucclLiteOperation;
enum {
  UCCL_LITE_ALLOW_PEER = 1u << 0,
  UCCL_LITE_ALLOW_HOST = 1u << 1,
  UCCL_LITE_ALLOW_PEER_PACKET = 1u << 2,
  UCCL_LITE_ALLOW_NETWORK = 1u << 3,
  UCCL_LITE_PROBE_RESOURCES = 1u << 4,
  UCCL_LITE_FALLBACK_ONLY = 1u << 5
};
typedef struct {
  size_t structSize;
  unsigned version;  // 1; initialize with ucclLiteOptionsInit.
  unsigned flags;
  size_t capacityBytes;
  ncclDataType_t dtype;  // BF16/FP32; FP16 currently uses native only.
  char const* nativeLibrary;
  char const* configurationIdentity;  // common native init/config identity
  char const* hardwareProfile;        // optional
} ucclLiteOptions;
typedef struct {
  unsigned inPlace;    // 0 or 1, legal NCCL alias for this logical rank
  unsigned alignment;  // exact minimum pointer alignment, capped at 256
} ucclLiteRankLayout;
typedef struct {
  ucclLiteOperation operation;
  size_t count;  // NCCL element-count semantics; reductions are SUM
  ucclLiteRankLayout const* layouts;  // all ranks, logical rank order
  size_t ranks;
  unsigned fallbackOnly;  // 0 or 1; collective per-plan policy
} ucclLiteRequest;

ncclResult_t ucclLiteOptionsInit(ucclLiteOptions* options);
// Generate once and distribute to all ranks through the application's control
// plane. This is separate from the native ncclUniqueId.
ncclResult_t ucclLiteGetUniqueId(ucclLiteUniqueId* id);
// Local, no transport initialization. Rank/device/count come from native comm.
// Strings are copied; on failure the output session is zero.
ncclResult_t ucclLiteAttach(ncclComm_t comm, ucclLiteUniqueId const* id,
                            ucclLiteOptions const* options,
                            ucclLiteSession* session);
// Collective, idle, outside capture/timing. Null output pointers/invalid local
// session handles are caller errors and cannot participate in consensus.
// Failed collective preparation invalidates execution until a successful one.
ncclResult_t ucclLitePrepare(ucclLiteSession session,
                             ucclLiteRequest const* request,
                             ucclLitePlan* plan);
ncclResult_t ucclLiteEnqueue(ucclLiteSession session, ucclLitePlan plan,
                             void const* input, void* output,
                             cudaStream_t stream);
// Graph bindings remain valid until collective release/detach. Eager and graph
// handles have different types internally and cannot be interchanged.
ncclResult_t ucclLitePrepareGraph(ucclLiteSession session,
                                  ucclLiteRequest const* request,
                                  void const* input, void* output,
                                  ucclLitePlan* graph);
ncclResult_t ucclLiteLaunchGraph(ucclLiteSession session, ucclLitePlan graph,
                                 cudaStream_t stream);
ncclResult_t ucclLiteReleaseGraph(ucclLiteSession session, ucclLitePlan graph);
ncclResult_t ucclLiteSynchronize(ucclLiteSession session);
// JSON evidence including placement, decision, identity and all paired samples.
// Query with buffer=NULL; bytes includes NUL. Short buffers update bytes and
// return ncclInvalidArgument without writing. Released graphs retain their
// immutable selection report until detach.
ncclResult_t ucclLitePlanReport(ucclLiteSession session, ucclLitePlan plan,
                                char* buffer, size_t* bytes);
// Collective lifetime boundary; retire submitted work, keep native comm alive.
// Synchronize before direct native calls. Complete direct native work before
// re-entering this API. This extension does not track arbitrary native calls.
ncclResult_t ucclLiteDetach(ucclLiteSession session);
// Thread-local diagnostic, valid until the next API call on this thread.
char const* ucclLiteLastError(void);

#ifdef __cplusplus
}
#endif
#endif
