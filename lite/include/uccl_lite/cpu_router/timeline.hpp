#pragma once
#include <nvtx3/nvToolsExt.h>
#include <cstdint>
#include <cstdlib>
namespace uccl_lite::cpu_router {
inline bool timelineEnabled() {
  static bool enabled = std::getenv("UCCL_LITE_TIMELINE") != nullptr;
  return enabled;
}
class TimelineRange {
 public:
  TimelineRange(char const* name, uint64_t generation = 0, bool lane = true)
      : active_(timelineEnabled() && lane) {
    if (active_) {
      nvtxEventAttributes_t a{};
      a.version = NVTX_VERSION;
      a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      a.messageType = NVTX_MESSAGE_TYPE_ASCII;
      a.message.ascii = name;
      a.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
      a.payload.ullValue = generation;
      nvtxRangePushEx(&a);
    }
  }
  ~TimelineRange() {
    if (active_) nvtxRangePop();
  }

 private:
  bool active_;
};
}  // namespace uccl_lite::cpu_router
