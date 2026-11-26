#include "prepared_api.h"
// Compile as C, not CUDA/C++, to keep the public extension usable from C/FFI.
ncclResult_t uccl_lite_c_options(ucclLiteOptions* options) {
  return ucclLiteOptionsInit(options);
}
