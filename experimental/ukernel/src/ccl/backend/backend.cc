#include "backend.h"

namespace UKernel {
namespace CCL {

// BatchBackend is now a pure abstract interface.
// Subclasses implement do_enqueue / do_drain / capacity / supports.
// Threading and queue management is handled by SprayExecutor.

}  // namespace CCL
}  // namespace UKernel
