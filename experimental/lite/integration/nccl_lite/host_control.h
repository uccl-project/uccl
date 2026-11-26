#pragma once
// Compatibility include for recorded integration/control harnesses.
#include "../../include/uccl_lite/bootstrap/control.hpp"
namespace uccl_lite::nccl_detail {
using bootstrap::HostControl;
using bootstrap::hostControlInitialPort;
}  // namespace uccl_lite::nccl_detail
