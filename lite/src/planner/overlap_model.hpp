#pragma once
#include "uccl_lite/planner.hpp"
#include <map>

namespace uccl_lite {
PlanEstimate estimateTaskGraph(
    std::map<std::string, Performance> const& resources,
    ExecutionPlan const& plan, bool allowAdvertised);
}
