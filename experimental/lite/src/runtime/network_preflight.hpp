#pragma once
#include "uccl_lite/network_calibration.hpp"
#include "uccl_lite/runtime/collective_backend.hpp"
#include <array>

namespace uccl_lite::runtime {
struct NetworkPreflight {
  std::array<int, 8> rankOrder{};
  int rail = -1;
  std::shared_ptr<NetworkProfile const> calibration;
};
// Collective planning after registered layout/configuration validation. When
// requested, missing profiles are measured on private temporary resources;
// evidence is retained even if no rail can subsequently win.
NetworkPreflight preflightNetwork(
    BackendContext const& context, mscclpp::Bootstrap& control,
    std::shared_ptr<NetworkProfile const>& calibration, bool probeResources);
std::unique_ptr<CollectiveBackend> preparedNetworkBackend(
    BackendContext const& context,
    std::shared_ptr<NetworkProfile const> calibration);
}  // namespace uccl_lite::runtime
