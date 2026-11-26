#pragma once
#include "prepared_api.h"
#include <string>
#include <vector>
namespace uccl_lite::integration {
struct AutoRequest {
  ucclLiteOperation operation;
  size_t count = 0;
  bool uniform = true;
  std::vector<ucclLiteRankLayout> layouts;
};
struct AutoManifest {
  ncclDataType_t dtype = ncclFloat32;
  size_t capacity = 0;
  unsigned flags = 0;
  std::vector<AutoRequest> requests;
  std::string contents;
};
AutoManifest loadAutoManifest(std::string const& file);
// Unsupported shapes for the actual rank extent are omitted, not extrapolated.
std::vector<AutoRequest> materializeAutoManifest(AutoManifest const& manifest,
                                                 int ranks);
}  // namespace uccl_lite::integration
