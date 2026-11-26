#include "prepared_api.h"
#include "native_fallback.hpp"
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <mutex>
#include <sstream>

namespace {
using namespace uccl_lite;
using namespace uccl_lite::runtime;
thread_local char diagnostic[1024]{};
std::mutex mutex;
uint64_t nextSession = 1;
struct Session {
  std::unique_ptr<PreparedRuntime> runtime;
  ncclDataType_t dtype;
};
std::map<uint64_t, Session> sessions;
template <class F>
ncclResult_t invoke(F f) noexcept {
  diagnostic[0] = 0;
  try {
    std::lock_guard<std::mutex> lock(mutex);
    f();
    return ncclSuccess;
  } catch (std::invalid_argument const& e) {
    std::snprintf(diagnostic, sizeof(diagnostic), "%s", e.what());
    return ncclInvalidArgument;
  } catch (std::out_of_range const& e) {
    std::snprintf(diagnostic, sizeof(diagnostic), "%s", e.what());
    return ncclInvalidArgument;
  } catch (std::logic_error const& e) {
    std::snprintf(diagnostic, sizeof(diagnostic), "%s", e.what());
    return ncclInvalidUsage;
  } catch (std::exception const& e) {
    std::snprintf(diagnostic, sizeof(diagnostic), "%s", e.what());
    return ncclSystemError;
  } catch (...) {
    std::snprintf(diagnostic, sizeof(diagnostic),
                  "unknown UCCL preparation error");
    return ncclInternalError;
  }
}
void require(bool valid, char const* message) {
  if (!valid) throw std::invalid_argument(message);
}
PreparedRuntime& preparedRuntime(ucclLiteSession session) {
  auto found = sessions.find(session);
  require(found != sessions.end(), "absent or detached UCCL session");
  return *found->second.runtime;
}
// Odd eager/even graph, nonzero, scoped to a session. Core indices are never
// reused, including after graph release. No heap allocation during enqueue.
ucclLitePlan encode(size_t index, bool graph) {
  require(index < (UINT64_MAX - 2) / 2, "plan handle space exhausted");
  return uint64_t(index) * 2 + (graph ? 2 : 1);
}
size_t decode(ucclLitePlan plan, bool graph) {
  require(plan && (plan % 2 == (graph ? 0u : 1u)),
          "invalid plan handle or execution mode");
  return size_t((plan - (graph ? 2 : 1)) / 2);
}
CollectiveRequest request(ucclLiteRequest const* r) {
  CollectiveRequest result;
  // Invalid request contents reach the core's collective envelope so one bad
  // rank does not leave its peers entering a different preparation protocol.
  if (!r || !r->layouts || !r->ranks || r->ranks > 32 || r->fallbackOnly > 1)
    return result;
  result.operation = static_cast<NetworkOperation>(r->operation);
  result.count = r->count;
  result.fallbackOnly = r->fallbackOnly;
  for (size_t i = 0; i < r->ranks; ++i) {
    auto l = r->layouts[i];
    result.layouts.push_back({l.inPlace != 0, l.inPlace > 1 ? 0 : l.alignment});
  }
  return result;
}
void quoted(std::ostream& out, std::string const& value) {
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << char(c);
    else if (c < 32 || c >= 127) {
      char escape[7];
      std::snprintf(escape, sizeof(escape), "\\u%04x", unsigned(c));
      out << escape;
    } else
      out << char(c);
  }
  out << '"';
}
template <class T>
void array(std::ostream& out, std::vector<T> const& values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}
std::string report(PreparedRuntime const& r, PreparedSelection const& s,
                   ncclDataType_t dtype) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17)
      << "{\"version\":1,\"domain\":" << int(r.placement().domain)
      << ",\"backend\":" << int(s.backend)
      << ",\"graph\":" << (s.graph ? "true" : "false")
      << ",\"from_cache\":" << (s.fromCache ? "true" : "false")
      << ",\"accelerated_resources\":"
      << (r.acceleratedResourcesInitialized() ? "true" : "false")
      << ",\"operation\":" << int(s.request.operation)
      << ",\"count\":" << s.request.count << ",\"dtype_nccl\":" << int(dtype)
      << ",\"fallback_only\":" << (s.request.fallbackOnly ? "true" : "false")
      << ",\"layouts\":[";
  for (size_t i = 0; i < s.request.layouts.size(); ++i) {
    if (i) out << ',';
    auto l = s.request.layouts[i];
    out << "{\"in_place\":" << (l.inPlace ? "true" : "false")
        << ",\"alignment\":" << l.alignment << '}';
  }
  out << ']';
  for (auto pair : {std::make_pair("reason", &s.reason),
                    {"cache_status", &s.cacheStatus},
                    {"topology_identity", &s.topologyIdentity},
                    {"executor_identity", &s.executorIdentity},
                    {"baseline_identity", &s.baselineIdentity}}) {
    out << ",\"" << pair.first << "\":";
    quoted(out, *pair.second);
  }
  out << ",\"participants\":[";
  for (size_t i = 0; i < r.placement().ranks.size(); ++i) {
    if (i) out << ',';
    auto const& p = r.placement().ranks[i];
    out << "{\"host\":";
    quoted(out, p.host);
    out << ",\"gpu\":";
    quoted(out, p.gpu);
    out << ",\"device\":" << p.device << '}';
  }
  out << "],\"pool_candidate_samples_us\":";
  array(out, s.poolCandidateSamples);
  out << ",\"pool_baseline_samples_us\":";
  array(out, s.poolBaselineSamples);
  out << ",\"candidates\":[";
  for (size_t i = 0; i < s.candidates.size(); ++i) {
    if (i) out << ',';
    auto const& c = s.candidates[i];
    out << "{\"from_cache\":" << (c.fromCache ? "true" : "false")
        << ",\"cache_status\":";
    quoted(out, c.cacheStatus);
    out << ",\"backend\":" << int(c.backend)
        << ",\"eligible\":" << (c.eligible ? "true" : "false")
        << ",\"correct\":" << (c.correct ? "true" : "false")
        << ",\"resource_estimate_us\":" << c.resourceEstimateUs
        << ",\"resource_serial_us\":" << c.resourceSerialUs
        << ",\"resource_dependency_us\":" << c.resourceDependencyUs
        << ",\"resource_load_us\":" << c.resourceLoadUs << ",\"samples_us\":";
    array(out, c.samples);
    out << ",\"baseline_samples_us\":";
    array(out, c.baselineSamples);
    out << ",\"validation_samples_us\":";
    array(out, c.validationSamples);
    out << ",\"validation_baseline_samples_us\":";
    array(out, c.validationBaselineSamples);
    out << ",\"execution_rank_order\":";
    array(out, c.executionRankOrder);
    out << ",\"rails\":";
    array(out, c.rails);
    out << ",\"peer_tree_parents\":";
    array(out, c.peerTreeParents);
    out << ",\"resource_model_identity\":";
    quoted(out, c.resourceModelIdentity);
    out << ",\"resource_model_reason\":";
    quoted(out, c.resourceModelReason);
    out << ",\"resource_model_observations\":" << c.resourceModelObservations;
    out << ",\"unmodeled_resources\":[";
    for (size_t j = 0; j < c.unmodeledResources.size(); ++j) {
      if (j) out << ',';
      quoted(out, c.unmodeledResources[j]);
    }
    out << ']';
    out << ",\"traffic\":{";
    bool firstTraffic = true;
    for (auto const& item : c.traffic) {
      if (!firstTraffic) out << ',';
      firstTraffic = false;
      quoted(out, item.first);
      out << ':' << item.second;
    }
    out << '}';
    out << ",\"configuration_identities\":[";
    for (size_t j = 0; j < c.configurationIdentities.size(); ++j) {
      if (j) out << ',';
      quoted(out, c.configurationIdentities[j]);
    }
    out << ']';
    out << ",\"network_calibration\":";
    if (c.networkCalibration) {
      auto const& p = *c.networkCalibration;
      out << "{\"probe_identity\":";
      quoted(out, p.probeIdentity);
      out << ",\"path_scope\":";
      quoted(out, p.pathScope);
      out << ",\"group_scope\":";
      quoted(out, p.groupScope);
      out << ",\"paths\":" << p.paths.size()
          << ",\"cohorts\":" << p.groups.size() << '}';
    } else
      out << "null";
    out << '}';
  }
  out << "]}";
  return out.str();
}
}  // namespace

extern "C" {
ncclResult_t ucclLiteOptionsInit(ucclLiteOptions* options) {
  return invoke([&] {
    require(options, "null options output");
    *options = {};
    options->structSize = sizeof(*options);
    options->version = 1;
    options->flags = UCCL_LITE_ALLOW_PEER | UCCL_LITE_ALLOW_HOST |
                     UCCL_LITE_ALLOW_PEER_PACKET | UCCL_LITE_ALLOW_NETWORK |
                     UCCL_LITE_PROBE_RESOURCES;
    options->dtype = ncclBfloat16;
  });
}
ncclResult_t ucclLiteGetUniqueId(ucclLiteUniqueId* id) {
  return invoke([&] {
    require(id, "null session ID output");
    auto generated = PreparedRuntime::createUniqueId();
    static_assert(sizeof(*id) == sizeof(generated));
    std::memcpy(id, generated.data(), sizeof(*id));
  });
}
ncclResult_t ucclLiteAttach(ncclComm_t comm, ucclLiteUniqueId const* id,
                            ucclLiteOptions const* o,
                            ucclLiteSession* session) {
  if (session) *session = 0;
  return invoke([&] {
    require(session && id && o, "null attach argument");
    require(o->structSize == sizeof(*o) && o->version == 1,
            "unsupported preparation options ABI");
    require(!(o->flags & ~63u) && o->nativeLibrary && o->configurationIdentity,
            "invalid flags or missing native configuration");
    require(sessions.empty(),
            "only one attached session per process supported");
    require(nextSession != UINT64_MAX, "session handle space exhausted");
    ScalarType dtype;
    switch (o->dtype) {
      case ncclFloat32:
        dtype = ScalarType::Float32;
        break;
      case ncclBfloat16:
        dtype = ScalarType::BFloat16;
        break;
      case ncclFloat16:
        dtype = ScalarType::Float16;
        break;
      default:
        throw std::invalid_argument("unsupported preparation dtype");
    }
    NetworkId bootstrap;
    std::memcpy(bootstrap.data(), id, sizeof(*id));
    auto context = integration::nativeNcclContext(comm, o->nativeLibrary,
                                                  bootstrap, o->capacityBytes);
    PreparationOptions options;
    options.dtype = dtype;
    options.fallback = integration::makeNativeNcclFallback(
        context, comm, o->nativeLibrary, o->configurationIdentity, dtype);
    options.fallbackOnly = o->flags & UCCL_LITE_FALLBACK_ONLY;
    options.allowPeer = o->flags & UCCL_LITE_ALLOW_PEER;
    options.allowHost = o->flags & UCCL_LITE_ALLOW_HOST;
    options.allowPeerPacket = o->flags & UCCL_LITE_ALLOW_PEER_PACKET;
    options.allowNetwork = o->flags & UCCL_LITE_ALLOW_NETWORK;
    options.probeResources = o->flags & UCCL_LITE_PROBE_RESOURCES;
    if (o->hardwareProfile) options.hardwareProfile = o->hardwareProfile;
    auto prepared = std::make_unique<PreparedRuntime>(context, options);
    sessions.emplace(nextSession, Session{std::move(prepared), o->dtype});
    *session = nextSession++;
  });
}
ncclResult_t ucclLitePrepare(ucclLiteSession session, ucclLiteRequest const* r,
                             ucclLitePlan* plan) {
  if (plan) *plan = 0;
  return invoke([&] {
    require(plan, "null plan output");
    *plan = encode(preparedRuntime(session).prepare(request(r)), false);
  });
}
ncclResult_t ucclLiteEnqueue(ucclLiteSession session, ucclLitePlan plan,
                             void const* input, void* output,
                             cudaStream_t stream) {
  return invoke([&] {
    preparedRuntime(session).enqueue(decode(plan, false), input, output,
                                     stream);
  });
}
ncclResult_t ucclLitePrepareGraph(ucclLiteSession session,
                                  ucclLiteRequest const* r, void const* input,
                                  void* output, ucclLitePlan* graph) {
  if (graph) *graph = 0;
  return invoke([&] {
    require(graph, "null graph output");
    *graph = encode(
        preparedRuntime(session).prepareGraph(request(r), input, output), true);
  });
}
ncclResult_t ucclLiteLaunchGraph(ucclLiteSession session, ucclLitePlan graph,
                                 cudaStream_t stream) {
  return invoke([&] {
    preparedRuntime(session).launchGraph(decode(graph, true), stream);
  });
}
ncclResult_t ucclLiteReleaseGraph(ucclLiteSession session, ucclLitePlan graph) {
  return invoke([&] {
    // Release is collective: even an invalid token on one rank must enter
    // the core envelope, rather than leave peers waiting for that rank.
    auto index = graph && graph % 2 == 0 ? decode(graph, true)
                                         : std::numeric_limits<size_t>::max();
    preparedRuntime(session).releaseGraph(index);
  });
}
ncclResult_t ucclLiteSynchronize(ucclLiteSession session) {
  return invoke([&] { preparedRuntime(session).synchronize(); });
}
ncclResult_t ucclLitePlanReport(ucclLiteSession session, ucclLitePlan plan,
                                char* buffer, size_t* bytes) {
  return invoke([&] {
    require(bytes, "null report size output");
    auto& r = preparedRuntime(session);
    bool graph = plan && plan % 2 == 0;
    auto index = decode(plan, graph);
    auto text = report(r, graph ? r.graphSelection(index) : r.selection(index),
                       sessions.at(session).dtype);
    auto available = buffer ? *bytes : 0;
    *bytes = text.size() + 1;
    if (buffer) {
      require(available >= *bytes, "report buffer too small");
      std::memcpy(buffer, text.c_str(), *bytes);
    }
  });
}
ncclResult_t ucclLiteDetach(ucclLiteSession session) {
  return invoke([&] {
    preparedRuntime(session).synchronize();
    sessions.erase(session);
  });
}
char const* ucclLiteLastError() { return diagnostic; }
}
