#pragma once
#include "channel.hpp"
#include "cpu_executor.hpp"
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uccl_lite::cpu_router {
// Operators consume CPU-visible spans and return the number of output bytes.
// Reduction retains the executor's FP32 accumulation / BF16 rounding policy.
struct ReduceBf16 {
  size_t operator()(MemorySpan const* sources, size_t count,
                    MemorySpan output) const {
    CpuExecutor::sumBf16(sources, count, output);
    return output.bytes;
  }
};
struct PackBytes {
  size_t operator()(MemorySpan const* sources, size_t count,
                    MemorySpan output) const;
};
// Stable, bytewise deduplication of fixed-width records. Intended for small
// routing metadata; applications can supply a hash-based operator for bulk
// data.
struct DeduplicateRecords {
  size_t recordBytes;
  size_t operator()(MemorySpan const* sources, size_t count,
                    MemorySpan output) const;
};
struct RouterDestination {
  Channel* channel;
  MemorySpan output;
  cudaStream_t stream;
};

// Three routing operations, independent of any collective and scheduler.
// Transfer tickets retain registered spans. CPU processing is synchronous on
// the calling lane; the executor overloads queue work on an existing lane.
// Input readiness and packet reuse credits are established by the caller.
class RouterCore {
 public:
  static ChannelCompletion forward(Channel& channel, MemorySpan output,
                                   MemorySpan input, cudaStream_t stream) {
    return channel.put(output, input, stream);
  }
  // A persistent pipeline already owns completion and storage lifetimes.
  static void forwardAsync(Channel& channel, MemorySpan output,
                           MemorySpan input, cudaStream_t stream) {
    channel.putAsync(output, input, stream);
  }
  static void forward(MemorySpan output, MemorySpan input) {
    if (output.kind == MemoryKind::Gpu || input.kind == MemoryKind::Gpu ||
        output.bytes < input.bytes || !output.address || !input.address)
      throw std::invalid_argument("invalid host forward spans");
    // Identical views are a no-op; other overlapping views are supported.
    if (output.address != input.address)
      std::memmove(output.address, input.address, input.bytes);
  }
  template <class Operator = ReduceBf16>
  static size_t process(MemorySpan const* sources, size_t count,
                        MemorySpan output, Operator op = {}) {
    if (count && !sources) throw std::invalid_argument("null process sources");
    return op(sources, count, std::move(output));
  }
  static CpuCompletion process(CpuExecutor& cpu, size_t lane,
                               MemorySpan const* sources, size_t count,
                               MemorySpan output, ReduceBf16 = {}) {
    return cpu.enqueueSum(lane, sources, count, std::move(output));
  }
  // Callback context (including any spans it uses) must outlive completion.
  static CpuCompletion process(CpuExecutor& cpu, size_t lane,
                               CpuExecutor::Callback op, void* context) {
    return cpu.enqueue(lane, op, context);
  }
  static std::vector<ChannelCompletion> multicast(
      MemorySpan input, std::vector<RouterDestination> const& destinations) {
    std::vector<ChannelCompletion> tickets;
    tickets.reserve(destinations.size());
    for (auto const& d : destinations) {
      if (!d.channel) throw std::invalid_argument("null multicast channel");
      tickets.push_back(forward(*d.channel, d.output, input, d.stream));
    }
    return tickets;
  }
  // Allocation-free fan-out for persistent transports. Send may transfer a
  // span or publish a shared immutable packet to readers. The caller owns the
  // completion protocol and cannot reuse input until every reader acknowledges.
  template <class Packet, class Destinations, class Send>
  static void multicast(Packet const& packet, Destinations const& destinations,
                        Send&& send) {
    for (auto const& destination : destinations) send(destination, packet);
  }
};
}  // namespace uccl_lite::cpu_router
