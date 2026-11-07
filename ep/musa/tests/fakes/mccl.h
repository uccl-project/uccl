#pragma once

#include "mpi.h"
#include "musa_runtime.h"
using mcclComm_t = int;
constexpr int mcclSuccess = 0, mcclFloat = 1;
namespace fake {
struct Message {
  void* data;
  size_t count;
  int peer;
  bool send;
};
inline thread_local std::vector<Message> messages;
inline std::atomic<int> transfers{0};
}  // namespace fake
inline int mcclGroupStart() {
  fake::messages.clear();
  return mcclSuccess;
}
inline int mcclSend(void const* data, size_t count, int, int peer, mcclComm_t,
                    musaStream_t) {
  assert(count && peer != fake::rank);
  fake::messages.push_back({const_cast<void*>(data), count, peer, true});
  return mcclSuccess;
}
inline int mcclRecv(void* data, size_t count, int, int peer, mcclComm_t,
                    musaStream_t) {
  assert(count && peer != fake::rank);
  fake::messages.push_back({data, count, peer, false});
  return mcclSuccess;
}
inline int mcclGroupEnd() {
  fake::collective(5, &fake::messages, [](auto const& all) {
    for (int source = 0; source < fake::ranks; ++source)
      for (auto const& send :
           *static_cast<std::vector<fake::Message>*>(all[source]))
        if (send.send) {
          auto const& target =
              *static_cast<std::vector<fake::Message>*>(all[send.peer]);
          auto recv = std::find_if(
              target.begin(), target.end(),
              [&](auto const& m) { return !m.send && m.peer == source; });
          assert(recv != target.end() && recv->count == send.count);
          std::memcpy(recv->data, send.data, send.count * sizeof(float));
          ++fake::transfers;
        }
  });
  return mcclSuccess;
}
