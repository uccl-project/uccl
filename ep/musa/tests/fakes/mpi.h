#pragma once

// In-process collective doubles for production Session tests, not an MPI ABI.
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <vector>

using MPI_Comm = int;
constexpr MPI_Comm MPI_COMM_WORLD = 0;
constexpr int MPI_SUCCESS = 0, MPI_INT = 1, MPI_UINT64_T = 2, MPI_INT64_T = 3;
constexpr int MPI_MIN = 1;

namespace fake {
inline int ranks = 1;
inline thread_local int rank = 0;
inline std::mutex mutex;
inline std::condition_variable cv;
inline unsigned arrived = 0, generation = 0;
inline int operation;
inline std::vector<void*> arguments;

inline size_t type_size(int type) {
  return type == MPI_INT ? sizeof(int) : sizeof(uint64_t);
}

inline void collective(int kind, void* args,
                       std::function<void(std::vector<void*> const&)> action) {
  std::unique_lock<std::mutex> lock(mutex);
  unsigned const old = generation;
  if (arrived == 0) {
    operation = kind;
    arguments.resize(ranks);
  }
  assert(operation == kind);
  arguments[rank] = args;
  if (++arrived == unsigned(ranks)) {
    action(arguments);
    arrived = 0;
    ++generation;
    cv.notify_all();
  } else {
    cv.wait(lock, [&] { return generation != old; });
  }
}
}  // namespace fake

inline int MPI_Comm_rank(MPI_Comm, int* rank) {
  *rank = fake::rank;
  return MPI_SUCCESS;
}
inline int MPI_Comm_size(MPI_Comm, int* ranks) {
  *ranks = fake::ranks;
  return MPI_SUCCESS;
}
[[noreturn]] inline int MPI_Abort(MPI_Comm, int) { std::abort(); }

inline int MPI_Allreduce(void const* send, void* recv, int count, int type,
                         int op, MPI_Comm) {
  assert(type == MPI_INT && op == MPI_MIN);
  struct Args {
    int const* send;
    int* recv;
  };
  Args args{static_cast<int const*>(send), static_cast<int*>(recv)};
  fake::collective(1, &args, [=](auto const& all) {
    for (int i = 0; i < count; ++i) {
      int result = 1;
      for (auto p : all)
        result = std::min(result, static_cast<Args*>(p)->send[i]);
      for (auto p : all) static_cast<Args*>(p)->recv[i] = result;
    }
  });
  return MPI_SUCCESS;
}
inline int MPI_Allgather(void const* send, int count, int type, void* recv,
                         int recv_count, int recv_type, MPI_Comm) {
  assert(count == recv_count && type == recv_type);
  struct Args {
    void const* send;
    void* recv;
  };
  Args args{send, recv};
  fake::collective(2, &args, [=](auto const& all) {
    for (auto p : all)
      for (int source = 0; source < fake::ranks; ++source)
        std::memcpy(static_cast<char*>(static_cast<Args*>(p)->recv) +
                        source * count * fake::type_size(type),
                    static_cast<Args*>(all[source])->send,
                    count * fake::type_size(type));
  });
  return MPI_SUCCESS;
}
inline int MPI_Alltoall(void const* send, int count, int type, void* recv,
                        int recv_count, int recv_type, MPI_Comm) {
  assert(count == recv_count && type == recv_type);
  struct Args {
    void const* send;
    void* recv;
  };
  Args args{send, recv};
  fake::collective(3, &args, [=](auto const& all) {
    for (int dst = 0; dst < fake::ranks; ++dst)
      for (int src = 0; src < fake::ranks; ++src)
        std::memcpy(
            static_cast<char*>(static_cast<Args*>(all[dst])->recv) +
                src * count * fake::type_size(type),
            static_cast<char const*>(static_cast<Args*>(all[src])->send) +
                dst * count * fake::type_size(type),
            count * fake::type_size(type));
  });
  return MPI_SUCCESS;
}
inline int MPI_Alltoallv(void const* send, int const* sc, int const* so,
                         int type, void* recv, int const* rc, int const* ro,
                         int recv_type, MPI_Comm) {
  assert(type == recv_type);
  struct Args {
    void const* send;
    int const* sc;
    int const* so;
    void* recv;
    int const* rc;
    int const* ro;
  };
  Args args{send, sc, so, recv, rc, ro};
  fake::collective(4, &args, [=](auto const& all) {
    for (int dst = 0; dst < fake::ranks; ++dst)
      for (int src = 0; src < fake::ranks; ++src) {
        auto* s = static_cast<Args*>(all[src]);
        auto* r = static_cast<Args*>(all[dst]);
        assert(s->sc[dst] == r->rc[src]);
        if (s->sc[dst])
          std::memcpy(
              static_cast<char*>(r->recv) + r->ro[src] * fake::type_size(type),
              static_cast<char const*>(s->send) +
                  s->so[dst] * fake::type_size(type),
              s->sc[dst] * fake::type_size(type));
      }
  });
  return MPI_SUCCESS;
}
