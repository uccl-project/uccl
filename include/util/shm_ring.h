// Local IPC inbox rings over POSIX shared memory.
//
// Three mechanisms keep the ring name, the ring inode and the ring state
// consistent across processes:
//
//   sidecar lock  flock(LOCK_EX) on "<ring name>.lck", an object that is
//                 never unlinked. It serialises every decision made on the
//                 name (create, reclaim, attach, destroy), so no two
//                 processes can act on different inodes under one name.
//   ring lock     flock(LOCK_EX) on the ring fd, held by the owner for the
//                 ring's lifetime. The kernel releases it however the
//                 owner ends (SIGKILL included), so a process that holds
//                 the sidecar and acquires the ring lock knows the previous
//                 owner is gone.
//   ready marker  jring_t::reserved[0]: kShmRingIniting before jring_init,
//                 kShmRingReady after. Attach consumers never enqueue into
//                 a half-initialised ring.
//
// A ring without the marker (reserved[0] == 0 but a sane jring header) was
// built by a binary without this protocol and is treated as legacy: it is
// never reclaimed and never liveness-probed -- its owner does not take the
// ring lock, so "lock acquired" would say nothing about its fate -- and
// attaches keep their historical semantics.
#ifndef UCCL_UTIL_SHM_RING_H_
#define UCCL_UTIL_SHM_RING_H_

#include "jring.h"
#include <chrono>
#include <string>
#include <thread>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace uccl {

// Ring metadata readiness marker, stored in jring_t::reserved[0] (unused
// elsewhere in the tree). jring_init never writes reserved[0], so the
// marker of a ring built by this code goes 0 -> INITING -> READY and stays
// READY.
constexpr uint32_t kShmRingIniting = 0x494E4954;  // "INIT"
constexpr uint32_t kShmRingReady = 0x52454144;    // "READ"

// Bound for create_shared_ring's retry loop. The sidecar lock already
// serialises every well-behaved process, so retries only absorb
// interference from binaries that predate the sidecar.
constexpr int kShmRingRetryMs = 2000;

enum class ShmRingClass {
  kOurs,           // marker INITING or READY
  kLegacy,         // no marker, but jring_init has run
  kUninitialised,  // all-zero: a creator has not finished initialising
};

static inline ShmRingClass shm_ring_classify(jring_t* ring) {
  uint32_t state = __atomic_load_n(&ring->reserved[0], __ATOMIC_ACQUIRE);
  if (state == kShmRingReady || state == kShmRingIniting)
    return ShmRingClass::kOurs;
  if (__atomic_load_n(&ring->size, __ATOMIC_ACQUIRE) != 0)
    return ShmRingClass::kLegacy;
  return ShmRingClass::kUninitialised;
}

// Bounded wait for the creator to finish initialising the ring. Marker
// values:
//   kShmRingReady   -> usable.
//   kShmRingIniting -> creator is still in jring_init; keep waiting.
//   0               -> decide on content: size != 0 means jring_init ran
//                      (legacy ring) and is accepted at once; all-zero
//                      means initialisation has not started, so wait.
static inline bool shm_ring_wait_ready(jring_t* ring) {
  for (int i = 0; i < 1000; i++) {
    uint32_t state = __atomic_load_n(&ring->reserved[0], __ATOMIC_ACQUIRE);
    if (state == kShmRingReady) return true;
    if (state == 0 && __atomic_load_n(&ring->size, __ATOMIC_ACQUIRE) != 0)
      return true;  // legacy ring
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

// shm_open with the failure diagnostics for an attach attempt: name the
// error, and on EACCES name the likely cause (a ring left by another
// account).
static inline int shm_ring_open_reported(char const* shm_name, int flags) {
  int fd = shm_open(shm_name, flags, 0666);
  if (fd >= 0) return fd;
  int const err = errno;
  fprintf(stderr, "shm_open attach failed for %s: %s\n", shm_name,
          strerror(err));
  if (err == EACCES) {
    struct stat st;
    std::string const path = std::string("/dev/shm") + shm_name;
    if (stat(path.c_str(), &st) == 0) {
      fprintf(stderr,
              "  %s is owned by uid %u with mode %o; this process runs as "
              "uid %u\n",
              path.c_str(), static_cast<unsigned>(st.st_uid),
              static_cast<unsigned>(st.st_mode & 0777),
              static_cast<unsigned>(geteuid()));
    }
  }
  errno = err;
  return -1;
}

// The sidecar object for a ring name. Never unlinked; O_CREAT idempotent.
static inline int shm_ring_sidecar_open(char const* shm_name) {
  std::string const path = std::string(shm_name) + ".lck";
  int fd = shm_open(path.c_str(), O_CREAT | O_RDWR, 0666);
  if (fd < 0) {
    int const err = errno;
    fprintf(stderr, "shm_open sidecar failed for %s: %s\n", path.c_str(),
            strerror(err));
    errno = err;
  }
  return fd;
}

// Initialise a freshly created, zero-filled object and take ownership of
// it. Takes ownership of shm_fd (released on failure together with the
// name). Returns the mapping.
static inline jring_t* shm_ring_init_fresh(char const* shm_name, int shm_fd,
                                           size_t shm_size, size_t element_size,
                                           size_t element_count) {
  // The ownership lock must be on the fd before the ring becomes visible
  // as "ours"; a fresh object has no other holder.
  if (flock(shm_fd, LOCK_EX | LOCK_NB) != 0) {
    perror("flock failed");
    close(shm_fd);
    shm_unlink(shm_name);
    return nullptr;
  }
  if (ftruncate(shm_fd, static_cast<off_t>(shm_size)) < 0) {
    perror("ftruncate failed");
    close(shm_fd);
    shm_unlink(shm_name);
    return nullptr;
  }
  void* ptr =
      mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    perror("mmap failed");
    close(shm_fd);
    shm_unlink(shm_name);
    return nullptr;
  }
  auto* ring = reinterpret_cast<jring_t*>(ptr);
  // Publish "not ready" before overwriting the ring metadata, so a
  // concurrent attacher never consumes a half-initialised ring.
  __atomic_store_n(&ring->reserved[0], kShmRingIniting, __ATOMIC_RELEASE);
  if (jring_init(ring, element_count, element_size, 1, 1) < 0) {
    munmap(ptr, shm_size);
    close(shm_fd);
    shm_unlink(shm_name);
    return nullptr;
  }
  __atomic_store_n(&ring->reserved[0], kShmRingReady, __ATOMIC_RELEASE);
  return ring;
}

// Map an already-opened ring object and verify it is usable. Takes
// ownership of shm_fd (closed on failure). Caller holds the sidecar lock.
// Sets errno to ENOENT when the object must not be adopted (stale, no live
// owner) so the caller can retry after the owner materialises.
static inline jring_t* shm_ring_attach_locked(char const* shm_name, int shm_fd,
                                              size_t& shm_size) {
  struct stat st;
  if (fstat(shm_fd, &st) < 0 || st.st_size == 0) {
    if (errno == 0) errno = ENOENT;
    close(shm_fd);
    return nullptr;
  }
  // Auto-detect size from the shared memory object if not provided. Never
  // map past the end of the object: touching such a mapping would SIGBUS.
  if (shm_size == 0) {
    shm_size = static_cast<size_t>(st.st_size);
  } else if (static_cast<size_t>(st.st_size) < shm_size) {
    errno = ENOENT;  // owner may still be sizing it; retry
    close(shm_fd);
    return nullptr;
  }
  void* ptr =
      mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    perror("mmap attach failed");
    close(shm_fd);
    return nullptr;
  }
  auto* ring = reinterpret_cast<jring_t*>(ptr);

  if (shm_ring_classify(ring) != ShmRingClass::kLegacy) {
    // Ours or uninitialised: require a live owner. If nobody holds the
    // ownership lock the ring is stale (or its creator died mid-init) and
    // must not be adopted here -- create_shared_ring replaces it. The
    // probe lock is released at once so it never delays a reclaim.
    if (flock(shm_fd, LOCK_SH | LOCK_NB) == 0) {
      flock(shm_fd, LOCK_UN);
      errno = ENOENT;
      munmap(ptr, shm_size);
      close(shm_fd);
      return nullptr;
    }
    if (errno != EWOULDBLOCK && errno != EAGAIN) {
      perror("flock failed");
      munmap(ptr, shm_size);
      close(shm_fd);
      return nullptr;
    }
  }

  if (!shm_ring_wait_ready(ring)) {
    errno = ETIMEDOUT;
    munmap(ptr, shm_size);
    close(shm_fd);
    return nullptr;
  }
  return ring;
}

// Decide what to do with an existing ring object and do it. Caller holds
// the sidecar lock. On failure sets errno: EACCES is fatal (reported),
// anything else is transient and worth retrying.
static inline jring_t* shm_ring_create_locked(char const* shm_name,
                                              size_t element_size,
                                              size_t element_count, int& shm_fd,
                                              size_t& shm_size,
                                              bool* is_creator) {
  shm_size = jring_get_buf_ring_size(element_size, element_count);

  shm_fd = shm_open(shm_name, O_CREAT | O_EXCL | O_RDWR, 0666);
  if (shm_fd >= 0) {
    *is_creator = true;
    return shm_ring_init_fresh(shm_name, shm_fd, shm_size, element_size,
                               element_count);
  }
  if (errno != EEXIST) {
    fprintf(stderr, "shm_open failed for %s: %s\n", shm_name, strerror(errno));
    return nullptr;
  }

  // Name present: classify before touching anything, the sidecar lock
  // guarantees the state cannot change under us.
  shm_fd = shm_ring_open_reported(shm_name, O_RDWR);
  if (shm_fd < 0) return nullptr;  // errno from open; EACCES is fatal

  struct stat st;
  if (fstat(shm_fd, &st) < 0 || st.st_size < shm_size) {
    if (errno == 0) errno = ENOENT;
    close(shm_fd);
    return nullptr;
  }
  void* ptr =
      mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    perror("mmap failed");
    close(shm_fd);
    return nullptr;
  }
  auto* ring = reinterpret_cast<jring_t*>(ptr);

  if (shm_ring_classify(ring) == ShmRingClass::kLegacy) {
    // Built by a binary without this protocol. Its owner does not hold the
    // ring lock, so a successful lock here would prove nothing; never
    // reclaim such a ring and keep the historical EEXIST semantics.
    *is_creator = false;
    if (!shm_ring_wait_ready(ring)) {
      errno = ETIMEDOUT;
      munmap(ptr, shm_size);
      close(shm_fd);
      return nullptr;
    }
    return ring;
  }

  if (flock(shm_fd, LOCK_EX | LOCK_NB) == 0) {
    // Nobody owns the ring: the previous creator is gone (or died before
    // finishing init). Replace the object: peers of the dead creator may
    // still hold mappings of this inode, and re-initialising it in place
    // would make their accesses SIGBUS. The old inode stays alive until
    // those peers drop it.
    shm_unlink(shm_name);
    munmap(ptr, shm_size);
    close(shm_fd);
    shm_fd = shm_open(shm_name, O_CREAT | O_EXCL | O_RDWR, 0666);
    if (shm_fd < 0) return nullptr;  // transient; retry
    *is_creator = true;
    return shm_ring_init_fresh(shm_name, shm_fd, shm_size, element_size,
                               element_count);
  }
  if (errno != EWOULDBLOCK && errno != EAGAIN) {
    perror("flock failed");
    munmap(ptr, shm_size);
    close(shm_fd);
    return nullptr;
  }

  // Live owner (the sidecar lock keeps it from reclaiming right now).
  *is_creator = false;
  if (!shm_ring_wait_ready(ring)) {
    errno = ETIMEDOUT;
    munmap(ptr, shm_size);
    close(shm_fd);
    return nullptr;
  }
  return ring;
}

// for consumer
static inline jring_t* attach_shared_ring(char const* shm_name, int& shm_fd,
                                          size_t& shm_size) {
  int const sidecar = shm_ring_sidecar_open(shm_name);
  if (sidecar < 0) return nullptr;
  if (flock(sidecar, LOCK_EX) != 0) {
    perror("flock sidecar failed");
    close(sidecar);
    return nullptr;
  }
  shm_fd = shm_ring_open_reported(shm_name, O_RDWR);
  jring_t* ring = nullptr;
  if (shm_fd >= 0) ring = shm_ring_attach_locked(shm_name, shm_fd, shm_size);
  flock(sidecar, LOCK_UN);
  close(sidecar);
  return ring;
}

// for creator
//
// On EEXIST the sidecar lock serialises the create/reclaim/attach decision
// and the ring lock tells a live owner from a dead one; the two together
// resolve a concurrent start to exactly one initialiser and attachers on
// the one ring that initialiser owns. See the file header for the roles of
// the three mechanisms and the legacy-ring carve-out.
static inline jring_t* create_shared_ring(char const* shm_name,
                                          size_t element_size,
                                          size_t element_count, int& shm_fd,
                                          size_t& shm_size, bool* is_creator) {
  int const sidecar = shm_ring_sidecar_open(shm_name);
  if (sidecar < 0) return nullptr;
  if (flock(sidecar, LOCK_EX) != 0) {
    perror("flock sidecar failed");
    close(sidecar);
    return nullptr;
  }

  jring_t* ring = nullptr;
  auto const deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kShmRingRetryMs);
  for (;;) {
    ring = shm_ring_create_locked(shm_name, element_size, element_count, shm_fd,
                                  shm_size, is_creator);
    if (ring != nullptr || errno == EACCES) break;
    if (std::chrono::steady_clock::now() >= deadline) {
      fprintf(stderr, "shm_open failed for %s: %s\n", shm_name,
              strerror(errno));
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  flock(sidecar, LOCK_UN);
  close(sidecar);
  return ring;
}

// for consumer
static inline void detach_shared_ring(jring_t* ring, int shm_fd,
                                      size_t shm_size) {
  if (ring) {
    munmap(reinterpret_cast<void*>(ring), shm_size);
  }

  if (shm_fd >= 0) {
    close(shm_fd);
  }
}

// for creator
static inline void destroy_shared_ring(char const* shm_name, jring_t* ring,
                                       int shm_fd, size_t shm_size) {
  // Unlink under the sidecar lock so no concurrent create/reclaim decision
  // can adopt an object that is about to disappear, and while this process
  // still holds the ownership lock on shm_fd.
  int const sidecar = shm_ring_sidecar_open(shm_name);
  if (sidecar >= 0) {
    if (flock(sidecar, LOCK_EX) == 0) {
      shm_unlink(shm_name);
      flock(sidecar, LOCK_UN);
    } else {
      perror("flock sidecar failed");
      shm_unlink(shm_name);
    }
    close(sidecar);
  } else {
    shm_unlink(shm_name);
  }
  detach_shared_ring(ring, shm_fd, shm_size);
}

}  // namespace uccl

#endif  // UCCL_UTIL_SHM_RING_H_
