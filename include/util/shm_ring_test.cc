// Lifecycle and reclaim tests for the local IPC inbox rings in shm_ring.h.
// Pure CPU: POSIX shared memory and flock only, no GPU or RDMA device.
//
// The concurrency cases model the two races reported on the PR:
//   - N processes creating one name at the same moment must resolve to one
//     initialiser on one inode (0z5a's stress: two inodes / null returns);
//   - a peer mapping of a stale inode must survive the reclaim untouched
//     (ftruncate-reclaim SIGBUS) and must not be adopted by attach when the
//     ring has no live owner.
#include "shm_ring.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

void require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}

std::string test_name(char const* suffix) {
  return "/uccl_shm_ring_test_" + std::to_string(getpid()) + "_" + suffix;
}

struct ChildResult {
  int ok;
  int is_creator;
  uint64_t ino;
};

uint64_t ino_of(int fd) {
  struct stat st;
  if (fstat(fd, &st) < 0) return 0;
  return static_cast<uint64_t>(st.st_ino);
}

// Leave a stale ring behind: create it and kill the owner with SIGKILL so
// no destructor runs (same shape as the issue's reproduction).
void make_stale_ring(std::string const& name) {
  fflush(stdout);
  pid_t victim = fork();
  if (victim == 0) {
    int fd = -1;
    size_t sz = 0;
    bool is_creator = false;
    jring_t* ring =
        uccl::create_shared_ring(name.c_str(), 64, 16, fd, sz, &is_creator);
    if (!ring || !is_creator) _exit(1);
    raise(SIGKILL);
    _exit(0);
  }
  int st;
  waitpid(victim, &st, 0);
  require(WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL,
          "stale-ring victim did not die to SIGKILL");
}

// A ring object with a sane jring header but no readiness marker: what a
// binary without this protocol leaves behind.
void make_legacy_ring(std::string const& name) {
  shm_unlink(name.c_str());
  int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
  require(fd >= 0, "legacy setup: shm_open");
  size_t sz = jring_get_buf_ring_size(64, 16);
  require(ftruncate(fd, static_cast<off_t>(sz)) == 0, "legacy setup: truncate");
  void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  require(p != MAP_FAILED, "legacy setup: mmap");
  require(jring_init(reinterpret_cast<jring_t*>(p), 16, 64, 1, 1) == 0,
          "legacy setup: jring_init");
  munmap(p, sz);
  close(fd);
}

void test_fresh_create_and_destroy() {
  std::string const name = test_name("fresh");
  int fd = -1;
  size_t sz = 0;
  bool is_creator = false;
  jring_t* ring =
      uccl::create_shared_ring(name.c_str(), 64, 16, fd, sz, &is_creator);
  require(ring != nullptr, "fresh create returned null");
  require(is_creator, "fresh create is not the creator");
  require(__atomic_load_n(&ring->reserved[0], __ATOMIC_ACQUIRE) ==
              uccl::kShmRingReady,
          "fresh create did not publish READY");
  require(jring_empty(ring), "fresh ring not empty");
  uccl::destroy_shared_ring(name.c_str(), ring, fd, sz);
  struct stat st;
  require(stat(("/dev/shm" + name).c_str(), &st) != 0,
          "destroy left the name behind");
}

void test_attach_live_owner() {
  std::string const name = test_name("live");
  int fd = -1;
  size_t sz = 0;
  bool is_creator = false;
  jring_t* ring =
      uccl::create_shared_ring(name.c_str(), 64, 16, fd, sz, &is_creator);
  require(ring != nullptr && is_creator, "owner create failed");

  int fd2 = -1;
  size_t sz2 = 0;
  bool is_creator2 = true;
  jring_t* ring2 =
      uccl::create_shared_ring(name.c_str(), 64, 16, fd2, sz2, &is_creator2);
  require(ring2 != nullptr, "second create returned null");
  require(!is_creator2, "second create stole ownership from the live owner");
  require(ino_of(fd) == ino_of(fd2), "live attach landed on another inode");

  int fd3 = -1;
  size_t sz3 = 0;
  jring_t* ring3 = uccl::attach_shared_ring(name.c_str(), fd3, sz3);
  require(ring3 != nullptr, "attach_shared_ring failed on a live owner");
  require(ino_of(fd) == ino_of(fd3), "attach landed on another inode");

  uccl::detach_shared_ring(ring2, fd2, sz2);
  uccl::detach_shared_ring(ring3, fd3, sz3);
  uccl::destroy_shared_ring(name.c_str(), ring, fd, sz);
}

void test_stale_reclaim_replaces_inode() {
  std::string const name = test_name("reclaim");
  make_stale_ring(name);

  // A peer of the dead owner keeps a mapping of the stale inode across the
  // reclaim; the old mapping must survive untouched (no SIGBUS).
  int peer_fd = -1;
  size_t peer_sz = 0;
  jring_t* peer_map = uccl::attach_shared_ring(name.c_str(), peer_fd, peer_sz);
  require(peer_map == nullptr,
          "attach must not adopt a ring with no live owner");
  // Map the stale inode directly to play the peer that attached while the
  // old owner was still alive (the shipped callers' shape).
  int raw_fd = shm_open(name.c_str(), O_RDWR, 0666);
  require(raw_fd >= 0, "raw open of stale ring");
  size_t raw_sz = jring_get_buf_ring_size(64, 16);
  void* raw =
      mmap(nullptr, raw_sz, PROT_READ | PROT_WRITE, MAP_SHARED, raw_fd, 0);
  require(raw != MAP_FAILED, "raw map of stale ring");
  reinterpret_cast<jring_t*>(raw)->reserved[1] = 0xCAFE;
  uint64_t const stale_ino = ino_of(raw_fd);

  int fd = -1;
  size_t sz = 0;
  bool is_creator = false;
  jring_t* ring =
      uccl::create_shared_ring(name.c_str(), 64, 16, fd, sz, &is_creator);
  require(ring != nullptr, "reclaim returned null");
  require(is_creator, "reclaim did not take ownership");
  require(ino_of(fd) != stale_ino, "reclaim re-used the stale inode");
  require(ring->reserved[1] == 0, "fresh ring carries stale content");
  require(__atomic_load_n(&ring->reserved[0], __ATOMIC_ACQUIRE) ==
              uccl::kShmRingReady,
          "reclaim did not publish READY");

  // The old mapping still reads what the old owner wrote.
  require(reinterpret_cast<jring_t*>(raw)->reserved[1] == 0xCAFE,
          "stale mapping corrupted across reclaim");

  uccl::destroy_shared_ring(name.c_str(), ring, fd, sz);
  munmap(raw, raw_sz);
  close(raw_fd);
}

void test_ready_marker_gating() {
  std::string const name = test_name("marker");
  int fd = -1;
  size_t sz = 0;
  bool is_creator = false;
  jring_t* ring =
      uccl::create_shared_ring(name.c_str(), 64, 16, fd, sz, &is_creator);
  require(ring != nullptr && is_creator, "marker setup create failed");

  __atomic_store_n(&ring->reserved[0], uccl::kShmRingIniting, __ATOMIC_RELEASE);
  auto const t0 = std::chrono::steady_clock::now();
  bool ok = uccl::shm_ring_wait_ready(ring);
  auto const ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  require(!ok, "wait_ready accepted an INITING ring");
  require(ms >= 1000, "wait_ready on INITING returned before the bound");

  __atomic_store_n(&ring->reserved[0], uccl::kShmRingReady, __ATOMIC_RELEASE);
  require(uccl::shm_ring_wait_ready(ring), "wait_ready refused a READY ring");

  // All-zero ring (creator has not initialised yet) must not pass as
  // legacy however long we wait.
  shm_unlink(name.c_str());
  int fdz = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
  require(fdz >= 0, "zero-ring setup: shm_open");
  require(ftruncate(fdz, static_cast<off_t>(sz)) == 0, "zero-ring truncate");
  void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fdz, 0);
  require(p != MAP_FAILED, "zero-ring mmap");
  require(uccl::shm_ring_classify(reinterpret_cast<jring_t*>(p)) ==
              uccl::ShmRingClass::kUninitialised,
          "all-zero ring misclassified");
  ok = uccl::shm_ring_wait_ready(reinterpret_cast<jring_t*>(p));
  require(!ok, "wait_ready accepted an all-zero ring");

  munmap(p, sz);
  close(fdz);
  shm_unlink(name.c_str());
  uccl::destroy_shared_ring(name.c_str(), ring, fd, sz);
}

void test_legacy_ring() {
  std::string const name = test_name("legacy");
  make_legacy_ring(name);

  int fd = -1;
  size_t sz = 0;
  bool is_creator = true;
  jring_t* ring =
      uccl::create_shared_ring(name.c_str(), 64, 16, fd, sz, &is_creator);
  require(ring != nullptr, "legacy ring refused by create_shared_ring");
  require(!is_creator,
          "legacy ring must never be reclaimed (owner may be a live "
          "pre-protocol binary)");

  int fd2 = -1;
  size_t sz2 = 0;
  jring_t* ring2 = uccl::attach_shared_ring(name.c_str(), fd2, sz2);
  require(ring2 != nullptr, "legacy ring refused by attach_shared_ring");

  uccl::detach_shared_ring(ring, fd, sz);
  uccl::detach_shared_ring(ring2, fd2, sz2);
  shm_unlink(name.c_str());
}

// N processes call create_shared_ring on one name at the same instant.
// Everyone must get a ring, exactly one of them owns it, and all of them
// must see one inode. Run with and without a stale ring to cover both of
// 0z5a's columns.
void race_round(std::string const& name, int n, bool with_stale) {
  shm_unlink(name.c_str());
  shm_unlink((name + ".lck").c_str());
  if (with_stale) make_stale_ring(name);

  int start[2], done[2], out[2];
  require(pipe(start) == 0 && pipe(done) == 0 && pipe(out) == 0,
          "race setup: pipe");

  std::vector<pid_t> kids;
  for (int i = 0; i < n; i++) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
      close(start[1]);
      close(done[1]);
      close(out[0]);
      char c;
      (void)!read(start[0], &c, 1);  // release together
      ChildResult r{0, 0, 0};
      int fd = -1;
      size_t sz = 0;
      bool is_creator = false;
      jring_t* ring =
          uccl::create_shared_ring(name.c_str(), 64, 16, fd, sz, &is_creator);
      r.ok = ring != nullptr;
      r.is_creator = is_creator;
      r.ino = r.ok ? ino_of(fd) : 0;
      (void)!write(out[1], &r, sizeof(r));
      (void)!read(done[0], &c, 1);  // hold the ring until the round is read
      _exit(0);
    }
    kids.push_back(pid);
  }
  close(start[0]);
  close(done[0]);
  close(out[1]);
  for (int i = 0; i < n; i++) {
    char c = 'x';
    (void)!write(start[1], &c, 1);
  }

  std::vector<ChildResult> results(n);
  for (int i = 0; i < n; i++) {
    size_t got = 0;
    while (got < sizeof(ChildResult)) {
      ssize_t r = read(out[0], reinterpret_cast<char*>(&results[i]) + got,
                       sizeof(ChildResult) - got);
      require(r > 0, "race setup: result pipe closed early");
      got += static_cast<size_t>(r);
    }
  }

  int creators = 0;
  for (int i = 0; i < n; i++) {
    require(results[i].ok, "a racer got null (name kept changing under us)");
    creators += results[i].is_creator;
    require(results[i].ino == results[0].ino, "racers split across inodes");
  }
  require(creators == 1, "racers did not resolve to exactly one initialiser");

  for (int i = 0; i < n; i++) {
    char c = 'x';
    (void)!write(done[1], &c, 1);
  }
  for (pid_t pid : kids) {
    int st;
    waitpid(pid, &st, 0);
    require(WIFEXITED(st) && WEXITSTATUS(st) == 0, "racer child died");
  }
  shm_unlink(name.c_str());
  shm_unlink((name + ".lck").c_str());
}

void test_concurrent_create_race() {
  std::string const name = test_name("race");
  for (int round = 0; round < 200; round++) race_round(name, 2, true);
  for (int round = 0; round < 200; round++) race_round(name, 2, false);
  for (int round = 0; round < 100; round++) race_round(name, 4, true);
  for (int round = 0; round < 100; round++) race_round(name, 4, false);
}

}  // namespace

int main() {
  try {
    test_fresh_create_and_destroy();
    std::cout << "fresh create/destroy: PASS\n";
    test_attach_live_owner();
    std::cout << "attach to a live owner: PASS\n";
    test_stale_reclaim_replaces_inode();
    std::cout
        << "stale reclaim replaces the inode, peer mapping intact: PASS\n";
    test_ready_marker_gating();
    std::cout << "ready marker gating (INITING/all-zero/READY): PASS\n";
    test_legacy_ring();
    std::cout << "legacy ring never reclaimed, attaches as before: PASS\n";
    test_concurrent_create_race();
    std::cout << "concurrent create race (N=2/4, 600 rounds): PASS\n";
    std::cout << "shm ring lifecycle: PASS\n";
  } catch (std::exception const& e) {
    std::cerr << "shm ring lifecycle: FAIL: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
