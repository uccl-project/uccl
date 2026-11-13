// Stress and diagnosis harness for the local IPC inbox rings.
//
// Mirrors the review experiment (0z5a): N processes call
// create_shared_ring on one name at the same moment for many rounds, and
// every round is classified by the outcome:
//
//   ok            all N got a ring, exactly one creator, one inode
//   two-creators  more than one is_creator (ownership split)
//   two-inodes    children landed on different inodes (ring split)
//   nlink0        a child held a ring whose inode was already unlinked
//                 (YangZhou1997's destroy-race observation)
//   zero-creator  everyone got a ring but nobody owns it (a stale ring was
//                 silently adopted)
//   null          at least one child got nullptr ("name kept changing")
//
// Additional modes:
//   ops        create/hold/destroy hammer; counts cross-unlinks (a creator
//              about to destroy finds the name pointing at another inode)
//   attachrace stale ring, then reclaimer and attacher race; the attacher
//              must land on the reclaimer's inode, never the stale one
//   soak       many create/destroy cycles in one process; fd and name leaks
//
// The implementation under test is selected at compile time so the same
// harness compares upstream main, the PR as reviewed, and the reworked
// protocol: -DSHM_RING_HEADER='"shm_ring_xxx.h"'.
#ifndef SHM_RING_HEADER
#define SHM_RING_HEADER "shm_ring.h"
#endif
#include SHM_RING_HEADER

#include <string>
#include <vector>
#include <dirent.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

namespace {

std::string g_name;

struct ChildResult {
  int ok;
  int is_creator;
  int nlink;
  uint64_t ino;
};

uint64_t ino_of(int fd) {
  struct stat st;
  if (fstat(fd, &st) < 0) return 0;
  return static_cast<uint64_t>(st.st_ino);
}

int nlink_of(int fd) {
  struct stat st;
  if (fstat(fd, &st) < 0) return -1;
  return static_cast<int>(st.st_nlink);
}

uint64_t name_ino() {
  struct stat st;
  std::string const path = "/dev/shm" + g_name;
  if (stat(path.c_str(), &st) < 0) return 0;
  return static_cast<uint64_t>(st.st_ino);
}

void clean_name() {
  shm_unlink(g_name.c_str());
  std::string const lck = g_name + ".lck";
  shm_unlink(lck.c_str());
}

void make_stale_ring() {
  fflush(stdout);
  pid_t victim = fork();
  if (victim == 0) {
    int fd = -1;
    size_t sz = 0;
    bool is_creator = false;
    jring_t* ring =
        uccl::create_shared_ring(g_name.c_str(), 64, 16, fd, sz, &is_creator);
    if (!ring || !is_creator) _exit(1);
    raise(SIGKILL);
    _exit(0);
  }
  int st;
  waitpid(victim, &st, 0);
}

int read_full(int fd, void* buf, size_t n) {
  size_t got = 0;
  while (got < n) {
    ssize_t r = read(fd, static_cast<char*>(buf) + got, n - got);
    if (r <= 0) return -1;
    got += static_cast<size_t>(r);
  }
  return 0;
}

// ---- race: the review experiment -------------------------------------

struct RaceStats {
  long rounds = 0, ok = 0, two_creators = 0, two_inodes = 0, nlink0 = 0,
       zero_creator = 0, nulls = 0;
};

void race_round(int n, bool with_stale, RaceStats& s) {
  clean_name();
  if (with_stale) make_stale_ring();

  int start[2], done[2], out[2];
  pipe(start);
  pipe(done);
  pipe(out);
  std::vector<pid_t> kids;
  for (int i = 0; i < n; i++) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
      close(start[1]);
      close(done[1]);
      close(out[0]);
      char c;
      (void)!read(start[0], &c, 1);
      ChildResult r{0, 0, 0, 0};
      int fd = -1;
      size_t sz = 0;
      bool is_creator = false;
      jring_t* ring =
          uccl::create_shared_ring(g_name.c_str(), 64, 16, fd, sz, &is_creator);
      r.ok = ring != nullptr;
      if (r.ok) {
        r.is_creator = is_creator;
        r.nlink = nlink_of(fd);
        r.ino = ino_of(fd);
      }
      (void)!write(out[1], &r, sizeof(r));
      (void)!read(done[0], &c, 1);
      if (ring) uccl::detach_shared_ring(ring, fd, sz);
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
  std::vector<ChildResult> res(n);
  for (int i = 0; i < n; i++) read_full(out[0], &res[i], sizeof(res[i]));

  s.rounds++;
  int nulls = 0, creators = 0, nlink0 = 0;
  uint64_t ino0 = 0;
  bool split = false;
  for (int i = 0; i < n; i++) {
    if (!res[i].ok) {
      nulls++;
      continue;
    }
    creators += res[i].is_creator;
    if (res[i].nlink == 0) nlink0++;
    if (ino0 == 0) ino0 = res[i].ino;
    if (res[i].ino != ino0) split = true;
  }
  if (nulls)
    s.nulls++;
  else if (nlink0)
    s.nlink0++;
  else if (split)
    s.two_inodes++;
  else if (creators > 1)
    s.two_creators++;
  else if (creators == 0)
    s.zero_creator++;
  else
    s.ok++;

  for (int i = 0; i < n; i++) {
    char c = 'x';
    (void)!write(done[1], &c, 1);
  }
  for (pid_t pid : kids) {
    int st;
    waitpid(pid, &st, 0);
  }
  close(start[1]);
  close(done[1]);
  close(out[0]);
  clean_name();
}

void run_race(long rounds, int n, bool with_stale) {
  RaceStats s;
  for (long r = 0; r < rounds; r++) race_round(n, with_stale, s);
  printf(
      "N=%d %s: rounds=%ld ok=%ld two-inodes=%ld two-creators=%ld "
      "nlink0=%ld zero-creator=%ld null=%ld\n",
      n, with_stale ? "stale " : "nostale", s.rounds, s.ok, s.two_inodes,
      s.two_creators, s.nlink0, s.zero_creator, s.nulls);
  fflush(stdout);
}

// ---- ops: create/hold/destroy hammer --------------------------------

struct OpsStats {
  long ops = 0, create_null = 0, nlink0_creator = 0, nlink0_attached = 0,
       cross_unlink = 0;
};

void ops_child(int iters, int pipefd, unsigned seed) {
  OpsStats s;
  for (int i = 0; i < iters; i++) {
    int fd = -1;
    size_t sz = 0;
    bool is_creator = false;
    jring_t* ring =
        uccl::create_shared_ring(g_name.c_str(), 64, 16, fd, sz, &is_creator);
    s.ops++;
    if (!ring) {
      s.create_null++;
      continue;
    }
    // nlink0 on a creator means create_shared_ring returned an inode that
    // is already unlinked (the review's steal-and-replace corruption); on
    // an attacher it can be benign churn (the owner destroyed right after
    // we attached).
    if (nlink_of(fd) == 0) {
      if (is_creator)
        s.nlink0_creator++;
      else
        s.nlink0_attached++;
    }
    seed = seed * 1103515245u + 12345u;  // hold a random beat
    usleep((seed >> 16) % 2000);
    if (is_creator) {
      if (name_ino() != ino_of(fd)) s.cross_unlink++;
      uccl::destroy_shared_ring(g_name.c_str(), ring, fd, sz);
    } else {
      uccl::detach_shared_ring(ring, fd, sz);
    }
  }
  (void)!write(pipefd, &s, sizeof(s));
  _exit(0);
}

void run_ops(int n, int iters) {
  clean_name();
  int out[2];
  pipe(out);
  std::vector<pid_t> kids;
  for (int i = 0; i < n; i++) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
      close(out[0]);
      ops_child(iters, out[1], (unsigned)(getpid() * 2654435761u));
    }
    kids.push_back(pid);
  }
  close(out[1]);
  OpsStats total;
  for (int i = 0; i < n; i++) {
    OpsStats s;
    if (read_full(out[0], &s, sizeof(s)) == 0) {
      total.ops += s.ops;
      total.create_null += s.create_null;
      total.nlink0_creator += s.nlink0_creator;
      total.nlink0_attached += s.nlink0_attached;
      total.cross_unlink += s.cross_unlink;
    }
  }
  for (pid_t pid : kids) {
    int st;
    waitpid(pid, &st, 0);
  }
  printf(
      "ops N=%d x %d: ops=%ld create-null=%ld nlink0-creator=%ld "
      "nlink0-attached=%ld cross-unlink=%ld\n",
      n, iters, total.ops, total.create_null, total.nlink0_creator,
      total.nlink0_attached, total.cross_unlink);
  fflush(stdout);
  clean_name();
}

// ---- attachrace: attacher must not land on a stale inode --------------

void run_attachrace(long rounds) {
  long good = 0, stale_attach = 0, refused = 0;
  for (long r = 0; r < rounds; r++) {
    clean_name();
    make_stale_ring();
    uint64_t const stale_ino = name_ino();

    int rec_out[2], att_out[2];
    pipe(rec_out);
    pipe(att_out);
    fflush(stdout);
    pid_t rec = fork();
    if (rec == 0) {  // reclaimer: take the ring and hold it briefly
      close(rec_out[0]);
      close(att_out[0]);
      close(att_out[1]);
      int fd = -1;
      size_t sz = 0;
      bool is_creator = false;
      jring_t* ring =
          uccl::create_shared_ring(g_name.c_str(), 64, 16, fd, sz, &is_creator);
      uint64_t ino = ring ? ino_of(fd) : 0;
      (void)!write(rec_out[1], &ino, sizeof(ino));
      usleep(300000);  // hold while the attacher retries
      if (ring) uccl::destroy_shared_ring(g_name.c_str(), ring, fd, sz);
      _exit(0);
    }
    pid_t att = fork();
    if (att == 0) {  // attacher: retry like connect_local does
      close(rec_out[0]);
      close(rec_out[1]);
      close(att_out[0]);
      int fd = -1;
      size_t sz = 0;
      uint64_t msg[2] = {0, 0};
      for (int i = 0; i < 300; i++) {
        jring_t* ring = uccl::attach_shared_ring(g_name.c_str(), fd, sz);
        if (ring) {
          msg[0] = 1;
          msg[1] = ino_of(fd);
          uccl::detach_shared_ring(ring, fd, sz);
          break;
        }
        usleep(10000);
      }
      (void)!write(att_out[1], msg, sizeof(msg));
      _exit(0);
    }
    close(rec_out[1]);
    close(att_out[1]);
    uint64_t rec_ino = 0, msg[2] = {0, 0};
    read_full(rec_out[0], &rec_ino, sizeof(rec_ino));
    read_full(att_out[0], msg, sizeof(msg));
    int st;
    waitpid(rec, &st, 0);
    waitpid(att, &st, 0);
    close(rec_out[0]);
    close(att_out[0]);

    if (!msg[0]) {
      refused++;
    } else if (msg[1] == stale_ino) {
      stale_attach++;
    } else if (rec_ino != 0 && msg[1] == rec_ino) {
      good++;
    } else {
      stale_attach++;  // attached to something neither stale nor reclaimed
    }
    clean_name();
  }
  printf(
      "attachrace rounds=%ld attach-to-reclaimed=%ld attach-to-stale=%ld "
      "refused=%ld\n",
      rounds, good, stale_attach, refused);
  fflush(stdout);
}

// ---- soak: fd and name leaks -----------------------------------------

int fd_count() {
  DIR* d = opendir("/proc/self/fd");
  if (!d) return -1;
  int n = 0;
  while (readdir(d)) n++;
  closedir(d);
  return n;
}

long residue_count(bool sidecars) {
  DIR* d = opendir("/dev/shm");
  if (!d) return -1;
  long n = 0;
  while (struct dirent* e = readdir(d)) {
    if (strstr(e->d_name, "uccl_shm_ring_soak") != e->d_name) continue;
    bool const is_lck = strstr(e->d_name, ".lck") != nullptr;
    if (is_lck == sidecars) n++;
  }
  closedir(d);
  return n;
}

void run_soak(int iters) {
  g_name = "/uccl_shm_ring_soak_" + std::to_string(getpid());
  clean_name();
  int const fd0 = fd_count();
  for (int i = 0; i < iters; i++) {
    int fd = -1;
    size_t sz = 0;
    bool is_creator = false;
    jring_t* ring =
        uccl::create_shared_ring(g_name.c_str(), 64, 16, fd, sz, &is_creator);
    if (!ring) {
      printf("soak: create failed at iter %d\n", i);
      break;
    }
    uccl::destroy_shared_ring(g_name.c_str(), ring, fd, sz);
  }
  int const fd1 = fd_count();
  long const ring_residue = residue_count(false);
  long const lck_residue = residue_count(true);
  printf(
      "soak iters=%d fds %d -> %d (delta %d), ring residue %ld, sidecar "
      "residue %ld (one per name is by design)\n",
      iters, fd0, fd1, fd1 - fd0, ring_residue, lck_residue);
  fflush(stdout);
  clean_name();
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "race";
  if (mode == "race") {
    long rounds = argc > 2 ? atol(argv[2]) : 2000;
    g_name = "/uccl_shm_ring_race_" + std::to_string(getpid());
    run_race(rounds, 2, true);
    run_race(rounds, 2, false);
    run_race(rounds, 4, true);
    run_race(rounds, 4, false);
  } else if (mode == "ops") {
    int n = argc > 2 ? atoi(argv[2]) : 4;
    int iters = argc > 3 ? atoi(argv[3]) : 500;
    g_name = "/uccl_shm_ring_ops_" + std::to_string(getpid());
    run_ops(n, iters);
  } else if (mode == "attachrace") {
    long rounds = argc > 2 ? atol(argv[2]) : 200;
    g_name = "/uccl_shm_ring_ar_" + std::to_string(getpid());
    run_attachrace(rounds);
  } else if (mode == "soak") {
    int iters = argc > 2 ? atoi(argv[2]) : 5000;
    run_soak(iters);
  } else {
    fprintf(stderr,
            "usage: %s [race [rounds]|ops [n iters]|attachrace "
            "[rounds]|soak [iters]]\n",
            argv[0]);
    return 2;
  }
  return 0;
}
