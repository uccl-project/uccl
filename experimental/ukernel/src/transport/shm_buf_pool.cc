#include "shm_buf_pool.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

ShmBufPool::ShmBufPool(std::string host_id)
    : host_id_(std::move(host_id)) {
  reg_shm_name_ = "/uccl_shm_buf_reg_" + host_id_;
  size_t reg_size = sizeof(Registry);

  // Try create (first process on this host).
  reg_shm_fd_ = shm_open(reg_shm_name_.c_str(),
                          O_CREAT | O_EXCL | O_RDWR, 0666);
  if (reg_shm_fd_ >= 0) {
    is_creator_ = true;
    if (ftruncate(reg_shm_fd_, static_cast<off_t>(reg_size)) < 0) {
      close(reg_shm_fd_);
      shm_unlink(reg_shm_name_.c_str());
      throw std::runtime_error("ShmBufPool: ftruncate registry failed");
    }
    void* ptr =
        mmap(nullptr, reg_size, PROT_READ | PROT_WRITE, MAP_SHARED, reg_shm_fd_, 0);
    if (ptr == MAP_FAILED) {
      close(reg_shm_fd_);
      shm_unlink(reg_shm_name_.c_str());
      throw std::runtime_error("ShmBufPool: mmap registry failed");
    }
    reg_ = reinterpret_cast<Registry*>(ptr);
    reg_->magic = kMagic;
    reg_->num_slots = static_cast<uint32_t>(kMaxSlots);
    reg_->next_buffer_id = 1;
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&reg_->mu, &attr);
    pthread_mutexattr_destroy(&attr);
    for (size_t i = 0; i < kMaxSlots; ++i) {
      reg_->slots[i].state.store(0, std::memory_order_relaxed);
      reg_->slots[i].buffer_id = 0;
      reg_->slots[i].capacity = 0;
      reg_->slots[i].shm_name[0] = '\0';
    }
    return;
  }

  // Open existing registry (other process).
  if (errno != EEXIST) {
    throw std::runtime_error("ShmBufPool: shm_open registry failed: " +
                             std::string(strerror(errno)));
  }
  is_creator_ = false;
  reg_shm_fd_ = shm_open(reg_shm_name_.c_str(), O_RDWR, 0666);
  if (reg_shm_fd_ < 0) {
    throw std::runtime_error("ShmBufPool: open existing registry failed");
  }
  struct stat st {};
  if (fstat(reg_shm_fd_, &st) != 0 || st.st_size < static_cast<off_t>(reg_size)) {
    close(reg_shm_fd_);
    throw std::runtime_error("ShmBufPool: registry size check failed");
  }
  void* ptr =
      mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ | PROT_WRITE,
           MAP_SHARED, reg_shm_fd_, 0);
  if (ptr == MAP_FAILED) {
    close(reg_shm_fd_);
    throw std::runtime_error("ShmBufPool: mmap existing registry failed");
  }
  reg_ = reinterpret_cast<Registry*>(ptr);
  if (reg_->magic != kMagic) {
    munmap(ptr, static_cast<size_t>(st.st_size));
    close(reg_shm_fd_);
    throw std::runtime_error("ShmBufPool: bad registry magic");
  }
}

ShmBufPool::~ShmBufPool() {
  // Unmap locally cached data SHMs.
  {
    std::lock_guard<std::mutex> lk(local_cache_mu_);
    for (auto& [buf_id, m] : local_cache_) {
      if (m.ptr != nullptr) munmap(m.ptr, m.capacity);
      if (m.shm_fd >= 0) close(m.shm_fd);
    }
    local_cache_.clear();
  }
  if (reg_ != nullptr) {
    if (is_creator_) {
      pthread_mutex_destroy(&reg_->mu);
      // Unlink all data SHMs the creator made.
      for (size_t i = 0; i < reg_->num_slots; ++i) {
        if (reg_->slots[i].shm_name[0] != '\0') {
          shm_unlink(reg_->slots[i].shm_name);
        }
      }
    }
    munmap(reinterpret_cast<void*>(reg_), sizeof(Registry));
    reg_ = nullptr;
  }
  if (reg_shm_fd_ >= 0) {
    close(reg_shm_fd_);
    if (is_creator_) shm_unlink(reg_shm_name_.c_str());
  }
}

void* ShmBufPool::open_or_create_data_shm(char const* name, size_t cap,
                                           int* out_fd) {
  *out_fd = -1;
  int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0666);
  if (fd >= 0) {
    // Created by us.
    if (ftruncate(fd, static_cast<off_t>(cap)) < 0) {
      close(fd);
      shm_unlink(name);
      return nullptr;
    }
  } else if (errno == EEXIST) {
    fd = shm_open(name, O_RDWR, 0666);
    if (fd < 0) return nullptr;
    struct stat st {};
    if (fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) < cap) {
      close(fd);
      return nullptr;
    }
  } else {
    return nullptr;
  }
  void* ptr = mmap(nullptr, cap, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    close(fd);
    if (is_creator_) shm_unlink(name);
    return nullptr;
  }
  *out_fd = fd;
  return ptr;
}

ShmBufPool::Slot* ShmBufPool::find_free_slot_locked(size_t min_bytes) {
  for (size_t i = 0; i < reg_->num_slots; ++i) {
    auto& slot = reg_->slots[i];
    uint32_t st = slot.state.load(std::memory_order_acquire);
    if (st != 0) continue;
    if (slot.capacity >= min_bytes) {
      // Reuse existing slot with enough capacity.
      if (slot.state.compare_exchange_strong(st, 1, std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
        return &slot;
      }
      continue;
    }
    if (slot.capacity == 0 && slot.shm_name[0] == '\0') {
      // Empty slot — allocate new data SHM.
      if (slot.state.compare_exchange_strong(st, 1, std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
        return &slot;
      }
    }
  }
  return nullptr;
}

ShmBufPool::Slot* ShmBufPool::find_slot_by_id_locked(uint32_t buffer_id) {
  for (size_t i = 0; i < reg_->num_slots; ++i) {
    if (reg_->slots[i].buffer_id == buffer_id &&
        reg_->slots[i].state.load(std::memory_order_acquire) != 0) {
      return &reg_->slots[i];
    }
  }
  return nullptr;
}

ShmBufSlotInfo ShmBufPool::acquire(size_t min_bytes) {
  if (reg_ == nullptr) return {};

  pthread_mutex_lock(&reg_->mu);

  Slot* slot = find_free_slot_locked(min_bytes);
  if (slot == nullptr) {
    pthread_mutex_unlock(&reg_->mu);
    std::cerr << "[ERROR] ShmBufPool: no free slot for " << min_bytes
              << " bytes" << std::endl;
    return {};
  }

  ShmBufSlotInfo info{};

  if (slot->capacity >= min_bytes) {
    // Reuse existing buffer.
    info.buffer_id = slot->buffer_id;
    {
      std::lock_guard<std::mutex> lk(local_cache_mu_);
      auto it = local_cache_.find(info.buffer_id);
      if (it != local_cache_.end()) {
        info.ptr = it->second.ptr;
      }
    }
  } else {
    // Allocate new data SHM.
    uint32_t buf_id = reg_->next_buffer_id;
    if (buf_id == 0) buf_id = 1;
    reg_->next_buffer_id = buf_id + 1;
    if (reg_->next_buffer_id == 0) reg_->next_buffer_id = 1;

    size_t cap = min_bytes;
    snprintf(slot->shm_name, sizeof(slot->shm_name),
             "/uccl_shm_buf_%s_%u", host_id_.c_str(), buf_id);

    int fd = -1;
    void* ptr = open_or_create_data_shm(slot->shm_name, cap, &fd);
    if (ptr == nullptr) {
      slot->capacity = 0;
      slot->buffer_id = 0;
      slot->shm_name[0] = '\0';
      slot->state.store(0, std::memory_order_release);
      pthread_mutex_unlock(&reg_->mu);
      return {};
    }

    slot->capacity = cap;
    slot->buffer_id = buf_id;
    info.buffer_id = buf_id;
    info.ptr = ptr;

    {
      std::lock_guard<std::mutex> lk(local_cache_mu_);
      local_cache_[buf_id] = {ptr, fd, cap};
    }
  }

  // If the caller didn't get ptr yet (reuse case), map it now.
  if (info.ptr == nullptr) {
    int fd = shm_open(slot->shm_name, O_RDWR, 0666);
    if (fd < 0) {
      slot->state.store(0, std::memory_order_release);
      pthread_mutex_unlock(&reg_->mu);
      return {};
    }
    void* ptr =
        mmap(nullptr, slot->capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
      close(fd);
      slot->state.store(0, std::memory_order_release);
      pthread_mutex_unlock(&reg_->mu);
      return {};
    }
    info.ptr = ptr;
    {
      std::lock_guard<std::mutex> lk(local_cache_mu_);
      local_cache_[info.buffer_id] = {ptr, fd, slot->capacity};
    }
  }

  pthread_mutex_unlock(&reg_->mu);
  return info;
}

void ShmBufPool::release(uint32_t buffer_id) {
  if (reg_ == nullptr) return;
  pthread_mutex_lock(&reg_->mu);
  Slot* slot = find_slot_by_id_locked(buffer_id);
  if (slot != nullptr) {
    slot->state.store(0, std::memory_order_release);
  }
  pthread_mutex_unlock(&reg_->mu);
}

void* ShmBufPool::get_ptr(uint32_t buffer_id) {
  if (reg_ == nullptr) return nullptr;

  // Check local cache first.
  {
    std::lock_guard<std::mutex> lk(local_cache_mu_);
    auto it = local_cache_.find(buffer_id);
    if (it != local_cache_.end()) return it->second.ptr;
  }

  // Look up from registry and mmap on demand.
  pthread_mutex_lock(&reg_->mu);
  Slot* slot = find_slot_by_id_locked(buffer_id);
  if (slot == nullptr || slot->shm_name[0] == '\0') {
    pthread_mutex_unlock(&reg_->mu);
    return nullptr;
  }
  // Copy shm_name before unlock.
  char name[128];
  std::strncpy(name, slot->shm_name, sizeof(name) - 1);
  name[sizeof(name) - 1] = '\0';
  size_t cap = slot->capacity;
  pthread_mutex_unlock(&reg_->mu);

  int fd = shm_open(name, O_RDWR, 0666);
  if (fd < 0) return nullptr;
  void* ptr = mmap(nullptr, cap, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    close(fd);
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lk(local_cache_mu_);
    local_cache_[buffer_id] = {ptr, fd, cap};
  }
  return ptr;
}

}  // namespace Transport
}  // namespace UKernel
