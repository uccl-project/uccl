#pragma once
#include "define.h"

template <typename T, size_t Capacity>
class EmptyRingBuffer {
 public:
  explicit EmptyRingBuffer(void* addr)
      : base_addr_(reinterpret_cast<uintptr_t>(addr)) {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two");
  }

  // Get address of element at given index (computed, no memory access)
  uintptr_t getElementAddress(size_t index) const {
    size_t actual_index = index & (Capacity - 1);
    return base_addr_ + actual_index * sizeof(T);
  }

  // Get the theoretical capacity of the ring
  constexpr size_t capacity() const { return Capacity; }

  // Get the total size in bytes
  constexpr size_t sizeInBytes() const { return Capacity * sizeof(T); }

 private:
  uintptr_t base_addr_;
};

template <typename T, size_t Capacity>
class RingBuffer {
 public:
  RingBuffer() : buffer_(new T[Capacity]), owns_buffer_(true) {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two");
  }

  // Constructor accepting pre-allocated RegMemBlock
  explicit RingBuffer(std::shared_ptr<RegMemBlock> mem_block)
      : buffer_(nullptr), owns_buffer_(false), mem_block_(mem_block) {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two");

    // Validate RegMemBlock
    assert(mem_block != nullptr && "RegMemBlock pointer must not be null");
    assert(mem_block->size >= Capacity * sizeof(T) &&
           "RegMemBlock size is too small for the requested capacity");
    assert(mem_block->type == MemoryType::HOST &&
           "RegMemBlock must be HOST memory type");
    assert(mem_block->addr != nullptr &&
           "RegMemBlock addr pointer must not be null");

    buffer_ = static_cast<T*>(mem_block->addr);
  }

  ~RingBuffer() {
    if (owns_buffer_ && buffer_) {
      delete[] buffer_;
    }
  }

  // Disable copy
  RingBuffer(RingBuffer const&) = delete;
  RingBuffer& operator=(RingBuffer const&) = delete;

  // Enable move
  RingBuffer(RingBuffer&& other) noexcept
      : buffer_(other.buffer_),
        owns_buffer_(other.owns_buffer_),
        mem_block_(std::move(other.mem_block_)),
        read_ptr(other.read_ptr.load(std::memory_order_relaxed)),
        write_ptr(other.write_ptr.load(std::memory_order_relaxed)) {
    other.buffer_ = nullptr;
    other.owns_buffer_ = false;
  }

  RingBuffer& operator=(RingBuffer&& other) noexcept {
    if (this != &other) {
      if (owns_buffer_ && buffer_) {
        delete[] buffer_;
      }
      buffer_ = other.buffer_;
      owns_buffer_ = other.owns_buffer_;
      mem_block_ = std::move(other.mem_block_);
      read_ptr.store(other.read_ptr.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
      write_ptr.store(other.write_ptr.load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
      other.buffer_ = nullptr;
      other.owns_buffer_ = false;
    }
    return *this;
  }

  int push(T const& item) {
    size_t current_write = write_ptr.load(std::memory_order_relaxed);
    size_t next_write = (current_write + 1) & (Capacity - 1);
    if (next_write == read_ptr.load(std::memory_order_acquire)) return -1;
    buffer_[current_write] = item;
    write_ptr.store(next_write, std::memory_order_release);
    return current_write;
  }

  bool pop(T& item) {
    size_t current_read = read_ptr.load(std::memory_order_relaxed);
    if (current_read == write_ptr.load(std::memory_order_acquire)) return false;
    item = buffer_[current_read];
    read_ptr.store((current_read + 1) & (Capacity - 1),
                   std::memory_order_release);
    return true;
  }

  // Push with type conversion: converts U to T using converter function
  template <typename U, typename ConvertFunc>
  int push_with_convert(U const& item, ConvertFunc&& converter) {
    size_t current_write = write_ptr.load(std::memory_order_relaxed);
    size_t next_write = (current_write + 1) & (Capacity - 1);
    if (next_write == read_ptr.load(std::memory_order_acquire)) return -1;
    converter(item, buffer_[current_write]);
    write_ptr.store(next_write, std::memory_order_release);
    return current_write;
  }

  // Pop with type conversion: converts T to U using converter function
  // Returns the index of popped item on success, -1 on failure
  template <typename U, typename ConvertFunc>
  int pop_with_convert(U& item, ConvertFunc&& converter) {
    size_t current_read = read_ptr.load(std::memory_order_relaxed);
    if (current_read == write_ptr.load(std::memory_order_acquire)) return -1;
    converter(buffer_[current_read], item);
    read_ptr.store((current_read + 1) & (Capacity - 1),
                   std::memory_order_release);
    return current_read;
  }

  // Modify element at given index using a modifier function
  // Returns true if modification was successful, false if index is out of range
  template <typename ModifyFunc>
  bool modify_at(size_t index, ModifyFunc&& modifier) {
    size_t current_read = read_ptr.load(std::memory_order_acquire);
    size_t current_write = write_ptr.load(std::memory_order_acquire);

    // Check if index is within valid range [current_read, current_write)
    size_t available = (current_write - current_read) & (Capacity - 1);
    if (index >= Capacity || available == 0) {
      return false;
    }

    // Calculate the actual position in buffer
    size_t pos = index & (Capacity - 1);

    // Verify that pos is within the valid range
    // This checks if pos is between read_ptr and write_ptr (circular)
    size_t offset = (pos - current_read) & (Capacity - 1);
    if (offset >= available) {
      return false;
    }

    modifier(buffer_[pos]);
    return true;
  }

  // Remove elements from read_ptr while predicate returns true
  // Returns the number of elements removed
  template <typename Predicate>
  size_t remove_while(Predicate&& should_remove) {
    size_t current_read = read_ptr.load(std::memory_order_relaxed);
    size_t current_write = write_ptr.load(std::memory_order_acquire);
    size_t removed = 0;

    while (current_read != current_write) {
      if (!should_remove(buffer_[current_read])) {
        break;
      }
      current_read = (current_read + 1) & (Capacity - 1);
      ++removed;
    }

    if (removed > 0) {
      read_ptr.store(current_read, std::memory_order_release);
    }

    return removed;
  }

  // Modify element at index and advance write_ptr if applicable
  // Returns the number of positions write_ptr was advanced (0 if index !=
  // write_ptr)
  template <typename CheckFunc, typename ModifyFunc>
  size_t modify_and_advance_write(size_t index, CheckFunc&& check,
                                  ModifyFunc&& modifier) {
    size_t pos = index & (Capacity - 1);

    // Modify the element at the specified position
    modifier(buffer_[pos]);

    // Only advance write_ptr if index is the current write position
    size_t current_write = write_ptr.load(std::memory_order_relaxed);
    if (pos != current_write) {
      return 0;
    }

    size_t current_read = read_ptr.load(std::memory_order_acquire);
    size_t advanced = 0;

    // Advance write_ptr while check function returns true
    while (true) {
      size_t next_write = (current_write + 1) & (Capacity - 1);
      // Check buffer full condition
      if (next_write == current_read) {
        break;
      }

      if (!check(buffer_[current_write])) {
        break;
      }

      current_write = next_write;
      advanced++;
    }

    if (advanced > 0) {
      write_ptr.store(current_write, std::memory_order_release);
    }

    return advanced;
  }

  // Get the size of element type T
  constexpr size_t elementSize() const { return sizeof(T); }

  // Get the address of element at given index as uint64_t
  uint64_t getElementAddress(size_t index) const {
    size_t actual_index = index & (Capacity - 1);
    return reinterpret_cast<uint64_t>(&buffer_[actual_index]);
  }

  // Check element at given index using a check function
  template <typename CheckFunc>
  bool check_at(size_t index, CheckFunc&& check) const {
    size_t pos = index & (Capacity - 1);
    return check(buffer_[pos]);
  }

  // Check if the ring buffer is empty
  bool empty() const {
    return read_ptr.load(std::memory_order_acquire) ==
           write_ptr.load(std::memory_order_acquire);
  }

 private:
  T* buffer_;
  bool owns_buffer_;
  std::shared_ptr<RegMemBlock> mem_block_;  // Keep memory block alive
  alignas(64) std::atomic<size_t> read_ptr{0};
  alignas(64) std::atomic<size_t> write_ptr{0};
};
