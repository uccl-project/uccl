#include "efa_ctrl_channel.h"
#include "memory_allocator.h"
#include "ring_spsc.h"
#include <chrono>
#include <iostream>
#include <thread>

enum class MetaDataFlag : int16_t { PENDING = 0, IN_PROGRESS = 1, IS_DONE = 2 };

struct MetaData {
  int64_t address;
  std::atomic<int16_t> flag;

  MetaData() : address(0), flag{static_cast<int16_t>(MetaDataFlag::PENDING)} {}
  MetaData(int64_t addr, MetaDataFlag f)
      : address(addr), flag{static_cast<int16_t>(f)} {}

  // Copy constructor for RingBuffer operations
  MetaData(MetaData const& other)
      : address(other.address),
        flag{other.flag.load(std::memory_order_relaxed)} {}

  MetaData& operator=(MetaData const& other) {
    address = other.address;
    flag.store(other.flag.load(std::memory_order_relaxed),
               std::memory_order_relaxed);
    return *this;
  }
};

// constexpr size_t RING_CAPACITY = 16;  // Must be power of 2

int main() {
  std::cout << "RingBuffer Test with MetaData and shared_ptr<RegMemBlock>"
            << std::endl;
  std::cout << "Ring capacity: " << RING_CAPACITY << std::endl;
  std::cout << "MetaData size: " << sizeof(MetaData) << " bytes" << std::endl;
  std::cout << std::endl;

  try {
    auto& mgr = RdmaDeviceManager::instance();
    auto dev = mgr.getDevice(0);
    if (!dev) {
      std::cerr << "Failed to get device 0" << std::endl;
      return 1;
    }

    auto ctx = std::make_shared<RdmaContext>(dev);
    MemoryAllocator allocator;

    // Allocate memory for RingBuffer
    size_t buffer_size = RING_CAPACITY * sizeof(MetaData);
    auto mem_block = allocator.allocate(buffer_size, MemoryType::HOST, ctx);

    std::cout << "Allocated " << mem_block->size << " bytes for RingBuffer at "
              << mem_block->addr << std::endl;

    // Create RingBuffer with shared_ptr<RegMemBlock>
    RingBuffer<MetaData, RING_CAPACITY> rb(mem_block);

    // Test push and pop
    std::cout << "\n=== Testing push and pop ===" << std::endl;
    MetaData data1(0x1000, MetaDataFlag::PENDING);
    int idx1 = rb.push(data1);
    std::cout << "Pushed data1 at index: " << idx1 << std::endl;

    MetaData data2(0x2000, MetaDataFlag::IN_PROGRESS);
    int idx2 = rb.push(data2);
    std::cout << "Pushed data2 at index: " << idx2 << std::endl;

    MetaData popped;
    if (rb.pop(popped)) {
      std::cout << "Popped: address=0x" << std::hex << popped.address
                << ", flag=" << std::dec << popped.flag.load() << std::endl;
    }

    // Test modify_at
    std::cout << "\n=== Testing modify_at ===" << std::endl;
    MetaData data3(0x3000, MetaDataFlag::PENDING);
    int idx3 = rb.push(data3);
    std::cout << "Pushed data3 at index: " << idx3 << std::endl;

    bool modified = rb.modify_at(idx3, [](MetaData& item) {
      item.flag.store(static_cast<int16_t>(MetaDataFlag::IS_DONE),
                      std::memory_order_relaxed);
      std::cout << "Modified item, changed flag to IS_DONE" << std::endl;
    });
    std::cout << "Modification result: " << (modified ? "success" : "failed")
              << std::endl;

    // Test remove_while
    std::cout << "\n=== Testing remove_while ===" << std::endl;
    // Clear buffer first
    MetaData temp;
    while (rb.pop(temp)) {
    }

    // Push multiple items with different flags
    rb.push(MetaData(0x4000, MetaDataFlag::IS_DONE));
    rb.push(MetaData(0x5000, MetaDataFlag::IS_DONE));
    rb.push(MetaData(0x6000, MetaDataFlag::IS_DONE));
    rb.push(
        MetaData(0x7000, MetaDataFlag::PENDING));  // This should stop removal
    rb.push(MetaData(0x8000, MetaDataFlag::IS_DONE));

    std::cout << "Pushed 5 items: 3 IS_DONE, 1 PENDING, 1 IS_DONE" << std::endl;

    // Remove all IS_DONE items from the front
    size_t removed = rb.remove_while([](MetaData const& item) {
      return item.flag.load(std::memory_order_relaxed) ==
             static_cast<int16_t>(MetaDataFlag::IS_DONE);
    });

    std::cout << "Removed " << removed
              << " items with IS_DONE flag (expected: 3)" << std::endl;

    // Verify remaining items
    MetaData remaining;
    int count = 0;
    while (rb.pop(remaining)) {
      std::cout << "Remaining item " << count++ << ": address=0x" << std::hex
                << remaining.address << ", flag=" << std::dec
                << remaining.flag.load() << std::endl;
    }
    std::cout << "Total remaining items: " << count << " (expected: 2)"
              << std::endl;

    // // Test push_bulk and pop_bulk
    // std::cout << "\n=== Testing bulk operations ===" << std::endl;
    // MetaData bulk_data[5];
    // for (int i = 0; i < 5; ++i) {
    //     bulk_data[i].address = 0x9000 + i * 0x100;
    //     bulk_data[i].flag.store(static_cast<int16_t>(MetaDataFlag::PENDING),
    //     std::memory_order_relaxed);
    // }

    // size_t pushed = rb.push_bulk(bulk_data, 5);
    // std::cout << "Pushed " << pushed << " items in bulk" << std::endl;

    // MetaData popped_data[3];
    // size_t popped_count = rb.pop_bulk(popped_data, 3);
    // std::cout << "Popped " << popped_count << " items in bulk:" << std::endl;
    // for (size_t i = 0; i < popped_count; ++i) {
    //     std::cout << "  Item " << i << ": address=0x" << std::hex <<
    //     popped_data[i].address
    //               << ", flag=" << std::dec << popped_data[i].flag.load() <<
    //               std::endl;
    // }

    std::cout << "\n=== All tests completed successfully ===" << std::endl;

  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
