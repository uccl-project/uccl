# Quick Start Guide

## What Was Fixed

1. **Compilation Errors** - Removed duplicate function declarations in headers
2. **Library Name** - Changed `-lefadv` to `-lefa`
3. **Thread Safety** - Implemented thread-safe singleton for `RdmaDeviceManager`
4. **Code Quality** - Fixed warnings and added error checking

## Build & Run

```bash
# Build
./build.sh

# Or manually:
g++ -std=c++17 -Wall -I. main.cpp -libverbs -lefa -lcuda -o efa_refactor

# Run
./efa_refactor
```

## Key Changes

### RdmaDeviceManager - Now Thread-Safe

```cpp
auto& mgr = RdmaDeviceManager::instance();  // Thread-safe singleton
mgr.initialize();  // Idempotent, can call multiple times
auto dev = mgr.getDevice(0);  // Thread-safe access
```

**Features:**
- ✅ Thread-safe initialization (C++11 magic statics)
- ✅ Mutex-protected access to devices
- ✅ Deleted copy/move constructors
- ✅ Idempotent initialize method
- ✅ RAII with smart pointers

### Thread Safety Example

```cpp
// Safe to use from multiple threads
std::thread t1([]() {
    auto& mgr = RdmaDeviceManager::instance();
    mgr.initialize();
    auto dev = mgr.getDevice(0);
});

std::thread t2([]() {
    auto& mgr = RdmaDeviceManager::instance();  // Same instance
    mgr.initialize();  // Safe, only runs once
    auto dev = mgr.getDevice(0);  // Thread-safe
});
```

## Files Structure

```
rdma_device.h       → RdmaDevice + RdmaDeviceManager (thread-safe singleton)
rdma_context.h      → RdmaContext
main.cpp            → Example usage
build.sh            → Build script
REFACTOR_CHANGES.md → Detailed documentation
QUICK_START.md      → This file
```

## Before vs After

### Before (Broken)
```cpp
class RdmaDeviceManager {
public:
    void initialize();  // Declaration
    void initialize() { ... }  // Definition - ERROR!

    // Not thread-safe
private:
    std::vector<...> devices_;  // No mutex
};
```

### After (Fixed)
```cpp
class RdmaDeviceManager {
public:
    void initialize() {
        std::lock_guard<std::mutex> lock(mutex_);  // Thread-safe
        if (initialized_) return;  // Idempotent
        // ... initialization ...
        initialized_ = true;
    }

    // Deleted copy/move - singleton guarantee
    RdmaDeviceManager(const RdmaDeviceManager&) = delete;

private:
    mutable std::mutex mutex_;  // Thread safety
    bool initialized_;
    std::vector<...> devices_;
};
```

## Compilation Command

### Correct ✅
```bash
g++ -std=c++17 -Wall -I. main.cpp -libverbs -lefa -lcuda -o efa_refactor
```

### Wrong ❌
```bash
# Missing .cpp files that don't exist
g++ main.cpp rdma_device.cpp rdma_device_manager.cpp rdma_context.cpp ...

# Wrong library name
g++ ... -lefadv ...  # Should be -lefa
```

## No Separate .cpp Files Needed

All implementations are inline in headers:
- `rdma_device.h` - Complete implementation
- `rdma_context.h` - Complete implementation
- `main.cpp` - Just the main function

## Verification

```bash
# Build
./build.sh

# Check binary
ls -lh efa_refactor

# Run (requires RDMA device)
./efa_refactor
```

Expected output:
```
Found N RDMA devices:
  [0] mlx5_0
GID[0]: 192.168.1.100
Context and PD initialized successfully.
Device count: N
```

## Documentation

- **REFACTOR_CHANGES.md** - Complete technical details
- **QUICK_START.md** - This file
- **build.sh** - Automated build script

## Need Help?

Check the detailed documentation:
```bash
cat REFACTOR_CHANGES.md
```
