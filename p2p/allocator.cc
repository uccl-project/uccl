#include "allocator.h"
#include "engine.h"
#include <atomic>
#include <iomanip>

std::atomic<void*> g_ipc_shm_ptr_{nullptr};

#define kIpcAlignment (1ul << 20)

struct ShmHeader {
    std::atomic<uint64_t> refcount;
    std::atomic<uint64_t> itemcount;
};

struct IPCEntry {
    std::atomic<int> used;     // 0=empty, 1=used
    char name[SHM_ITEM_NAME_MAX_LEN];   // key
    IPCMemHandle handle;       // value
};

constexpr size_t ipc_table_size_bytes() {
    return SHM_ITEM_COUNT * sizeof(IPCEntry);
}

// open or attach a shm mem
void* shm_open_or_attach(const char* shm_file_name, size_t data_size) {
    char shm_name[256];
    snprintf(shm_name, sizeof(shm_name), "/%s", shm_file_name);

    size_t total_size = sizeof(struct ShmHeader) + data_size;

    int fd;
    bool created = false;

    // std::cout << "[shm_open_or_attach] try create shm '" << shm_name
    //           << "' size=" << total_size << std::endl;

    fd = shm_open(shm_name, O_CREAT | O_EXCL | O_RDWR, 0666);
    if (fd != -1) {
        // std::cout << "[shm_open_or_attach] created new shm" << std::endl;
        if (ftruncate(fd, total_size) == -1) {
            std::cerr << "[shm_open_or_attach] ftruncate failed, errno=" << errno << std::endl;
            close(fd);
            shm_unlink(shm_name);
            return nullptr;
        }
        created = true;
    } else {
        if (errno == EEXIST) {
            // std::cout << "[shm_open_or_attach] shm exists, attaching" << std::endl;
            fd = shm_open(shm_name, O_RDWR, 0666);
            if (fd == -1) {
                std::cerr << "[shm_open_or_attach] attach failed, errno=" << errno << std::endl;
                return nullptr;
            }
        } else {
            std::cerr << "[shm_open_or_attach] create failed, errno=" << errno << std::endl;
            return nullptr;
        }
    }

    void* addr = mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (addr == MAP_FAILED) {
        std::cerr << "[shm_open_or_attach] mmap failed, errno=" << errno << std::endl;
        if (created) {
            shm_unlink(shm_name);
        }
        return nullptr;
    }

    if (created) {
        // std::cout << "[shm_open_or_attach] zero init shm region" << std::endl;
        memset(addr, 0, total_size);
    }

    auto* shmheader = static_cast<ShmHeader*>(addr);
    if (created) {
        shmheader->refcount.store(0, std::memory_order_relaxed);
        shmheader->itemcount.store(0, std::memory_order_relaxed);
        // std::cout << "[shm_open_or_attach] header atomics initialized" << std::endl;
    }

    uint64_t old_count = shmheader->refcount.fetch_add(1, std::memory_order_acq_rel);
    // std::cout << "[shm_open_or_attach] attached, refcount was "
    //           << old_count << ", now " << (old_count + 1) << std::endl;

    return addr;
}

ShmHeader* shm_get_shm_header(void* shm_ptr) {
    if (!shm_ptr) {
        // std::cout << "[shm_get_shm_header] null shm_ptr" << std::endl;
        return nullptr;
    }
    return reinterpret_cast<ShmHeader*>(shm_ptr);
}

IPCEntry* shm_get_table(void* shm_ptr) {
    if (!shm_ptr) {
        // std::cout << "[shm_get_table] null shm_ptr" << std::endl;
        return nullptr;
    }
    return reinterpret_cast<IPCEntry*>((char*)shm_ptr + sizeof(struct ShmHeader));
}

bool shm_detach_with_name(void* shm_ptr, size_t data_size, const char* shm_file_name) {
    if (!shm_ptr) {
        // std::cout << "[shm_detach_with_name] null shm_ptr" << std::endl;
        return false;
    }

    size_t total_size = sizeof(struct ShmHeader) + data_size;
    auto* header = shm_get_shm_header(shm_ptr);

    uint64_t old_count = header->refcount.fetch_sub(1, std::memory_order_acq_rel);
    // std::cout << "[shm_detach_with_name] detach from '" << shm_file_name
    //           << "', refcount now " << (old_count - 1) << std::endl;

    void* expected = shm_ptr;
    bool cleared = g_ipc_shm_ptr_.compare_exchange_strong(
        expected, nullptr, std::memory_order_acq_rel, std::memory_order_acquire);
    // std::cout << "[shm_detach_with_name] cleared global ptr? " << (cleared ? "yes" : "no") << std::endl;

    if (munmap(shm_ptr, total_size) != 0) {
        std::cerr << "[shm_detach_with_name] munmap failed, errno=" << errno << std::endl;
    } 

    if (old_count == 1) {
        char shm_name[256];
        snprintf(shm_name, sizeof(shm_name), "/%s", shm_file_name);
        shm_unlink(shm_name);
        // std::cout << "[shm_detach_with_name] shm destroyed: " << shm_name << std::endl;
    }
    return true;
}

void* check_and_get_g_ipc_shm_ptr() {
    void* cur = g_ipc_shm_ptr_.load(std::memory_order_acquire);
    if (!cur) {
        size_t table_size = ipc_table_size_bytes();
        void* addr = shm_open_or_attach(IPC_SHM_PATH, table_size);
        if (!addr) return nullptr;

        void* expected = nullptr;
        bool installed = g_ipc_shm_ptr_.compare_exchange_strong(
            expected, addr,
            std::memory_order_acq_rel, std::memory_order_acquire);

        if (!installed) {
            // somebody else installed a mapping: if it's _not_ the same addr we created, release ours
            if (expected != addr) {
                munmap(addr, table_size);
            }
            cur = expected; // use the other thread's mapping
        } else {
            cur = addr;
        }
    }
    return cur;
}

bool reg_ipc_with_name(void* ptr, size_t size, const std::string name) {
    void* cur = check_and_get_g_ipc_shm_ptr();
    if (!cur) {
        // std::cout << "[reg_ipc_with_name] shm_ptr=null, fail. key=" << name << std::endl;
        return false;
    }

    // prepare handle
    IPCMemHandle handle{};
    handle.size = size;
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t ptr_aligned = ptr_val & ~(kIpcAlignment - 1);
    handle.offset = ptr_val - ptr_aligned;
    GPU_RT_CHECK(gpuIpcGetMemHandle(&handle.handle,
                                    reinterpret_cast<void*>(ptr_aligned)));

    ShmHeader* header = shm_get_shm_header(cur);
    IPCEntry* table = shm_get_table(cur);

    size_t h = std::hash<std::string>{}(name) % SHM_ITEM_COUNT;

    for (size_t i = 0; i < SHM_ITEM_COUNT; ++i) {
        size_t idx = (h + i) % SHM_ITEM_COUNT;

        int used_val = table[idx].used.load(std::memory_order_acquire);

        // case 1: already in table with same key -> overwrite
        if (used_val == 1 &&
            strncmp(table[idx].name, name.c_str(), SHM_ITEM_NAME_MAX_LEN) == 0) {
            table[idx].handle = handle;

            // We increment itemcount when overwriting, since P2PTensor will call dereg as well.
            uint64_t new_count =
                header->itemcount.fetch_add(1, std::memory_order_acq_rel) + 1;

            // std::cout << "[reg_ipc_with_name] overwrite key=" << name
            //           << " idx=" << idx
            //           << " size=" << size
            //           << " ptr=0x" << std::hex << ptr_val << std::dec
            //           << std::endl;
            return true;
        }

        // case 2: empty slot -> insert new
        int expected = 0;
        if (table[idx].used.compare_exchange_strong(
                expected, 2,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            // reserved -> fill
            memset(table[idx].name, 0, SHM_ITEM_NAME_MAX_LEN);
            strncpy(table[idx].name, name.c_str(), SHM_ITEM_NAME_MAX_LEN - 1);
            table[idx].handle = handle;

            // publish
            table[idx].used.store(1, std::memory_order_release);

            // increment itemcount
            uint64_t new_count =
                header->itemcount.fetch_add(1, std::memory_order_acq_rel) + 1;

            // std::cout << "[reg_ipc_with_name] success key=" << name
            //           << " idx=" << idx
            //           << " size=" << size
            //           << " ptr=0x" << std::hex << ptr_val << std::dec
            //           << " itemcount=" << new_count
            //           << std::endl;
            return true;
        }
    }

    // std::cout << "[reg_ipc_with_name] failed to insert, table full? key=" << name
    //           << std::endl;
    return false;
}

bool dereg_ipc_with_name(const std::string name) {
    void* cur = check_and_get_g_ipc_shm_ptr();
    if (!cur) {
        // std::cout << "[dereg_ipc_with_name] shm_ptr=null, fail. key=" << name << std::endl;
        return false;
    }

    IPCEntry* table = shm_get_table(cur);
    ShmHeader* header = shm_get_shm_header(cur);

    size_t h = std::hash<std::string>{}(name) % SHM_ITEM_COUNT;
    for (size_t i = 0; i < SHM_ITEM_COUNT; ++i) {
        size_t idx = (h + i) % SHM_ITEM_COUNT;
        if (table[idx].used.load(std::memory_order_acquire) == 1 &&
            strncmp(table[idx].name, name.c_str(), SHM_ITEM_NAME_MAX_LEN) == 0) {
            // mark empty
            table[idx].used.store(0, std::memory_order_release);
            uint64_t old_id = header->itemcount.fetch_sub(1, std::memory_order_acq_rel);
            // std::cout << "[dereg_ipc_with_name] success key=" << name
            //           << " idx=" << idx
            //           << " old_itemcount=" << old_id
            //           << " new_itemcount=" << (old_id - 1)
            //           << std::endl;
            if (old_id == 1) {
                // last item removed -> detach
                shm_detach_with_name(cur, ipc_table_size_bytes(), IPC_SHM_PATH);
            }
            return true;
        }
    }
    // std::cout << "[dereg_ipc_with_name] not found key=" << name << std::endl;
    // We should decrease item_count even if we didn't find the key.
    uint64_t old_id = header->itemcount.fetch_sub(1, std::memory_order_acq_rel);
    if (old_id == 1) {
        // last item removed -> detach
        shm_detach_with_name(cur, ipc_table_size_bytes(), IPC_SHM_PATH);
    }
    return false;
}

IPCMemHandle get_ipc_by_name_once(const std::string name) {
    IPCMemHandle null_handle{};
    void* cur = check_and_get_g_ipc_shm_ptr();
    if (!cur) return null_handle;

    IPCEntry* table = shm_get_table(cur);
    size_t h = std::hash<std::string>{}(name) % SHM_ITEM_COUNT;
    for (size_t i = 0; i < SHM_ITEM_COUNT; ++i) {
        size_t idx = (h + i) % SHM_ITEM_COUNT;
        if (table[idx].used.load(std::memory_order_acquire) == 1 &&
            strncmp(table[idx].name, name.c_str(), SHM_ITEM_NAME_MAX_LEN) == 0) {
            return table[idx].handle;
        }
    }
    return null_handle;
}

IPCMemHandle get_ipc_by_name_blocking(const std::string name) {
    IPCMemHandle null_handle{};
    void* cur = check_and_get_g_ipc_shm_ptr();
    if (!cur) return null_handle;
    while (true) {
        IPCEntry* table = shm_get_table(cur);
        size_t h = std::hash<std::string>{}(name) % SHM_ITEM_COUNT;
        for (size_t i = 0; i < SHM_ITEM_COUNT; ++i) {
            size_t idx = (h + i) % SHM_ITEM_COUNT;
            if (table[idx].used.load(std::memory_order_acquire) == 1 &&
                strncmp(table[idx].name, name.c_str(), SHM_ITEM_NAME_MAX_LEN) == 0) {
                return table[idx].handle;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// 1 g_ipc error, 2 success, 3 not found
int check_ipc_by_name_once(const std::string name) {
    void* cur = check_and_get_g_ipc_shm_ptr();
    if (!cur) return 1;
    IPCEntry* table = shm_get_table(cur);
    size_t h = std::hash<std::string>{}(name) % SHM_ITEM_COUNT;
    for (size_t i = 0; i < SHM_ITEM_COUNT; ++i) {
        size_t idx = (h + i) % SHM_ITEM_COUNT;
        if (table[idx].used.load(std::memory_order_acquire) == 1 &&
            strncmp(table[idx].name, name.c_str(), SHM_ITEM_NAME_MAX_LEN) == 0) {
            return 2;
        }
    }
    return 3;
}