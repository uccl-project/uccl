
#pragma once

#include "util/gpu_rt.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#define SHM_ITEM_NAME_MAX_LEN 64
#define SHM_ITEM_COUNT 1024
#define IPC_SHM_PATH "p2p_ipc"

struct IPCMemHandle {
    gpuIpcMemHandle_t handle;
    uintptr_t offset;
    size_t size;
};

// for engine
IPCMemHandle get_ipc_by_name_once(std::string name);
IPCMemHandle get_ipc_by_name_blocking(std::string name);

// for P2PTensor
bool reg_ipc_with_name(void* ptr, size_t size, std::string name);
bool dereg_ipc_with_name(std::string name);
// for test
int check_ipc_by_name_once(std::string name);
