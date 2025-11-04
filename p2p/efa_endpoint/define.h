#pragma once
#include <infiniband/verbs.h>
#include <cstdint>
#include <infiniband/verbs.h>
#include <infiniband/efadv.h>
#include <memory>
#include <cstring>
#include <iostream>
#include <cassert>

struct metadata {
    uint32_t qpn;
    union ibv_gid gid;
    uint32_t rkey;
    uint64_t addr;
};

struct EFASendRequest {
    uint64_t addr;
    uint32_t rkey;
    size_t length;
    uint32_t imm_data;  // immediate data
};

struct EFARecvRequest {
    uint64_t addr;
    uint32_t rkey;
    size_t length;
};

