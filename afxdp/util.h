#pragma once

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <cstdint>
#include <vector>

template <class T>
T Percentile(std::vector<T>& vectorIn, double percent) {
    if (vectorIn.size() == 0) return (T)0;
    auto nth = vectorIn.begin() + (percent * vectorIn.size()) / 100;
    std::nth_element(vectorIn.begin(), nth, vectorIn.end());
    return *nth;
}

uint16_t ipv4_checksum(const void* data, size_t header_length) {
    unsigned long sum = 0;

    const uint16_t* p = (const uint16_t*)data;

    while (header_length > 1) {
        sum += *p++;
        if (sum & 0x80000000) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        header_length -= 2;
    }

    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    return ~sum;
}

bool pin_thread_to_cpu(int cpu) {
    int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu < 0 || cpu >= num_cpus) return false;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);

    pthread_t current_thread = pthread_self();

    return !pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

inline void apply_setsockopt(int xsk_fd) {
    int ret;
    int sock_opt;

    sock_opt = 1;

    ret = setsockopt(xsk_fd, SOL_SOCKET, SO_PREFER_BUSY_POLL, (void*)&sock_opt,
                     sizeof(sock_opt));
    if (ret == -EPERM) {
        fprintf(stderr,
                "Ignore SO_PREFER_BUSY_POLL as it failed: this option needs "
                "privileged mode.\n");
    } else if (ret < 0) {
        fprintf(stderr, "Ignore SO_PREFER_BUSY_POLL as it failed\n");
    }

    sock_opt = 20;
    if (setsockopt(xsk_fd, SOL_SOCKET, SO_BUSY_POLL, (void*)&sock_opt,
                   sizeof(sock_opt)) < 0) {
        fprintf(stderr, "Ignore SO_BUSY_POLL as it failed\n");
    }

    sock_opt = 64;
    ret = setsockopt(xsk_fd, SOL_SOCKET, SO_BUSY_POLL_BUDGET, (void*)&sock_opt,
                     sizeof(sock_opt));
    if (ret == -EPERM) {
        fprintf(stderr,
                "Ignore SO_BUSY_POLL_BUDGET as it failed: this option needs "
                "privileged mode.\n");
    } else if (ret < 0) {
        fprintf(stderr, "Ignore SO_BUSY_POLL_BUDGET as it failed\n");
    }
}