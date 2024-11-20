/**
 * @file timer.h
 * @brief Helper functions for timers
 */

#pragma once

#include <stdint.h>
#include <stdlib.h>

#include <chrono>

#include "glog/logging.h"

namespace uccl {

/// Return the TSC
static inline size_t rdtsc() {
    uint64_t rax;
    uint64_t rdx;
    asm volatile("rdtsc" : "=a"(rax), "=d"(rdx));
    return static_cast<size_t>((rdx << 32) | rax);
}

/// An alias for rdtsc() to distinguish calls on the critical path
static const auto &dpath_rdtsc = rdtsc;

static void nano_sleep(size_t ns, double freq_ghz) {
    size_t start = rdtsc();
    size_t end = start;
    size_t upp = static_cast<size_t>(freq_ghz * ns);
    while (end - start < upp) end = rdtsc();
}

/// Simple time that uses std::chrono
class ChronoTimer {
   public:
    ChronoTimer() { reset(); }
    void reset() { start_time_ = std::chrono::high_resolution_clock::now(); }

    /// Return seconds elapsed since this timer was created or last reset
    double get_sec() const { return get_ns() / 1e9; }

    /// Return milliseconds elapsed since this timer was created or last reset
    double get_ms() const { return get_ns() / 1e6; }

    /// Return microseconds elapsed since this timer was created or last reset
    double get_us() const { return get_ns() / 1e3; }

    /// Return nanoseconds elapsed since this timer was created or last reset
    size_t get_ns() const {
        return static_cast<size_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - start_time_)
                .count());
    }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

static double measure_rdtsc_freq() {
    ChronoTimer chrono_timer;
    const uint64_t rdtsc_start = rdtsc();

    // Do not change this loop! The hardcoded value below depends on this loop
    // and prevents it from being optimized out.
    uint64_t sum = 5;
    for (uint64_t i = 0; i < 1000000; i++) {
        sum += i + (sum + i) * (i % sum);
    }
    CHECK(sum == 13580802877818827968ull) << "Error in RDTSC freq measurement";

    const uint64_t rdtsc_cycles = rdtsc() - rdtsc_start;
    const double freq_ghz = rdtsc_cycles * 1.0 / chrono_timer.get_ns();
    CHECK(freq_ghz >= 0.5 && freq_ghz <= 5.0) << "Invalid RDTSC frequency";

    return freq_ghz;
}

/// Convert cycles measured by rdtsc with frequence \p freq_ghz to seconds
static double to_sec(size_t cycles, double freq_ghz) {
    return (cycles / (freq_ghz * 1000000000));
}

/// Convert cycles measured by rdtsc with frequence \p freq_ghz to msec
static double to_msec(size_t cycles, double freq_ghz) {
    return (cycles / (freq_ghz * 1000000));
}

/// Convert cycles measured by rdtsc with frequence \p freq_ghz to usec
static double to_usec(size_t cycles, double freq_ghz) {
    return (cycles / (freq_ghz * 1000));
}

static size_t ms_to_cycles(double ms, double freq_ghz) {
    return static_cast<size_t>(ms * 1000 * 1000 * freq_ghz);
}

static size_t us_to_cycles(double us, double freq_ghz) {
    return static_cast<size_t>(us * 1000 * freq_ghz);
}

static size_t ns_to_cycles(double ns, double freq_ghz) {
    return static_cast<size_t>(ns * freq_ghz);
}

/// Convert cycles measured by rdtsc with frequence \p freq_ghz to nsec
static double to_nsec(size_t cycles, double freq_ghz) {
    return (cycles / freq_ghz);
}

static inline double rdtsc_to_us(uint64_t tsc) {
    static double ghz = 0;
    if (unlikely(!ghz)) ghz = measure_rdtsc_freq();
    return tsc / ghz / 1000.0;
}

/// Simple time that uses RDTSC
class TscTimer {
   public:
    size_t start_tsc_ = 0;
    size_t tsc_sum_ = 0;
    size_t num_calls_ = 0;

    inline void start() { start_tsc_ = rdtsc(); }
    inline void stop() {
        tsc_sum_ += (rdtsc() - start_tsc_);
        num_calls_++;
    }

    void reset() {
        start_tsc_ = 0;
        tsc_sum_ = 0;
        num_calls_ = 0;
    }

    size_t avg_cycles() const { return tsc_sum_ / num_calls_; }
    double avg_sec(double freq_ghz) const {
        return to_sec(avg_cycles(), freq_ghz);
    }

    double avg_usec(double freq_ghz) const {
        return to_usec(avg_cycles(), freq_ghz);
    }

    double avg_nsec(double freq_ghz) const {
        return to_nsec(avg_cycles(), freq_ghz);
    }
};
}  // namespace uccl
