#ifndef __SG_COPY_IF__
#define __SG_COPY_IF__

#include <atomic>
#include <cassert>
#include <chrono>
#include <tuple>
#include <vector>

#define cudaCheckErrors(msg)                                        \
    do {                                                            \
        cudaError_t __err = cudaGetLastError();                     \
        if (__err != cudaSuccess) {                                 \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, \
                    cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n");             \
            exit(1);                                                \
        }                                                           \
    } while (0)

static constexpr int kNumThBlocks = 4;
static constexpr int kNumThPerBlock = 512;

// Yang: 64 max scattered IOVs; 256 for MTU 4096
#ifdef MTU_4096
#define kMaxIovs 256
#else
#define kMaxIovs 64
#endif

// Use shared memory for IOV when kMaxIovs is <= 64, otherwise use HBM pointer to avoid shared memory overflow
#if kMaxIovs <= 64
#define USE_SHARED_MEMORY
#endif

// proxy.h: NCCL_PROXY_MAX_SUBS = 32, NCCL_STEPS = 8; double provisioning
static constexpr int kFifoCap = 32 * 8 / kNumThBlocks * 2;
static constexpr uint64_t kAbortTailValue = (uint64_t)-2;

struct alignas(8) Iov {
    void *srcs[kMaxIovs];
    void *dsts[kMaxIovs];
    int lens[kMaxIovs];
    int iov_n;
    int gpu_idx;  // for debugging
    int step;     // for debugging
};

struct IovFifo {
    uint64_t head;  // GPU writes finished index
    uint64_t tail;  // CPU posts working index
    struct Iov iovs[kFifoCap];
};

class iovMultiFifo {
    static constexpr int kMaxFifos = 32;

    int num_fifo_;
    volatile struct IovFifo *fifo_vec_[kMaxFifos];
    std::atomic<uint64_t>
        fifo_slot_idx_[kMaxFifos];  // monotonic increasing index
    cudaStream_t sg_stream;

   public:
    iovMultiFifo(int num_fifo) : num_fifo_(num_fifo) {
        for (int i = 0; i < num_fifo; i++) {
            cudaHostAlloc(&fifo_vec_[i], sizeof(struct IovFifo),
                          cudaHostAllocMapped);
            cudaCheckErrors("cudaMallocManaged failed");
        }
        initFifo();

        cudaStreamCreate(&sg_stream);
        cudaCheckErrors("cudaStreamCreate failed");
    }

    // Init all fifo head, tail, iov_n, and slot_idx.
    void initFifo() {
        for (int i = 0; i < num_fifo_; i++) {
            // Initialize the fifo
            volatile struct IovFifo *fifo = fifo_vec_[i];
            fifo->head = (uint64_t)-1;
            fifo->tail = (uint64_t)-1;
            for (int j = 0; j < kFifoCap; j++) {
                fifo->iovs[j].iov_n = -1;
            }
            __sync_synchronize();

            fifo_slot_idx_[i] = 0;
        }
    }

    void launchSGCopyKernel();

    ~iovMultiFifo() {
        for (int i = 0; i < num_fifo_; i++) {
            cudaFreeHost((void *)fifo_vec_[i]);
            cudaCheckErrors("cudaFreeHost failed");
        }
    }

    struct IovFifo **get_fifo_vec() { return (struct IovFifo **)fifo_vec_; }

    std::tuple<uint64_t, volatile struct Iov *> reserve_fifo_slot(
        int fifo_idx) {
        auto slot_idx = fifo_slot_idx_[fifo_idx]++;
        auto reserved_iov = fifo_vec_[fifo_idx]->iovs + slot_idx % kFifoCap;
        return {slot_idx, reserved_iov};
    }

    void dispatch_task(int fifo_idx) {
        auto slot_idx = fifo_slot_idx_[fifo_idx] - 1;
        fifo_vec_[fifo_idx]->tail = slot_idx;
        __sync_synchronize();
    }

    bool check_completion(int fifo_idx, uint64_t slot_idx) {
        if (slot_idx >= fifo_slot_idx_[fifo_idx]) {
            printf("slot_idx %lu >= fifo_slot_idx_[%d] %lu\n", slot_idx,
                   fifo_idx, fifo_slot_idx_[fifo_idx].load());
        }
        assert(slot_idx < fifo_slot_idx_[fifo_idx]);
        // The intial -1 will always be less than 0.
        return (int64_t)fifo_vec_[fifo_idx]->head >= (int64_t)slot_idx;
    }

    void abort() {
        for (int i = 0; i < num_fifo_; i++) {
            fifo_vec_[i]->tail = kAbortTailValue;
            __sync_synchronize();
        }
    }

    void sync_stream() {
        cudaStreamSynchronize(sg_stream);
        cudaCheckErrors("cudaStreamSynchronize failed");
    }
};

#endif  // __SG_COPY_IF__