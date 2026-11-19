from pathlib import Path
p=Path('src/multinode/device_collectives.cuh');s=p.read_text();a=s.index('template <bool ClientIO = false, bool Packet = false>\n__device__ __forceinline__ void hierarchicalAllGatherSerialBody(');b=s.index('\ntemplate <bool ClientIO = false>\n__global__ void hierarchicalAllGatherSerial(',a);t=s[a:b]
needle='''      for (size_t i = threadIdx.x; i < units; i += blockDim.x)
        target[i] = input[i];'''
new='''      for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
        uint4 value = input[i];
        target[i] = value;
#ifdef UCCL_LITE_AG_FUSED_OWN
#if defined(UCCL_LITE_AG_PARALLEL_FANOUT) || defined(UCCL_LITE_AG_EARLY_OWN)
#error "Fused own-output is a separate serial fan-out experiment"
#endif
        if constexpr (ClientIO && Packet)
          if (nelem == 32768 && step == kLocalRanks - 1)
            storeClientQuad(clientOutput, ownOff / 4 + i, value);
#endif
      }'''
assert needle in t;t=t.replace(needle,new,1)
needle='''      for (size_t i = threadIdx.x; i < units; i += blockDim.x)
        target[i] = received[i];'''
new='''      for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
        uint4 value = received[i];
        target[i] = value;
#ifdef UCCL_LITE_AG_FUSED_OWN
        if constexpr (ClientIO && Packet)
          if (nelem == 32768 && step == kLocalRanks - 1)
            storeClientQuad(clientOutput, remoteOff / 4 + i, value);
#endif
      }'''
assert needle in t;t=t.replace(needle,new,1)
t=t.replace('#ifdef UCCL_LITE_AG_EARLY_OWN\n      if constexpr (Packet)', '#if defined(UCCL_LITE_AG_EARLY_OWN) || defined(UCCL_LITE_AG_FUSED_OWN)\n      if constexpr (Packet)',1)
s=s[:a]+t+s[b:]
a=s.index('template <bool ClientIO = false, bool Packet = false>\n__device__ __forceinline__ void hierarchicalAllReduceSerialBody(');b=s.index('\ntemplate <bool ClientIO = false>\n__global__ void hierarchicalAllReduceSerial(',a);t=s[a:b]
needle='''    __syncthreads();
  }
  if constexpr (ClientIO)'''
replacement='''#ifdef UCCL_LITE_AR_COMPACT_BARRIERS
    if constexpr (ClientIO) {
      if (step == kLocalRanks - 1) __syncthreads();
    } else
#endif
      __syncthreads();
  }
  if constexpr (ClientIO)'''
assert t.count(needle)==2;t=t.replace(needle,replacement)
needle='''  publish();
#ifdef UCCL_LITE_AR_EARLY_FLUSH'''
replacement='''#ifdef UCCL_LITE_AR_COMPACT_BARRIERS
  // This serial body has one CTA. The following PortChannel publication
  // provides system release after every producer reaches this CTA barrier.
  if constexpr (ClientIO)
    __syncthreads();
  else
#endif
    publish();
#ifdef UCCL_LITE_AR_EARLY_FLUSH'''
assert needle in t;t=t.replace(needle,replacement,1)
s=s[:a]+t+s[b:];p.write_text(s)
p=Path('src/multinode/async_host_gather.cuh');s=p.read_text();s=s.replace('#include <chrono>','#include <atomic>\n#include <chrono>',1)
s=s.replace('  bool pending_ = false, stop_ = false, active_ = false;','''  bool pending_ = false, stop_ = false, active_ = false;
#ifdef UCCL_LITE_HOST_SPIN_PROGRESS
  std::atomic<size_t> queuedWork_{0};
  std::atomic<bool> stopRequested_{false};
  static void relaxCpu() {
#if defined(__x86_64__) || defined(__i386__)
    asm volatile("pause");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
  }
#endif''',1)
s=s.replace('''        std::this_thread::yield();''','''#ifdef UCCL_LITE_HOST_SPIN_PROGRESS
        relaxCpu();
#else
        std::this_thread::yield();
#endif''',1)
s=s.replace('''      for (;;) {
        Work work;''','''      for (;;) {
#ifdef UCCL_LITE_HOST_SPIN_PROGRESS
        // A queue doorbell avoids per-call OS wakeup while leaving queue,
        // failure and active-work state protected by the original mutex.
        while (!queuedWork_.load(std::memory_order_acquire) &&
               !stopRequested_.load(std::memory_order_acquire))
          relaxCpu();
#endif
        Work work;''',1)
s=s.replace('''          changed_.wait(lock, [&] { return stop_ || !queue_.empty(); });''','''#ifndef UCCL_LITE_HOST_SPIN_PROGRESS
          changed_.wait(lock, [&] { return stop_ || !queue_.empty(); });
#endif''',1)
s=s.replace('''          queue_.pop_front();
          active_ = true;''','''          queue_.pop_front();
#ifdef UCCL_LITE_HOST_SPIN_PROGRESS
          queuedWork_.fetch_sub(1, std::memory_order_release);
#endif
          active_ = true;''',1)
s=s.replace('''      stop_ = true;''','''      stop_ = true;
#ifdef UCCL_LITE_HOST_SPIN_PROGRESS
      stopRequested_.store(true, std::memory_order_release);
#endif''',1)
s=s.replace('''      queue_.push_back(work);
      changed_.notify_all();''','''      queue_.push_back(work);
#ifdef UCCL_LITE_HOST_SPIN_PROGRESS
      queuedWork_.fetch_add(1, std::memory_order_release);
#endif
      changed_.notify_all();''',1)
p.write_text(s)
