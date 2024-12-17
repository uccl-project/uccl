#include <atomic>
#include <chrono>
#include <cstring>  // For memcpy
#include <iostream>
#include <thread>
#include <vector>

using namespace std;
using namespace std::chrono;

const size_t PAGE_SIZE = 4 * 1024;  // 4 KB memory page size

// Function to perform memcpy on 4KB pages in a single thread
void memory_copy_thread(const char* src, char* dst, size_t total_size,
                        size_t iterations) {
    size_t num_pages = total_size / PAGE_SIZE;  // Total number of pages
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t page = 0; page < num_pages; ++page) {
            memcpy(dst + page * PAGE_SIZE, src + page * PAGE_SIZE, PAGE_SIZE);
        }
    }
}

int main() {
    const size_t size_per_thread = 128 * 1024 * 1024;  // 128 MB per thread
    const size_t num_threads = 4;                      // Number of threads
    const size_t iterations = 100;  // Number of iterations per thread
    const size_t total_size = size_per_thread * num_threads;

    cout
        << "Multi-threaded Memory Copy Bandwidth using 4KB pages with memcpy\n";
    cout << "Number of Threads: " << num_threads << "\n";
    cout << "Total Buffer Size: " << total_size / (1024 * 1024) << " MB\n";

    // Allocate buffers for each thread
    vector<vector<char>> src_buffers(num_threads,
                                     vector<char>(size_per_thread, 1));
    vector<vector<char>> dst_buffers(num_threads,
                                     vector<char>(size_per_thread, 0));

    // Prepare threads
    vector<thread> threads(num_threads);
    atomic<bool> ready(false);  // Synchronize thread start

    // Start threads, wait for signal
    for (size_t t = 0; t < num_threads; ++t) {
        threads[t] = thread([&, t]() {
            while (!ready) {
            }  // Busy-wait until signal to start
            memory_copy_thread(src_buffers[t].data(), dst_buffers[t].data(),
                               size_per_thread, iterations);
        });
    }

    // Measure only memcpy time
    auto start = high_resolution_clock::now();
    ready = true;  // Signal threads to begin work

    for (auto& t : threads) {
        t.join();  // Wait for all threads
    }
    auto end = high_resolution_clock::now();

    // Calculate Time and Bandwidth
    auto time_taken = duration_cast<duration<double>>(end - start).count();
    double total_data_copied =
        iterations * total_size;  // Total bytes copied by all threads
    double bandwidth = (total_data_copied / (1e9 * time_taken));  // GB/s

    // Output Results
    cout << "Total Time (excluding thread creation): " << time_taken
         << " seconds\n";
    cout << "Copy Bandwidth: " << bandwidth << " GB/s\n";

    return 0;
}
