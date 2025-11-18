/**
 * jring Example - Demonstrates usage of the jring ring buffer
 *
 * This example shows:
 * 1. Single-producer single-consumer (SPSC) usage
 * 2. Multi-producer multi-consumer (MPMC) usage
 * 3. Bulk enqueue/dequeue operations
 * 4. Burst enqueue/dequeue operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#include "../../include/util/jring.h"

// Example data structure to store in the ring
struct Message {
    uint32_t id;
    uint32_t data;
    char payload[8];
};

/**
 * Example 1: Single-Producer Single-Consumer (SPSC) Ring Buffer
 */
void example_spsc() {
    printf("\n=== Example 1: Single-Producer Single-Consumer ===\n");

    const uint32_t ring_size = 16;  // Must be power of 2
    const uint32_t element_size = sizeof(Message);

    // Allocate memory for the ring
    size_t ring_mem_size = jring_get_buf_ring_size(element_size, ring_size);
    struct jring* ring = (struct jring*)aligned_alloc(CACHE_LINE_SIZE, ring_mem_size);
    if (!ring) {
        printf("Failed to allocate ring\n");
        return;
    }

    // Initialize the ring (single-producer=1, single-consumer=1)
    int ret = jring_init(ring, ring_size, element_size, 0, 0);
    if (ret != 0) {
        printf("Failed to initialize ring\n");
        free(ring);
        return;
    }

    printf("Ring initialized: size=%u, capacity=%u, element_size=%u\n",
           ring->size, ring->capacity, ring->esize);

    // Enqueue some messages
    Message msgs[5];
    for (int i = 0; i < 5; i++) {
        msgs[i].id = i;
        msgs[i].data = i * 100;
        snprintf(msgs[i].payload, sizeof(msgs[i].payload), "msg_%d", i);
    }

    unsigned int free_space = 0;
    unsigned int enqueued = jring_sp_enqueue_bulk(ring, msgs, 5, &free_space);
    printf("Enqueued %u messages, free_space=%u\n", enqueued, free_space);
    printf("Ring count: %u, free count: %u, empty: %d, full: %d\n",
           jring_count(ring), jring_free_count(ring), jring_empty(ring), jring_full(ring));

    // Dequeue messages
    Message recv_msgs[5];
    unsigned int available = 0;
    unsigned int dequeued = jring_sc_dequeue_bulk(ring, recv_msgs, 3, &available);
    printf("Dequeued %u messages, available=%u\n", dequeued, available);

    for (unsigned int i = 0; i < dequeued; i++) {
        printf("  Message[%u]: id=%u, data=%u, payload=%s\n",
               i, recv_msgs[i].id, recv_msgs[i].data, recv_msgs[i].payload);
    }

    printf("Ring count after dequeue: %u\n", jring_count(ring));

    free(ring);
}

/**
 * Example 2: Burst Operations (enqueue/dequeue as many as possible)
 */
void example_burst_operations() {
    printf("\n=== Example 2: Burst Operations ===\n");

    const uint32_t ring_size = 8;
    const uint32_t element_size = sizeof(uint64_t);

    size_t ring_mem_size = jring_get_buf_ring_size(element_size, ring_size);
    struct jring* ring = (struct jring*)aligned_alloc(CACHE_LINE_SIZE, ring_mem_size);
    jring_init(ring, ring_size, element_size, 0, 0);

    // Try to enqueue more items than the ring can hold
    uint64_t data[20];
    for (int i = 0; i < 20; i++) {
        data[i] = i * 11;
    }

    unsigned int enqueued = jring_sp_enqueue_burst(ring, data, 20, NULL);
    printf("Tried to enqueue 20 items, actually enqueued: %u (capacity: %u)\n",
           enqueued, ring->capacity);

    // Dequeue in smaller bursts
    uint64_t recv_data[5];
    unsigned int total_dequeued = 0;

    while (!jring_empty(ring)) {
        unsigned int n = jring_sc_dequeue_burst(ring, recv_data, 5, NULL);
        printf("Dequeued burst of %u items: ", n);
        for (unsigned int i = 0; i < n; i++) {
            printf("%lu ", recv_data[i]);
        }
        printf("\n");
        total_dequeued += n;
    }

    printf("Total dequeued: %u\n", total_dequeued);

    free(ring);
}

/**
 * Example 3: Multi-Producer Multi-Consumer (MPMC) with threads
 */
struct ThreadContext {
    struct jring* ring;
    int thread_id;
    int num_items;
};

// Use a properly sized element (16 bytes) to avoid alignment warnings
struct MPMCElement {
    uint32_t value;
    uint32_t padding[3];  // Pad to 16 bytes for optimal performance
};

void* producer_thread(void* arg) {
    ThreadContext* ctx = (ThreadContext*)arg;

    for (int i = 0; i < ctx->num_items; i++) {
        MPMCElement elem;
        elem.value = ctx->thread_id * 1000 + i;
        unsigned int enqueued = 0;

        // Keep trying until enqueued
        while (enqueued == 0) {
            enqueued = jring_mp_enqueue_bulk(ctx->ring, &elem, 1, NULL);
            if (enqueued == 0) {
                usleep(10);  // Ring is full, wait a bit
            }
        }

        printf("[Producer %d] Enqueued: %u\n", ctx->thread_id, elem.value);
        usleep(1000);  // Simulate some work
    }

    return NULL;
}

void* consumer_thread(void* arg) {
    ThreadContext* ctx = (ThreadContext*)arg;
    int consumed = 0;

    while (consumed < ctx->num_items) {
        MPMCElement elem;
        unsigned int dequeued = jring_mc_dequeue_bulk(ctx->ring, &elem, 1, NULL);

        if (dequeued > 0) {
            printf("[Consumer %d] Dequeued: %u\n", ctx->thread_id, elem.value);
            consumed++;
        } else {
            usleep(10);  // Ring is empty, wait a bit
        }
    }

    return NULL;
}

void example_mpmc() {
    printf("\n=== Example 3: Multi-Producer Multi-Consumer ===\n");

    const uint32_t ring_size = 16;
    const uint32_t element_size = sizeof(MPMCElement);

    size_t ring_mem_size = jring_get_buf_ring_size(element_size, ring_size);
    struct jring* ring = (struct jring*)aligned_alloc(CACHE_LINE_SIZE, ring_mem_size);

    // Initialize as multi-producer (mp=1), multi-consumer (mc=1)
    jring_init(ring, ring_size, element_size, 1, 1);

    const int num_producers = 2;
    const int num_consumers = 2;
    const int items_per_thread = 5;

    pthread_t producers[num_producers];
    pthread_t consumers[num_consumers];
    ThreadContext prod_contexts[num_producers];
    ThreadContext cons_contexts[num_consumers];

    // Create producer threads
    for (int i = 0; i < num_producers; i++) {
        prod_contexts[i].ring = ring;
        prod_contexts[i].thread_id = i;
        prod_contexts[i].num_items = items_per_thread;
        pthread_create(&producers[i], NULL, producer_thread, &prod_contexts[i]);
    }

    // Create consumer threads
    for (int i = 0; i < num_consumers; i++) {
        cons_contexts[i].ring = ring;
        cons_contexts[i].thread_id = i;
        cons_contexts[i].num_items = items_per_thread;
        pthread_create(&consumers[i], NULL, consumer_thread, &cons_contexts[i]);
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_producers; i++) {
        pthread_join(producers[i], NULL);
    }
    for (int i = 0; i < num_consumers; i++) {
        pthread_join(consumers[i], NULL);
    }

    printf("All threads completed. Final ring count: %u\n", jring_count(ring));

    free(ring);
}

int main() {
    printf("=================================================\n");
    printf("           jring Ring Buffer Examples            \n");
    printf("=================================================\n");

    example_spsc();
    example_burst_operations();
    example_mpmc();

    printf("\n=================================================\n");
    printf("           All examples completed!               \n");
    printf("=================================================\n");

    return 0;
}
