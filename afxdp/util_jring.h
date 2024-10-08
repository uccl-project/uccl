/*-
 * SPDX-License-Identifier: BSD-2-Clause-FreeBSD
 *
 * Copyright (c) 2019 Arm Limited
 * Copyright (c) 2010-2017 Intel Corporation*
 * Copyright (c) 2007-2009 Kip Macy <kmacy@freebsd.org>
 * Copyright (c) 2022 Ilias Marinos <ilias@marinos.io>
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * $FreeBSD$
 *
 */

#pragma once

/**
 * @file
 * Buffer Ring with user defined element size.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>

#define RING_SZ_MASK (unsigned)(0x0fffffff) /**< Ring slots mask */
#define CACHE_LINE_SIZE 64

#define ISPOWEROF2(x) (((((x) - 1) & (x)) == 0) && x)
#define __ROUND_MASK(x, y) ((__typeof__(x))((y) - 1))
#define ALIGN_UP_POW2(x, y) ((((x) - 1) | __ROUND_MASK(x, y)) + 1)
#define ROUND_DOWN_POW2(x, y) ((x) & ~__ROUND_MASK(x, y))

/** enqueue/dequeue behavior types */
enum jring_queue_behavior {
    /** Enq/Deq a fixed number of items from a ring */
    JRING_QUEUE_FIXED = 0,
    /** Enq/Deq as many items as possible from ring */
    JRING_QUEUE_VARIABLE
};

/** prod/cons sync types */
enum jring_sync_type {
    JRING_SYNC_MT = 0, /**< multi-thread safe */
    JRING_SYNC_ST = 1, /**< single-thread */
};

struct jring_headtail {
    volatile uint32_t head;
    volatile uint32_t tail;
    enum jring_sync_type sync;
};

struct jring {
    uint32_t size;     /**< Size of ring. */
    uint32_t mask;     /**< Mask (size-1) of ring. */
    uint32_t capacity; /**< Usable size of ring */
    uint32_t esize;    /**< Size of each element in the ring. */
    uint32_t reserved[2];

    // Producer head/tail.
    struct jring_headtail prod __attribute__((aligned(CACHE_LINE_SIZE)));

    // Empty cache line.
    char pad0 __attribute__((aligned(CACHE_LINE_SIZE)));

    // Consumer head/tail.
    struct jring_headtail cons __attribute__((aligned(CACHE_LINE_SIZE)));

    // Empty cache line.
    char pad1 __attribute__((aligned(CACHE_LINE_SIZE)));
    void *ring[0] __attribute__((aligned(CACHE_LINE_SIZE)));
};
typedef struct jring jring_t;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#include "util_jring_elem_private.h"
#pragma GCC diagnostic pop

/**
 * Calculate the memory size needed for a ring with given element size.
 *
 * This function returns the number of bytes needed for a ring, given
 * the number of elements in it and the size of the element. This value
 * is the sum of the size of the structure buf_ring and the size of the
 * memory needed for storing the elements. The value is aligned to a cache
 * line size.
 *
 * @param element_size
 *   The size of ring element, in bytes. It must be a multiple of 4B.
 * @param count
 *   The number of elements in the ring (must be a power of 2).
 * @return
 *   - The memory size needed for the ring on success.
 *   - (size_t)-1 - element_size is not a multiple of 4 or count
 *      provided is not a power of 2.
 */
static inline size_t jring_get_buf_ring_size(uint32_t element_size,
                                             uint32_t count) {
    if ((element_size % 4) != 0) return -1;

    if (!(ISPOWEROF2(count)) || (count > RING_SZ_MASK)) return -1;

    size_t sz = sizeof(struct jring) + (count * element_size);

    sz = ALIGN_UP_POW2(sz, CACHE_LINE_SIZE);
    return sz;
}

/**
 * Function to initialize a ring.
 *
 * @param r Pointer to the ring structure.
 * @param count The number of elements in the ring (must be a power of 2).
 * @param esize Element size in bytes.
 * @param mp Set to 1 if the ring is to be multi-producer safe.
 * @param mc Set to 1 if the ring is to be multi-consumer safe.
 * @return 0 on success, -1 on failure.
 */
static inline int jring_init(struct jring *r, uint32_t count, uint32_t esize,
                             int mp, int mc) {
    // The buffer ring needs to be a power of two.
    if (!ISPOWEROF2(count)) {
        return -1;
    }
    // Element size needs to be a multiple of 4.
    if (esize % 4 != 0) return -1;

    r->size = count;
    r->mask = r->size - 1;
    r->capacity = r->mask;  // Usable size of the ring.
    r->esize = esize;

    r->prod.head = 0;
    r->prod.tail = 0;
    !!mp ? (r->prod.sync = JRING_SYNC_MT) : (r->prod.sync = JRING_SYNC_ST);
    r->cons.head = 0;
    r->cons.tail = 0;
    !!mc ? (r->cons.sync = JRING_SYNC_MT) : (r->cons.sync = JRING_SYNC_ST);

    return 0;
}

/**
 * Enqueue a specific amount of objects on a ring (Single producer only).
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of void * pointers (objects).
 * @param n
 *   The number of objects to add in the ring from the obj_table.
 * @param free_space
 *   if non-NULL, returns the amount of space in the ring after the
 *   enqueue operation has finished.
 * @return
 *   The number of objects enqueued, either 0 or n
 */
static __attribute__((always_inline)) inline unsigned int jring_sp_enqueue_bulk(
    struct jring *r, const void *obj_table, unsigned int n,
    unsigned int *free_space) {
    return __jring_do_enqueue_elem(r, obj_table, r->esize, n, JRING_QUEUE_FIXED,
                                   JRING_SYNC_ST, free_space);
}

/**
 * Enqueue a specific amount of objects on a ring (multi-producer safe).
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of void * pointers (objects).
 * @param n
 *   The number of objects to add in the ring from the obj_table.
 * @param free_space
 *   if non-NULL, returns the amount of space in the ring after the
 *   enqueue operation has finished.
 * @return
 *   The number of objects enqueued, either 0 or n
 */
static __attribute__((always_inline)) inline unsigned int jring_mp_enqueue_bulk(
    struct jring *r, const void *obj_table, unsigned int n,
    unsigned int *free_space) {
    return __jring_do_enqueue_elem(r, obj_table, r->esize, n, JRING_QUEUE_FIXED,
                                   JRING_SYNC_MT, free_space);
}

/**
 * Enqueue a specific amount of objects on a ring. This function performs
 * a runtime check to find the ring's synchronization type.
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of void * pointers (objects).
 * @param n
 *   The number of objects to add in the ring from the obj_table.
 * @param free_space
 *   if non-NULL, returns the amount of space in the ring after the
 *   enqueue operation has finished.
 * @return
 *   The number of objects enqueued, either 0 or n
 */
static __attribute((always_inline)) inline unsigned int jring_enqueue_bulk(
    struct jring *r, const void *obj_table, unsigned int n,
    unsigned int *free_space) {
    return (r->prod.sync == JRING_SYNC_ST)
               ? jring_sp_enqueue_bulk(r, obj_table, n, free_space)
               : jring_mp_enqueue_bulk(r, obj_table, n, free_space);
}

/**
 * Enqueue up to a specific amount of objects on a ring (Single producer only).
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of void * pointers (objects).
 * @param n
 *   The number of objects to add in the ring from the obj_table.
 * @param free_space
 *   if non-NULL, returns the amount of space in the ring after the
 *   enqueue operation has finished.
 * @return
 *   The number of objects enqueued, ranging in [0,n].
 */
static __attribute__((always_inline)) inline unsigned int
jring_sp_enqueue_burst(struct jring *r, const void *obj_table, unsigned int n,
                       unsigned int *free_space) {
    return __jring_do_enqueue_elem(r, obj_table, r->esize, n,
                                   JRING_QUEUE_VARIABLE, JRING_SYNC_ST,
                                   free_space);
}

/**
 * Enqueue up to a specific amount of objects on a ring (multi-producer safe).
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of void * pointers (objects).
 * @param n
 *   The number of objects to add in the ring from the obj_table.
 * @param free_space
 *   if non-NULL, returns the amount of space in the ring after the
 *   enqueue operation has finished.
 * @return
 *   The number of objects enqueued, ranging in [0,n].
 */
static __attribute__((always_inline)) inline unsigned int
jring_mp_enqueue_burst(struct jring *r, const void *obj_table, unsigned int n,
                       unsigned int *free_space) {
    return __jring_do_enqueue_elem(r, obj_table, r->esize, n,
                                   JRING_QUEUE_VARIABLE, JRING_SYNC_MT,
                                   free_space);
}

/**
 * Enqueue up to a specific amount of objects on a ring. This function performs
 * a runtime check to find the ring's synchronization type.
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of void * pointers (objects).
 * @param n
 *   The number of objects to add in the ring from the obj_table.
 * @param free_space
 *   if non-NULL, returns the amount of space in the ring after the
 *   enqueue operation has finished.
 * @return
 *   The number of objects enqueued, ranging in [0,n].
 */
static __attribute__((always_inline)) inline unsigned int jring_enqueue_burst(
    struct jring *r, const void *obj_table, unsigned int n,
    unsigned int *free_space) {
    return (r->prod.sync == JRING_SYNC_ST)
               ? jring_sp_enqueue_burst(r, obj_table, n, free_space)
               : jring_mp_enqueue_burst(r, obj_table, n, free_space);
}

/**
 * Dequeue a specific amount of objects from a ring (NOT multi-consumers safe).
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of objects that will be filled.
 * @param n
 *   The number of objects to dequeue from the ring to the obj_table,
 *   must be strictly positive.
 * @param available
 *   If non-NULL, returns the number of remaining ring entries after the
 *   dequeue has finished.
 * @return
 *   The number of objects dequeued, either 0 or n
 */
static __attribute((always_inline)) inline unsigned int jring_sc_dequeue_bulk(
    struct jring *r, void *obj_table, unsigned int n, unsigned int *available) {
    return __jring_do_dequeue_elem(r, obj_table, r->esize, n, JRING_QUEUE_FIXED,
                                   JRING_SYNC_ST, available);
}

/**
 * Dequeue a specific amount of objects from a ring (Multi-consumer safe).
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of objects that will be filled.
 * @param n
 *   The number of objects to dequeue from the ring to the obj_table,
 *   must be strictly positive.
 * @param available
 *   If non-NULL, returns the number of remaining ring entries after the
 *   dequeue has finished.
 * @return
 *   The number of objects dequeued, either 0 or n
 */
static __attribute((always_inline)) inline unsigned int jring_mc_dequeue_bulk(
    struct jring *r, void *obj_table, unsigned int n, unsigned int *available) {
    return __jring_do_dequeue_elem(r, obj_table, r->esize, n, JRING_QUEUE_FIXED,
                                   JRING_SYNC_MT, available);
}

/**
 * Dequeue a specific amount of objects on a ring. This function performs
 * a runtime check to find the ring's synchronization type.
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of void * pointers (objects).
 * @param n
 *   The number of objects to add in the ring from the obj_table.
 * @param available
 *   If non-NULL, returns the number of remaining ring entries after the
 *   dequeue has finished.
 * @return
 *   The number of objects dequeued, either 0 or n.
 */
static __attribute((always_inline)) inline unsigned int jring_dequeue_bulk(
    struct jring *r, void *obj_table, unsigned int n, unsigned int *available) {
    return (r->cons.sync == JRING_SYNC_ST)
               ? jring_sc_dequeue_bulk(r, obj_table, n, available)
               : jring_mc_dequeue_bulk(r, obj_table, n, available);
}

/**
 * Dequeue up to a specific amount of  objects from a ring (NOT multi-consumer
 * safe).
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of objects that will be filled.
 * @param n
 *   The number of objects to dequeue from the ring to the obj_table,
 *   must be strictly positive.
 * @param available
 *   If non-NULL, returns the number of remaining ring entries after the
 *   dequeue has finished.
 * @return
 *   The number of objects dequeued, in the range [0, n]
 */
static __attribute((always_inline)) inline unsigned int jring_sc_dequeue_burst(
    struct jring *r, void *obj_table, unsigned int n, unsigned int *available) {
    return __jring_do_dequeue_elem(r, obj_table, r->esize, n,
                                   JRING_QUEUE_VARIABLE, JRING_SYNC_ST,
                                   available);
}

/**
 * Dequeue up to a specific amount of  objects from a ring (multi-consumer
 * safe).
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of objects that will be filled.
 * @param n
 *   The number of objects to dequeue from the ring to the obj_table,
 *   must be strictly positive.
 * @param available
 *   If non-NULL, returns the number of remaining ring entries after the
 *   dequeue has finished.
 * @return
 *   The number of objects dequeued, in the range [0, n]
 */
static __attribute((always_inline)) inline unsigned int jring_mc_dequeue_burst(
    struct jring *r, void *obj_table, unsigned int n, unsigned int *available) {
    return __jring_do_dequeue_elem(r, obj_table, r->esize, n,
                                   JRING_QUEUE_VARIABLE, JRING_SYNC_MT,
                                   available);
}

/**
 * Dequeue up to a specific amount of objects on a ring. This function performs
 * a runtime check to find the ring's synchronization type.
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of void * pointers (objects).
 * @param n
 *   The number of objects to add in the ring from the obj_table.
 * @param available
 *   If non-NULL, returns the number of remaining ring entries after the
 *   dequeue has finished.
 * @return
 *   The number of objects dequeued, ranging in [0,n].
 */
static __attribute((always_inline)) inline unsigned int jring_dequeue_burst(
    struct jring *r, void *obj_table, unsigned int n, unsigned int *available) {
    return (r->cons.sync == JRING_SYNC_ST)
               ? jring_sc_dequeue_burst(r, obj_table, n, available)
               : jring_mc_dequeue_burst(r, obj_table, n, available);
}

/**
 * Return the number of entries in a ring.
 *
 * @param r
 *   A pointer to the ring structure.
 * @return
 *   The number of entries in the ring.
 */
static __attribute__((always_inline)) inline unsigned int jring_count(
    const struct jring *r) {
    uint32_t prod_tail = r->prod.tail;
    uint32_t cons_tail = r->cons.tail;
    uint32_t count = (prod_tail - cons_tail) & r->mask;
    return (count > r->capacity) ? r->capacity : count;
}

/**
 * Return the number of free entries in a ring.
 *
 * @param r
 *   A pointer to the ring structure.
 * @return
 *   The number of free entries in the ring.
 */
static __attribute__((always_inline)) inline unsigned int jring_free_count(
    const struct jring *r) {
    return r->capacity - jring_count(r);
}

/**
 * Test if a ring is full.
 *
 * @param r
 *   A pointer to the ring structure.
 * @return
 *   - 1: The ring is full.
 *   - 0: The ring is not full.
 */
static __attribute__((always_inline)) inline int jring_full(
    const struct jring *r) {
    return jring_free_count(r) == 0;
}

/**
 * Test if a ring is empty.
 *
 * @param r
 *   A pointer to the ring structure.
 * @return
 *   - 1: The ring is empty.
 *   - 0: The ring is not empty.
 */
static __attribute__((always_inline)) inline int jring_empty(
    const struct jring *r) {
    uint32_t prod_tail = r->prod.tail;
    uint32_t cons_tail = r->cons.tail;
    return cons_tail == prod_tail;
}

#ifdef __cplusplus
}
#endif
