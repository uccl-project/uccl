/* SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2017,2018 HXT-semitech Corporation.
 * Copyright (c) 2007-2009 Kip Macy kmacy@freebsd.org
 * All rights reserved.
 * Derived from FreeBSD's bufring.h
 * Used as BSD-3 Licensed with permission from Kip Macy.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "pause.h"
#include <memory.h>
#include <stdint.h>

#ifndef likely
#define likely(x) __builtin_expect((x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif

typedef struct {
  union {
    uint64_t val[2];
  };
} __attribute__((aligned(16))) j_int128_t;

static inline __attribute__((always_inline)) void __jring_enqueue_elems_32(
    struct jring* r, const uint32_t size, uint32_t idx, void const* obj_table,
    uint32_t n) {
  unsigned int i;
  uint32_t* ring = (uint32_t*)&r[1];
  uint32_t const* obj = (uint32_t const*)obj_table;
  if (likely(idx + n < size)) {
    for (i = 0; i < (n & ~0x7); i += 8, idx += 8) {
      ring[idx] = obj[i];
      ring[idx + 1] = obj[i + 1];
      ring[idx + 2] = obj[i + 2];
      ring[idx + 3] = obj[i + 3];
      ring[idx + 4] = obj[i + 4];
      ring[idx + 5] = obj[i + 5];
      ring[idx + 6] = obj[i + 6];
      ring[idx + 7] = obj[i + 7];
    }
    switch (n & 0x7) {
      case 7:
        ring[idx++] = obj[i++]; /* fallthrough */
      case 6:
        ring[idx++] = obj[i++]; /* fallthrough */
      case 5:
        ring[idx++] = obj[i++]; /* fallthrough */
      case 4:
        ring[idx++] = obj[i++]; /* fallthrough */
      case 3:
        ring[idx++] = obj[i++]; /* fallthrough */
      case 2:
        ring[idx++] = obj[i++]; /* fallthrough */
      case 1:
        ring[idx++] = obj[i++]; /* fallthrough */
    }
  } else {
    for (i = 0; idx < size; i++, idx++) ring[idx] = obj[i];
    /* Start at the beginning */
    for (idx = 0; i < n; i++, idx++) ring[idx] = obj[i];
  }
}

static __attribute__((always_inline)) inline void __jring_enqueue_elems_64(
    struct jring* r, uint32_t prod_head, void const* obj_table, uint32_t n) {
  unsigned int i;
  const uint32_t size = r->size;
  uint32_t idx = prod_head & r->mask;
  uint64_t* ring = (uint64_t*)&r[1];
  uint64_t const* obj = (uint64_t const*)obj_table;
  if (likely(idx + n < size)) {
    for (i = 0; i < (n & ~0x3); i += 4, idx += 4) {
      ring[idx] = obj[i];
      ring[idx + 1] = obj[i + 1];
      ring[idx + 2] = obj[i + 2];
      ring[idx + 3] = obj[i + 3];
    }
    switch (n & 0x3) {
      case 3:
        ring[idx++] = obj[i++]; /* fallthrough */
      case 2:
        ring[idx++] = obj[i++]; /* fallthrough */
      case 1:
        ring[idx++] = obj[i++];
    }
  } else {
    for (i = 0; idx < size; i++, idx++) ring[idx] = obj[i];
    /* Start at the beginning */
    for (idx = 0; i < n; i++, idx++) ring[idx] = obj[i];
  }
}

static inline __attribute__((always_inline)) void __jring_enqueue_elems_128(
    struct jring* r, uint32_t prod_head, void const* obj_table, uint32_t n) {
  unsigned int i;
  const uint32_t size = r->size;
  uint32_t idx = prod_head & r->mask;
  j_int128_t* ring = (j_int128_t*)&r[1];
  j_int128_t const* obj = (j_int128_t const*)obj_table;
  if (likely(idx + n < size)) {
    for (i = 0; i < (n & ~0x1); i += 2, idx += 2)
      memcpy((void*)(ring + idx), (void const*)(obj + i), 32);
    switch (n & 0x1) {
      case 1:
        memcpy((void*)(ring + idx), (void const*)(obj + i), 16);
    }
  } else {
    for (i = 0; idx < size; i++, idx++)
      memcpy((void*)(ring + idx), (void const*)(obj + i), 16);
    /* Start at the beginning */
    for (idx = 0; i < n; i++, idx++)
      memcpy((void*)(ring + idx), (void const*)(obj + i), 16);
  }
}

/* the actual enqueue of elements on the ring.
 * Placed here since identical code needed in both
 * single and multi producer enqueue functions.
 */
static inline __attribute__((always_inline)) void __jring_enqueue_elems(
    struct jring* r, uint32_t prod_head, void const* obj_table, uint32_t esize,
    uint32_t num) {
  /* 8B and 16B copies implemented individually to retain
   * the current performance.
   */
  if (esize == 8) {
    __jring_enqueue_elems_64(r, prod_head, obj_table, num);
  } else if (esize == 16) {
    __jring_enqueue_elems_128(r, prod_head, obj_table, num);
  } else {
    uint32_t idx, scale, nr_idx, nr_num, nr_size;

    /* Normalize to uint32_t */
    scale = esize / sizeof(uint32_t);
    nr_num = num * scale;
    idx = prod_head & r->mask;
    nr_idx = idx * scale;
    nr_size = r->size * scale;
    __jring_enqueue_elems_32(r, nr_size, nr_idx, obj_table, nr_num);
  }
}

static inline __attribute__((always_inline)) void __jring_dequeue_elems_32(
    struct jring* r, const uint32_t size, uint32_t idx, void* obj_table,
    uint32_t n) {
  unsigned int i;
  uint32_t* ring = (uint32_t*)&r[1];
  uint32_t* obj = (uint32_t*)obj_table;
  if (likely(idx + n < size)) {
    for (i = 0; i < (n & ~0x7); i += 8, idx += 8) {
      obj[i] = ring[idx];
      obj[i + 1] = ring[idx + 1];
      obj[i + 2] = ring[idx + 2];
      obj[i + 3] = ring[idx + 3];
      obj[i + 4] = ring[idx + 4];
      obj[i + 5] = ring[idx + 5];
      obj[i + 6] = ring[idx + 6];
      obj[i + 7] = ring[idx + 7];
    }
    switch (n & 0x7) {
      case 7:
        obj[i++] = ring[idx++]; /* fallthrough */
      case 6:
        obj[i++] = ring[idx++]; /* fallthrough */
      case 5:
        obj[i++] = ring[idx++]; /* fallthrough */
      case 4:
        obj[i++] = ring[idx++]; /* fallthrough */
      case 3:
        obj[i++] = ring[idx++]; /* fallthrough */
      case 2:
        obj[i++] = ring[idx++]; /* fallthrough */
      case 1:
        obj[i++] = ring[idx++]; /* fallthrough */
    }
  } else {
    for (i = 0; idx < size; i++, idx++) obj[i] = ring[idx];
    /* Start at the beginning */
    for (idx = 0; i < n; i++, idx++) obj[i] = ring[idx];
  }
}

static inline __attribute__((always_inline)) void __jring_dequeue_elems_64(
    struct jring* r, uint32_t prod_head, void* obj_table, uint32_t n) {
  unsigned int i;
  const uint32_t size = r->size;
  uint32_t idx = prod_head & r->mask;
  uint64_t* ring = (uint64_t*)&r[1];
  uint64_t* obj = (uint64_t*)obj_table;
  if (likely(idx + n < size)) {
    for (i = 0; i < (n & ~0x3); i += 4, idx += 4) {
      obj[i] = ring[idx];
      obj[i + 1] = ring[idx + 1];
      obj[i + 2] = ring[idx + 2];
      obj[i + 3] = ring[idx + 3];
    }
    switch (n & 0x3) {
      case 3:
        obj[i++] = ring[idx++]; /* fallthrough */
      case 2:
        obj[i++] = ring[idx++]; /* fallthrough */
      case 1:
        obj[i++] = ring[idx++]; /* fallthrough */
    }
  } else {
    for (i = 0; idx < size; i++, idx++) obj[i] = ring[idx];
    /* Start at the beginning */
    for (idx = 0; i < n; i++, idx++) obj[i] = ring[idx];
  }
}

static inline __attribute__((always_inline)) void __jring_dequeue_elems_128(
    struct jring* r, uint32_t prod_head, void* obj_table, uint32_t n) {
  unsigned int i;
  const uint32_t size = r->size;
  uint32_t idx = prod_head & r->mask;
  j_int128_t* ring = (j_int128_t*)&r[1];
  j_int128_t* obj = (j_int128_t*)obj_table;
  if (likely(idx + n < size)) {
    for (i = 0; i < (n & ~0x1); i += 2, idx += 2)
      memcpy((void*)(obj + i), (void*)(ring + idx), 32);
    switch (n & 0x1) {
      case 1:
        memcpy((void*)(obj + i), (void*)(ring + idx), 16);
    }
  } else {
    for (i = 0; idx < size; i++, idx++)
      memcpy((void*)(obj + i), (void*)(ring + idx), 16);
    /* Start at the beginning */
    for (idx = 0; i < n; i++, idx++)
      memcpy((void*)(obj + i), (void*)(ring + idx), 16);
  }
}

/* the actual dequeue of elements from the ring.
 * Placed here since identical code needed in both
 * single and multi producer enqueue functions.
 */
static inline __attribute__((always_inline)) void __jring_dequeue_elems(
    struct jring* r, uint32_t cons_head, void* obj_table, uint32_t esize,
    uint32_t num) {
  /* 8B and 16B copies implemented individually to retain
   * the current performance.
   */
  if (esize == 8) {
    __jring_dequeue_elems_64(r, cons_head, obj_table, num);
  } else if (esize == 16) {
    __jring_dequeue_elems_128(r, cons_head, obj_table, num);
  } else {
    uint32_t idx, scale, nr_idx, nr_num, nr_size;

    /* Normalize to uint32_t */
    scale = esize / sizeof(uint32_t);
    nr_num = num * scale;
    idx = cons_head & r->mask;
    nr_idx = idx * scale;
    nr_size = r->size * scale;
    __jring_dequeue_elems_32(r, nr_size, nr_idx, obj_table, nr_num);
  }
}

static inline __attribute__((always_inline)) void __jring_wait_until_equal_32(
    uint32_t volatile* addr, uint32_t expected, int memorder) {
  // assert(memorder == __ATOMIC_ACQUIRE || memorder == __ATOMIC_RELAXED);

  while (__atomic_load_n(addr, memorder) != expected) machnet_pause();
}

static inline __attribute__((always_inline)) void __jring_update_tail(
    struct jring_headtail* ht, uint32_t old_val, uint32_t new_val,
    uint32_t single, __attribute__((unused)) uint32_t enqueue) {
  /*
   * If there are other enqueues/dequeues in progress that preceded us,
   * we need to wait for them to complete
   */
  if (!single)
    __jring_wait_until_equal_32(&ht->tail, old_val, __ATOMIC_RELAXED);

  __atomic_store_n(&ht->tail, new_val, __ATOMIC_RELEASE);
}

/**
 * @internal This function updates the producer head for enqueue
 *
 * @param r
 *   A pointer to the ring structure
 * @param is_sp
 *   Indicates whether multi-producer path is needed or not
 * @param n
 *   The number of elements we will want to enqueue, i.e. how far should the
 *   head be moved
 * @param behavior
 *   JRING_QUEUE_FIXED:    Enqueue a fixed number of items from a ring
 *   JRING_QUEUE_VARIABLE: Enqueue as many items as possible from ring
 * @param old_head
 *   Returns head value as it was before the move, i.e. where enqueue starts
 * @param new_head
 *   Returns the current/new head value i.e. where enqueue finishes
 * @param free_entries
 *   Returns the amount of free space in the ring BEFORE head was moved
 * @return
 *   Actual number of objects enqueued.
 *   If behavior == JRING_QUEUE_FIXED, this will be 0 or n only.
 */
static inline __attribute__((always_inline)) unsigned int
__jring_move_prod_head(struct jring* r, unsigned int is_sp, unsigned int n,
                       enum jring_queue_behavior behavior, uint32_t* old_head,
                       uint32_t* new_head, uint32_t* free_entries) {
  const uint32_t capacity = r->capacity;
  uint32_t cons_tail;
  unsigned int max = n;
  int success;

  *old_head = __atomic_load_n(&r->prod.head, __ATOMIC_RELAXED);
  do {
    /* Reset n to the initial burst count */
    n = max;

    /* Ensure the head is read before tail */
    __atomic_thread_fence(__ATOMIC_ACQUIRE);

    /* load-acquire synchronize with store-release of ht->tail
     * in update_tail.
     */
    cons_tail = __atomic_load_n(&r->cons.tail, __ATOMIC_ACQUIRE);

    /* The subtraction is done between two unsigned 32bits value
     * (the result is always modulo 32 bits even if we have
     * *old_head > cons_tail). So 'free_entries' is always between 0
     * and capacity (which is < size).
     */
    *free_entries = (capacity + cons_tail - *old_head);

    /* check that we have enough room in ring */
    if (unlikely(n > *free_entries))
      n = (behavior == JRING_QUEUE_FIXED) ? 0 : *free_entries;

    if (n == 0) return 0;

    *new_head = *old_head + n;
    if (is_sp)
      r->prod.head = *new_head, success = 1;
    else
      /* on failure, *old_head is updated */
      success =
          __atomic_compare_exchange_n(&r->prod.head, old_head, *new_head, 0,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  } while (unlikely(success == 0));
  return n;
}

/**
 * @internal This function updates the consumer head for dequeue
 *
 * @param r
 *   A pointer to the ring structure
 * @param is_sc
 *   Indicates whether multi-consumer path is needed or not
 * @param n
 *   The number of elements we will want to dequeue, i.e. how far should the
 *   head be moved
 * @param behavior
 *   JRING_QUEUE_FIXED:    Dequeue a fixed number of items from a ring
 *   JRING_QUEUE_VARIABLE: Dequeue as many items as possible from ring
 * @param old_head
 *   Returns head value as it was before the move, i.e. where dequeue starts
 * @param new_head
 *   Returns the current/new head value i.e. where dequeue finishes
 * @param entries
 *   Returns the number of entries in the ring BEFORE head was moved
 * @return
 *   - Actual number of objects dequeued.
 *     If behavior == JRING_QUEUE_FIXED, this will be 0 or n only.
 */
static inline __attribute__((always_inline)) unsigned int
__jring_move_cons_head(struct jring* r, int is_sc, unsigned int n,
                       enum jring_queue_behavior behavior, uint32_t* old_head,
                       uint32_t* new_head, uint32_t* entries) {
  unsigned int max = n;
  uint32_t prod_tail;
  int success;

  /* move cons.head atomically */
  *old_head = __atomic_load_n(&r->cons.head, __ATOMIC_RELAXED);
  do {
    /* Restore n as it may change every loop */
    n = max;

    /* Ensure the head is read before tail */
    __atomic_thread_fence(__ATOMIC_ACQUIRE);

    /* this load-acquire synchronize with store-release of ht->tail
     * in update_tail.
     */
    prod_tail = __atomic_load_n(&r->prod.tail, __ATOMIC_ACQUIRE);

    /* The subtraction is done between two unsigned 32bits value
     * (the result is always modulo 32 bits even if we have
     * cons_head > prod_tail). So 'entries' is always between 0
     * and size(ring)-1.
     */
    *entries = (prod_tail - *old_head);

    /* Set the actual entries for dequeue */
    if (n > *entries) n = (behavior == JRING_QUEUE_FIXED) ? 0 : *entries;

    if (unlikely(n == 0)) return 0;

    *new_head = *old_head + n;
    if (is_sc)
      r->cons.head = *new_head, success = 1;
    else
      /* on failure, *old_head will be updated */
      success =
          __atomic_compare_exchange_n(&r->cons.head, old_head, *new_head, 0,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  } while (unlikely(success == 0));
  return n;
}

/**
 * @internal Enqueue several objects on the ring
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of objects.
 * @param esize
 *   The size of ring element, in bytes. It must be a multiple of 4.
 *   This must be the same value used while creating the ring. Otherwise
 *   the results are undefined.
 * @param n
 *   The number of objects to add in the ring from the obj_table.
 * @param behavior
 *   jRING_QUEUE_FIXED:    Enqueue a fixed number of items from a ring
 *   jRING_QUEUE_VARIABLE: Enqueue as many items as possible from ring
 * @param is_sp
 *   Indicates whether to use single producer or multi-producer head update
 * @param free_space
 *   returns the amount of space after the enqueue operation has finished
 * @return
 *   Actual number of objects enqueued.
 *   If behavior == jRING_QUEUE_FIXED, this will be 0 or n only.
 */
static inline __attribute__((always_inline)) unsigned int
__jring_do_enqueue_elem(struct jring* r, void const* obj_table,
                        unsigned int esize, unsigned int n,
                        enum jring_queue_behavior behavior, unsigned int is_sp,
                        unsigned int* free_space) {
  uint32_t prod_head, prod_next;
  uint32_t free_entries;

  n = __jring_move_prod_head(r, is_sp, n, behavior, &prod_head, &prod_next,
                             &free_entries);
  if (n == 0) goto end;

  __jring_enqueue_elems(r, prod_head, obj_table, esize, n);

  __jring_update_tail(&r->prod, prod_head, prod_next, is_sp, 1);
end:
  if (free_space != NULL) *free_space = free_entries - n;
  return n;
}

/**
 * @internal Dequeue several objects from the ring
 *
 * @param r
 *   A pointer to the ring structure.
 * @param obj_table
 *   A pointer to a table of objects.
 * @param esize
 *   The size of ring element, in bytes. It must be a multiple of 4.
 *   This must be the same value used while creating the ring. Otherwise
 *   the results are undefined.
 * @param n
 *   The number of objects to pull from the ring.
 * @param behavior
 *   jRING_QUEUE_FIXED:    Dequeue a fixed number of items from a ring
 *   jRING_QUEUE_VARIABLE: Dequeue as many items as possible from ring
 * @param is_sc
 *   Indicates whether to use single consumer or multi-consumer head update
 * @param available
 *   returns the number of remaining ring entries after the dequeue has finished
 * @return
 *   - Actual number of objects dequeued.
 *     If behavior == jRING_QUEUE_FIXED, this will be 0 or n only.
 */
static inline __attribute__((always_inline)) unsigned int
__jring_do_dequeue_elem(struct jring* r, void* obj_table, unsigned int esize,
                        unsigned int n, enum jring_queue_behavior behavior,
                        unsigned int is_sc, unsigned int* available) {
  uint32_t cons_head, cons_next;
  uint32_t entries;

  n = __jring_move_cons_head(r, (int)is_sc, n, behavior, &cons_head, &cons_next,
                             &entries);
  if (n == 0) goto end;

  __jring_dequeue_elems(r, cons_head, obj_table, esize, n);

  __jring_update_tail(&r->cons, cons_head, cons_next, is_sc, 0);

end:
  if (available != NULL) *available = entries - n;
  return n;
}

#ifdef __cplusplus
}
#endif
