#pragma once

/**
 * @brief A Linux-style intrusive doubly linked list implementation.
 */

#include <cstddef>  // for offsetof

namespace uccl {

extern "C" {

struct list_head {
  list_head* next;
  list_head* prev;
};

#define LIST_HEAD_INIT(name) \
  { &(name), &(name) }

#define LIST_HEAD(name) struct list_head name = LIST_HEAD_INIT(name)

#define container_of(ptr, type, member) \
  ((type*)((char*)(ptr)-offsetof(type, member)))

static inline void INIT_LIST_HEAD(struct list_head* list) {
  list->next = list;
  list->prev = list;
}

static inline void __list_add(struct list_head* new_node,
                              struct list_head* prev, struct list_head* next) {
  next->prev = new_node;
  new_node->next = next;
  new_node->prev = prev;
  prev->next = new_node;
}

static inline void list_add(struct list_head* new_node,
                            struct list_head* head) {
  __list_add(new_node, head, head->next);
}

static inline void list_add_tail(struct list_head* new_node,
                                 struct list_head* head) {
  __list_add(new_node, head->prev, head);
}

static inline void __list_del(struct list_head* prev, struct list_head* next) {
  next->prev = prev;
  prev->next = next;
}

static inline void list_del(struct list_head* entry) {
  __list_del(entry->prev, entry->next);
  entry->next = entry;
  entry->prev = entry;
}

static inline int list_empty(const struct list_head* head) {
  return head->next == head;
}

#define list_entry(ptr, type, member) container_of(ptr, type, member)

#define list_for_each(pos, head) \
  for (pos = (head)->next; pos != (head); pos = pos->next)

#define list_for_each_safe(pos, n, head) \
  for (pos = (head)->next, n = pos->next; pos != (head); pos = n, n = pos->next)
}

}  // namespace uccl