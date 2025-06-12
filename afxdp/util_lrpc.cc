/*
 * lrpc.cc - shared memory communication channels
 */

#include "util/lrpc.h"
#include <errno.h>

int __lrpc_send(struct lrpc_chan_out* chan, lrpc_msg* msg) {
  lrpc_msg* dst;
  uint64_t cmd = msg->cmd;

  assert(chan->send_head - chan->send_tail == chan->size);

  chan->send_tail = load_acquire(chan->recv_head_wb);

  if (chan->send_head - chan->send_tail == chan->size) return -1;

  dst = &chan->tbl[chan->send_head & (chan->size - 1)];

  memcpy(dst->data, msg->data, sizeof(dst->data));

  cmd |= (chan->send_head++ & chan->size) ? 0 : LRPC_DONE_PARITY;
  store_release(&dst->cmd, cmd);
  return 0;
}

/**
 * lrpc_init_out - initializes an egress shared memory channel
 * @chan: the channel struct to initialize
 * @tbl: a buffer to store channel messages
 * @size: the number of message elements in the buffer
 * @recv_head_wb: a pointer to the head position of the receiver
 *
 * returns 0 if successful, or -EINVAL if @size is not a power of two.
 */
int lrpc_init_out(struct lrpc_chan_out* chan, lrpc_msg* tbl, unsigned int size,
                  uint32_t* recv_head_wb) {
  if (!is_power_of_two(size)) return -EINVAL;

  memset(chan, 0, sizeof(*chan));
  chan->tbl = tbl;
  chan->size = size;
  chan->recv_head_wb = recv_head_wb;
  return 0;
}

/**
 * lrpc_init_in - initializes an ingress shared memory channel
 * @chan: the channel struct to initialize
 * @tbl: a buffer to store channel messages
 * @size: the number of message elements in the buffer
 * @recv_head_wb: a pointer to the head position of the receiver
 *
 * returns 0 if successful, or -EINVAL if @size is not a power of two.
 */
int lrpc_init_in(struct lrpc_chan_in* chan, lrpc_msg* tbl, unsigned int size,
                 uint32_t* recv_head_wb) {
  if (!is_power_of_two(size)) return -EINVAL;

  memset(chan, 0, sizeof(*chan));
  chan->tbl = tbl;
  chan->size = size;
  chan->recv_head_wb = recv_head_wb;
  return 0;
}