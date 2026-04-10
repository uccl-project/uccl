#pragma once
#include "uccl_engine.h"

// Function pointer table for runtime transport dispatch.
// Each backend .so exports uccl_p2p_get_ops() returning a pointer to this
// struct populated with its implementations.
struct uccl_p2p_ops {
  decltype(&uccl_engine_create)          create;
  decltype(&uccl_engine_destroy)         destroy;
  decltype(&uccl_engine_connect)         connect;
  decltype(&uccl_engine_accept)          accept;
  decltype(&uccl_engine_reg)             reg;
  decltype(&uccl_engine_read)            read;
  decltype(&uccl_engine_read_vector)     read_vector;
  decltype(&uccl_engine_send)            send;
  decltype(&uccl_engine_write)           write;
  decltype(&uccl_engine_write_vector)    write_vector;
  decltype(&uccl_engine_recv)            recv;
  decltype(&uccl_engine_xfer_status)     xfer_status;
  decltype(&uccl_engine_start_listener)  start_listener;
  decltype(&uccl_engine_stop_accept)     stop_accept;
  decltype(&uccl_engine_conn_destroy)    conn_destroy;
  decltype(&uccl_engine_mr_destroy)      mr_destroy;
  decltype(&uccl_engine_get_metadata)    get_metadata;
  decltype(&uccl_engine_get_notifs)      get_notifs;
  decltype(&uccl_engine_send_notif)      send_notif;
  decltype(&uccl_engine_prepare_fifo)    prepare_fifo;
  decltype(&uccl_engine_update_fifo)     update_fifo;
  decltype(&uccl_engine_conn_is_local)   conn_is_local;
  decltype(&uccl_engine_get_ipc_info)    get_ipc_info;
  decltype(&uccl_engine_update_ipc_info) update_ipc_info;
};

// Single extern "C" symbol that dlsym resolves. Everything else is accessed
// through the returned ops struct.
extern "C" uccl_p2p_ops* uccl_p2p_get_ops();
