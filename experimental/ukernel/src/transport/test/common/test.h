#pragma once

// Unit tests
void test_transport_core();
void test_generate_host_id();
void test_transport_host_bounce_pool();
void test_transport_tcp_adapter();
void test_transport_oob_exchangeables();
void test_socket_oob();
void test_socket_meta_exchange_multi_threads(int world_size);
void test_shm_oob();

// Integration tests
int test_transport_communicator(int argc, char** argv);
void test_transport_communicator_local();
