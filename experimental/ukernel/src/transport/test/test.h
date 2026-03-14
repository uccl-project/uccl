#pragma once

void test_transport_core();
void test_generate_host_id();
int test_transport_communicator(int argc, char** argv);
void test_transport_communicator_local();
void test_socket_oob();
void test_socket_meta_exchange_multi_threads(int world_size);
void test_redis_oob();
void test_redis_meta_exchange_multi_threads(int world_size);
void test_uds_oob();
