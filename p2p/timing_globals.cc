// Thread-local timing accumulators
// These are defined here and declared extern in endpoint_wrapper.h

thread_local double g_set_request_alloc_time_us = 0;
thread_local double g_set_request_writeorread_time_us = 0;
thread_local int g_set_request_count = 0;
thread_local double g_poll_sendchannel_time_us = 0;
thread_local double g_poll_checksend_time_us = 0;
thread_local double g_poll_recvchannel_time_us = 0;
thread_local double g_poll_checkrecv_time_us = 0;
thread_local int g_poll_count = 0;

// Made with Bob
