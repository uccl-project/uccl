#pragma once

#include <string>
#include <cstdlib>

#define DEFAULT_CHUNK_SIZE 1024*1024
#define DEFAULT_QP_PER_EP 4
#define DEFAULT_CQ_PER_EP 4
#define DEFAULT_CQ_POLLER_THREADS 4
#define DEFAULT_MAX_RETRY_TIMES 10
#define DEFAULT_SOCKET_IP "0.0.0.0"
#define DEFAULT_SOCKET_PORT 9999
#define DEFAULT_RESOLVE_TIMEOUT_MS 2000
#define DEFAULT_QP_MAX_SEND_WR 128
#define DEFAULT_QP_MAX_RECV_WR 128
#define DEFAUTL_QP_MAX_SGE 1
#define DEFAULT_REDIS_SERVER_IP "0.0.0.0"
#define DEFAULT_REDIS_SERVER_PORT 6379

struct Config {
    int chunk_size;
    int qp_count_per_ep;
    int cq_count_per_ep;
    int cq_poller_threads;
    int max_retry_times;
    int resolve_timeout_ms;
    int qp_max_send_wr;
    int qp_max_recv_wr;
    int qp_max_sge;
    std::string redis_ip;
    int redis_port;

    Config()
        : chunk_size(getEnvOrDefault("ECCL_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
          qp_count_per_ep(getEnvOrDefault("ECCL_QP_COUNT", DEFAULT_QP_PER_EP)),
          cq_count_per_ep(getEnvOrDefault("ECCL_CQ_COUNT", DEFAULT_CQ_PER_EP)),
          cq_poller_threads(getEnvOrDefault("ECCL_CQ_POLLER_THREADS", DEFAULT_CQ_POLLER_THREADS)),
          max_retry_times(getEnvOrDefault("ECCL_MAX_RETRY_TIMES", DEFAULT_MAX_RETRY_TIMES)),
          resolve_timeout_ms(getEnvOrDefault("ECCL_RESOLVE_TIMEOUT_MS", DEFAULT_RESOLVE_TIMEOUT_MS)),
          qp_max_send_wr(getEnvOrDefault("ECCL_QP_MAX_SEND_WR", DEFAULT_QP_MAX_SEND_WR)),
          qp_max_recv_wr(getEnvOrDefault("ECCL_QP_MAX_RECV_WR", DEFAULT_QP_MAX_RECV_WR)),
          qp_max_sge(getEnvOrDefault("ECCL_QP_MAX_SGE", DEFAUTL_QP_MAX_SGE)),
          redis_ip(getEnvOrDefault("ECCL_REDIS_SERVER_IP", DEFAULT_REDIS_SERVER_IP)),
          redis_port(getEnvOrDefault("ECCL_REDIS_SERVER_PORT", DEFAULT_REDIS_SERVER_PORT))
        {}

private:
    static int getEnvOrDefault(const char* env_name, int default_val) {
        const char* val = std::getenv(env_name);
        if (val) {
            try {
                return std::stoi(val);
            } catch (...) {
                return default_val;
            }
        }
        return default_val;
    }

    static std::string getEnvOrDefault(const char* env_name, const std::string& default_val) {
        const char* val = std::getenv(env_name);
        if (val) return std::string(val);
        return default_val;
    }
};
