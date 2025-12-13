#pragma once

#include <map>
#include <string>

// for redis
#ifdef USE_REDIS_OOB
#include <hiredis/hiredis.h>
#endif
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <thread>

// for RDMAConnectionInfo
#include <iomanip>
#include <sstream>
#include <vector>

// for socket
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <functional>

struct Exchangeable {
  virtual std::map<std::string, std::string> to_map() const = 0;
  virtual void from_map(std::map<std::string, std::string> const& kv) = 0;
  virtual ~Exchangeable() {}
};

struct CommunicatorMeta : public Exchangeable {
  std::string host_id;
  std::string ip;
  bool is_ready;
  // TODO: connection abality // Support RDMA, PCIe, NVlink, UAlink, etc.

  CommunicatorMeta() = default;

  std::map<std::string, std::string> to_map() const override {
    std::map<std::string, std::string> kv;
    kv["host_id"] = host_id;
    kv["ip"] = ip;
    kv["is_ready"] = is_ready ? "1" : "0";
    return kv;
  }

  void from_map(std::map<std::string, std::string> const& kv) override {
    host_id = kv.at("host_id");
    ip = kv.at("ip");
    std::string const& ready_str = kv.at("is_ready");
    is_ready = (ready_str == "1");
  }
};

struct QpInfo {
  uint32_t qp_num;
  uint32_t psn;
  uint16_t lid;     // Local ID
  uint8_t gid[16];  // Global ID for RoCE (optional)
};

struct RDMAInfo : public Exchangeable {
  std::vector<QpInfo> qps;

  RDMAInfo() = default;

  std::map<std::string, std::string> to_map() const override {
    std::map<std::string, std::string> kv;
    kv["count"] = std::to_string(qps.size());

    for (size_t i = 0; i < qps.size(); ++i) {
      auto const& qp = qps[i];
      kv["qp_" + std::to_string(i) + "_num"] = std::to_string(qp.qp_num);
      kv["qp_" + std::to_string(i) + "_psn"] = std::to_string(qp.psn);
      kv["qp_" + std::to_string(i) + "_lid"] = std::to_string(qp.lid);

      std::ostringstream oss;
      oss << std::hex << std::setfill('0');
      for (int j = 0; j < 16; ++j) {
        oss << std::setw(2) << static_cast<int>(qp.gid[j]);
      }
      kv["qp_" + std::to_string(i) + "_gid"] = oss.str();
    }

    return kv;
  }

  void from_map(std::map<std::string, std::string> const& kv) override {
    size_t count = std::stoul(kv.at("count"));
    qps.resize(count);

    for (size_t i = 0; i < count; ++i) {
      auto& qp = qps[i];
      qp.qp_num = std::stoul(kv.at("qp_" + std::to_string(i) + "_num"));
      qp.psn = std::stoul(kv.at("qp_" + std::to_string(i) + "_psn"));
      qp.lid = std::stoul(kv.at("qp_" + std::to_string(i) + "_lid"));

      std::string const& hex = kv.at("qp_" + std::to_string(i) + "_gid");
      for (int j = 0; j < 16; ++j) {
        unsigned int byte;
        std::stringstream ss;
        ss << std::hex << hex.substr(j * 2, 2);
        ss >> byte;
        qp.gid[j] = static_cast<unsigned char>(byte);
      }
    }
  }
};

struct MR {
  uint32_t id;
  uint64_t address;
  uint32_t length;
  uint32_t lkey;
  uint32_t key;
};

struct MRInfos : public Exchangeable {
  std::vector<MR> mrs;

  MRInfos() = default;

  std::map<std::string, std::string> to_map() const override {
    std::map<std::string, std::string> kv;
    kv["count"] = std::to_string(mrs.size());

    for (size_t i = 0; i < mrs.size(); ++i) {
      auto const& mr = mrs[i];
      kv["mr_" + std::to_string(i) + "_id"] = std::to_string(mr.id);

      {
        std::ostringstream oss;
        oss << std::hex << std::setw(16) << std::setfill('0') << mr.address;
        kv["mr_" + std::to_string(i) + "_addr"] = oss.str();
      }

      kv["mr_" + std::to_string(i) + "_len"] = std::to_string(mr.length);
      kv["mr_" + std::to_string(i) + "_key"] = std::to_string(mr.key);
    }

    return kv;
  }

  void from_map(std::map<std::string, std::string> const& kv) override {
    size_t count = std::stoul(kv.at("count"));
    mrs.resize(count);

    for (size_t i = 0; i < count; ++i) {
      auto& mr = mrs[i];
      mr.id = std::stoul(kv.at("mr_" + std::to_string(i) + "_id"));
      // address (64-bit)
      {
        std::string hexaddr = kv.at("mr_" + std::to_string(i) + "_addr");
        mr.address = std::stoull(hexaddr, nullptr, 16);
      }
      // length and key
      mr.length = std::stoul(kv.at("mr_" + std::to_string(i) + "_len"));
      mr.key = std::stoul(kv.at("mr_" + std::to_string(i) + "_key"));
    }
  }
};

class Exchanger {
public:
    virtual ~Exchanger() = default;

    virtual bool valid() const = 0;
    virtual bool publish(const std::string& key, const Exchangeable& obj) = 0;
    virtual bool fetch(const std::string& key, Exchangeable& obj) = 0;
    virtual bool wait_and_fetch(
        const std::string& key,
        Exchangeable& obj,
        int max_retries = 50,
        int delay_ms = 100) = 0;
};

class RedisExchanger : public Exchanger {
public:
    RedisExchanger(const std::string& host = "127.0.0.1", int port = 6379,
                   int timeout_ms = 2000);
    ~RedisExchanger();

    bool valid() const override;
    bool publish(const std::string& key, const Exchangeable& obj) override;
    bool fetch(const std::string& key, Exchangeable& obj) override;
    bool wait_and_fetch(const std::string& key, Exchangeable& obj,
                        int max_retries = 50, int delay_ms = 100) override;

private:
#ifdef USE_REDIS_OOB
    redisContext* ctx_;
#endif
};

class SockExchanger : public Exchanger {
public:
    SockExchanger(bool is_server,
                  const std::string& host,
                  int port,
                  int timeout_ms = 3000,
                  size_t max_line_bytes = 1 * 1024 * 1024 /* 1MB */);
    ~SockExchanger();

    bool valid() const;

    // client-only
    bool publish(const std::string& key, const Exchangeable& obj);
    bool fetch(const std::string& key, Exchangeable& obj);

    bool wait_and_fetch(const std::string& key, Exchangeable& obj,
                        int max_retries = -1, int delay_ms = 100);

private:
    bool start_server();
    bool connect_client();
    bool send_cmd_and_recv(const std::string& cmd, std::string& resp);

private:
    int sock_fd_;
    std::string host_;
    int port_;
    int timeout_ms_;
    bool is_server_;
    int listen_fd_;
    std::atomic<bool> running_;
    size_t max_line_bytes_;

    // server side state
    std::unordered_map<std::string, std::map<std::string, std::string>> store_;
    std::mutex store_mutex_;
    std::thread server_thread_;
    std::mutex conn_threads_mutex_;
    std::vector<std::thread> conn_threads_;

    friend void handle_connection(int,
                                  std::unordered_map<std::string, std::map<std::string, std::string>>&,
                                  std::mutex&,
                                  std::atomic<bool>&,
                                  size_t,
                                  std::function<void(std::thread&&)>);
};
