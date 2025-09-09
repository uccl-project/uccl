#pragma once

#include <map>
#include <string>

// for redis
#include <hiredis/hiredis.h>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <thread>

// for RDMAConnectionInfo
#include <iomanip>
#include <sstream>
#include <vector>

struct Exchangeable {
  virtual std::map<std::string, std::string> to_map() const = 0;
  virtual void from_map(std::map<std::string, std::string> const& kv) = 0;
  virtual ~Exchangeable() {}
};

struct CommunicatorMeta : public Exchangeable {
  std::string host_id;
  bool is_ready;
  // TODO: connection abality // support RDMA, PCIe, NVlink, UAlink, etc.

  CommunicatorMeta() = default;

  std::map<std::string, std::string> to_map() const override {
    std::map<std::string, std::string> kv;
    kv["host_id"] = host_id;
    kv["is_ready"] = is_ready ? "1" : "0";  // 用 "1"/"0" 表示 bool
    return kv;
  }

  void from_map(std::map<std::string, std::string> const& kv) override {
    host_id = kv.at("host_id");
    std::string const& ready_str = kv.at("is_ready");
    is_ready = (ready_str == "1");  // 解析 "1" 为 true，其他为 false
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

class RedisExchanger {
 public:
  RedisExchanger(std::string const& host = "127.0.0.1", int port = 6379,
                 int timeout_ms = 2000);

  ~RedisExchanger();

  bool valid() const { return ctx_ != nullptr; }

  bool publish(std::string const& key, Exchangeable const& obj);
  bool fetch(std::string const& key, Exchangeable& obj);
  bool wait_and_fetch(std::string const& key, Exchangeable& obj,
                      int max_retries = 50, int delay_ms = 100);

 private:
  redisContext* ctx_;
};
