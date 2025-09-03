#pragma once

#include <map>
#include <string>

// for redis
#include <hiredis/hiredis.h>
#include <string>
#include <map>
#include <thread>
#include <chrono>
#include <iostream>

// for RDMAConnectionInfo
#include <sstream>
#include <iomanip>

struct Exchangeable {
  virtual std::map<std::string, std::string> to_map() const = 0;
  virtual void from_map(const std::map<std::string, std::string>& kv) = 0;
  virtual ~Exchangeable() {}
};

struct CommunicatorMeta : public Exchangeable {
    std::string host_id;
    bool is_ready;

    CommunicatorMeta() = default;

    std::map<std::string, std::string> to_map() const override {
        std::map<std::string, std::string> kv;
        kv["host_id"]  = host_id;
        kv["is_ready"] = is_ready ? "1" : "0";  // 用 "1"/"0" 表示 bool
        return kv;
    }

    void from_map(const std::map<std::string, std::string>& kv) override {
        host_id = kv.at("host_id");
        const std::string& ready_str = kv.at("is_ready");
        is_ready = (ready_str == "1");  // 解析 "1" 为 true，其他为 false
    }
};

struct RDMAConnectionInfo : public Exchangeable {
  uint32_t qp_num;
  uint32_t rkey;
  uintptr_t addr;
  unsigned char gid[16];

  RDMAConnectionInfo() = default;

  std::map<std::string, std::string> to_map() const override {
    std::map<std::string, std::string> kv;
    kv["qp_num"] = std::to_string(qp_num);
    kv["rkey"]   = std::to_string(rkey);
    kv["addr"]   = std::to_string(addr);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
      oss << std::setw(2) << static_cast<int>(gid[i]);
    }
    kv["gid"] = oss.str();

    return kv;
  }

  void from_map(const std::map<std::string, std::string>& kv) override {
    qp_num = std::stoul(kv.at("qp_num"));
    rkey   = std::stoul(kv.at("rkey"));
    addr   = std::stoull(kv.at("addr"));

    const std::string& hex = kv.at("gid");
    for (int i = 0; i < 16; ++i) {
      unsigned int byte;
      std::stringstream ss;
      ss << std::hex << hex.substr(i*2, 2);
      ss >> byte;
      gid[i] = static_cast<unsigned char>(byte);
    }
  }
};


class RedisExchanger {
public:
  RedisExchanger(const std::string& host="127.0.0.1", int port=6379, int timeout_ms=2000);

  ~RedisExchanger();

  bool valid() const { return ctx_ != nullptr; }

  bool publish(const std::string& key, const Exchangeable& obj);
  bool fetch(const std::string& key, Exchangeable& obj);
  bool wait_and_fetch(const std::string& key, Exchangeable& obj,
                      int max_retries=50, int delay_ms=100);

private:
  redisContext* ctx_;
};
