#include "oob.h"

RedisExchanger::RedisExchanger(std::string const& host, int port,
                               int timeout_ms) {
#ifdef USE_REDIS_OOB
  struct timeval timeout;
  timeout.tv_sec = timeout_ms / 1000;
  timeout.tv_usec = (timeout_ms % 1000) * 1000;
  ctx_ = redisConnectWithTimeout(host.c_str(), port, timeout);
  if (!ctx_ || ctx_->err) {
    if (ctx_) {
      std::cerr << "Redis connection error: " << ctx_->errstr << std::endl;
      redisFree(ctx_);
    } else {
      std::cerr << "Redis allocation error" << std::endl;
    }
    ctx_ = nullptr;
  }
#else
#endif
}

RedisExchanger::~RedisExchanger() {
#ifdef USE_REDIS_OOB
  if (ctx_) redisFree(ctx_);
#endif
}

bool RedisExchanger::valid() const {
#ifdef USE_REDIS_OOB
  return ctx_ != nullptr;
#else
  return false;
#endif
}

bool RedisExchanger::publish(std::string const& key, Exchangeable const& obj) {
#ifdef USE_REDIS_OOB
  if (!ctx_) return false;

  auto kv = obj.to_map();
  std::string cmd = "HSET " + key;
  for (auto& p : kv) {
    cmd += " " + p.first + " " + p.second;
  }

  redisReply* reply = (redisReply*)redisCommand(ctx_, cmd.c_str());
  if (!reply) {
    std::cerr << "Redis publish failed: " << ctx_->errstr << std::endl;
    return false;
  }
  freeReplyObject(reply);
  return true;
#else
  return false;
#endif
}

bool RedisExchanger::fetch(std::string const& key, Exchangeable& obj) {
#ifdef USE_REDIS_OOB
  if (!ctx_) return false;

  redisReply* reply =
      (redisReply*)redisCommand(ctx_, "HGETALL %s", key.c_str());
  if (!reply) {
    std::cerr << "Redis fetch failed: " << ctx_->errstr << std::endl;
    return false;
  }

  if (reply->type != REDIS_REPLY_ARRAY || reply->elements == 0) {
    freeReplyObject(reply);
    return false;  // key does not exist
  }

  std::map<std::string, std::string> kv;
  for (size_t i = 0; i + 1 < reply->elements; i += 2) {
    std::string field(reply->element[i]->str, reply->element[i]->len);
    std::string value(reply->element[i + 1]->str, reply->element[i + 1]->len);
    kv[field] = value;
  }

  obj.from_map(kv);
  freeReplyObject(reply);
  return true;
#else
  return false;
#endif
}

bool RedisExchanger::wait_and_fetch(std::string const& key, Exchangeable& obj,
                                    int max_retries, int delay_ms) {
#ifdef USE_REDIS_OOB
  if (max_retries > 0) {
    for (int i = 0; i < max_retries; ++i) {
      if (fetch(key, obj)) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
  } else {  // -1
    while (true) {
      if (fetch(key, obj)) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
  }
  return false;
#else
  return false;
#endif
}
