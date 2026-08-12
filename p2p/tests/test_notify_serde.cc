// Standalone test for the variable-length notification wire format in
// p2p/util/common.h. Compiles against the real header, no CUDA needed.
#include "util/common.h"
#include <cassert>
#include <cstdio>
#include <random>
#include <string>

static int failures = 0;
#define CHECK(cond, what)                                  \
  do {                                                     \
    if (!(cond)) {                                         \
      std::printf("FAIL: %s (line %d)\n", what, __LINE__); \
      ++failures;                                          \
    }                                                      \
  } while (0)

static void roundtrip(std::string const& name, std::string const& msg,
                      uint32_t msg_type) {
  NotifyMsg in{msg_type, name, msg};
  std::string wire = serialize_notify_msg(in);
  CHECK(wire.size() == NOTIFY_MSG_HDR_SIZE + name.size() + msg.size(),
        "wire size");
  NotifyMsg out;
  CHECK(deserialize_notify_msg(wire, out), "deserialize ok");
  CHECK(out.msg_type == msg_type, "msg_type preserved");
  CHECK(out.name == name, "name preserved");
  CHECK(out.msg == msg, "msg preserved");
}

int main() {
  // Basic round-trips across the sizes that matter: empty, the old 256-byte
  // cliff, the 16 KiB stopgap cliff, and a multi-megabyte payload.
  roundtrip("", "", 0);
  roundtrip("agent-a", "hello", 7);
  roundtrip("prefill-agent-0", std::string(255, 'x'), 0);
  roundtrip("prefill-agent-0", std::string(256, 'x'), 0);
  roundtrip("prefill-agent-0", std::string(305, 'x'), 0);  // the observed HB
  roundtrip("prefill-agent-0", std::string(16384, 'x'), 0);
  roundtrip("prefill-agent-0", std::string(16385, 'x'), 0);
  roundtrip("prefill-agent-0", std::string(8u << 20, 'x'), 0);
  // Embedded NULs and arbitrary bytes survive (payloads are opaque blobs).
  std::string blob =
      std::string("a\0b", 3) + "\xde\xad\xde\xad" + std::string(1000, '\xff');
  roundtrip(std::string("n\0m", 3), blob, 42);

  // A realistic vLLM heartbeat at high concurrency: 3072 request ids.
  {
    std::string hb = "HB:";
    for (int i = 0; i < 3072; ++i) {
      hb += "chatcmpl-3c1f9a2e-8b4d-4f7a-9c2e-";
      hb += std::to_string(100000 + i);
      hb += ",";
    }
    CHECK(hb.size() > 100000, "hb representative size");
    roundtrip("decode-agent-3", hb, 0);
  }

  // Rejection: truncated header, truncated body, oversized body, bad magic,
  // trailing garbage.
  {
    NotifyMsg m{0, "agent", std::string(400, 'y')};
    std::string wire = serialize_notify_msg(m);
    NotifyMsg out;
    CHECK(!deserialize_notify_msg(std::string(), out), "empty rejected");
    CHECK(!deserialize_notify_msg(wire.substr(0, 8), out),
          "truncated header rejected");
    CHECK(!deserialize_notify_msg(wire.substr(0, wire.size() - 1), out),
          "truncated body rejected");
    CHECK(!deserialize_notify_msg(wire + "x", out),
          "trailing garbage rejected");
    std::string bad = wire;
    bad[0] ^= 0x1;
    CHECK(!deserialize_notify_msg(bad, out), "bad magic rejected");
    // Length fields that overflow past the payload must be rejected (u32
    // arithmetic must not wrap).
    std::string wrap = wire;
    uint32_t huge = 0xFFFFFFF0u;
    std::memcpy(wrap.data() + 8, &huge, 4);
    CHECK(!deserialize_notify_msg(wrap, out), "wrapping name_len rejected");
    std::memcpy(wrap.data() + 8, &huge, 4);
    std::memcpy(wrap.data() + 12, &huge, 4);
    CHECK(!deserialize_notify_msg(wrap, out), "double-wrap rejected");
  }

  // Discrimination safety: metadata-exchange payloads must not parse as
  // notifications. Simulate payloads whose first bytes vary randomly —
  // acceptance requires magic AND exact length bookkeeping.
  {
    std::mt19937 rng(20260703);
    NotifyMsg out;
    int accepted = 0;
    for (int i = 0; i < 100000; ++i) {
      std::string p(32 + rng() % 64, '\0');
      for (auto& c : p) c = static_cast<char>(rng());
      if (deserialize_notify_msg(p, out)) ++accepted;
    }
    CHECK(accepted == 0, "random payloads never parse as notifications");
    // Even a payload that deliberately starts with the magic fails unless the
    // lengths match exactly.
    std::string spoof(64, '\0');
    uint32_t magic = NOTIFY_MSG_MAGIC;
    std::memcpy(spoof.data(), &magic, 4);
    CHECK(!deserialize_notify_msg(spoof, out),
          "magic-only spoof rejected (lengths inconsistent)");
  }

  if (failures == 0) {
    std::printf("OK: all notification serde tests passed\n");
    return 0;
  }
  std::printf("%d failure(s)\n", failures);
  return 1;
}
