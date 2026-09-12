#include "cc/swift.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

using uccl::swift::SwiftCC;

static void test_decrease_interval() {
  constexpr double kFreqGhz = 2.0;
  SwiftCC cc(kFreqGhz, 100e9 / 8);
  cc.last_decrease_tsc_ = 0;
  uint64_t now = 10000000;
  cc.adjust_wnd(1000, SwiftCC::kMSS, now);
  auto first_window = cc.get_wnd();
  assert(first_window < SwiftCC::kDefaultWnd);
  assert(cc.last_decrease_tsc_ == now);

  // ACKs in one burst must not repeatedly halve the window.
  for (uint64_t i = 1; i <= 32; ++i) {
    cc.adjust_wnd(1000, SwiftCC::kMSS, now + i);
    assert(cc.get_wnd() == first_window);
    assert(cc.last_decrease_tsc_ == now);
  }

  uint64_t interval =
      static_cast<uint64_t>(std::ceil(cc.rtt_ * kFreqGhz * 1000));
  assert(!cc.can_decrease(now + interval - 1));
  assert(cc.can_decrease(now + interval));
  now += interval;
  cc.adjust_wnd(1000, SwiftCC::kMSS, now);
  assert(cc.get_wnd() < first_window);
  assert(cc.last_decrease_tsc_ == now);
}

static void test_only_actual_decreases_reset_clock() {
  SwiftCC cc(2.0, 100e9 / 8);
  cc.last_decrease_tsc_ = 1000000;
  cc.swift_cwnd_ = SwiftCC::kMinCwnd * 16;
  auto before = cc.get_wnd();
  cc.adjust_wnd(1, before, 10000000);
  assert(cc.get_wnd() > before);
  assert(cc.last_decrease_tsc_ == 1000000);

  cc.swift_cwnd_ = SwiftCC::kMinCwnd;
  cc.adjust_wnd(1000, SwiftCC::kMSS, 20000000);
  assert(cc.get_wnd() == SwiftCC::kMinCwnd);
  assert(cc.last_decrease_tsc_ == 1000000);

  // Existing two-argument callers (including collective) remain supported.
  cc.adjust_wnd(1, SwiftCC::kMSS);
  assert(cc.get_wnd() >= SwiftCC::kMinCwnd);
  assert(cc.get_wnd() <= SwiftCC::kMaxCwnd);
}

int main() {
  test_decrease_interval();
  test_only_actual_decreases_reset_clock();
  std::cout << "Swift decrease interval tests passed\n";
}
