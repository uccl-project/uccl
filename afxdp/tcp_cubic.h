#include "transport_config.h"
#include "util/timer.h"
#include <glog/logging.h>
#include <chrono>
#include <cmath>
#include <iostream>

using namespace uccl;

struct CubicCC {
  CubicCC()
      : max_cwnd(1024.0),
        cwnd(1.0),
        ssthresh(64.0),
        C(0.4),
        beta(0.7),
        last_max_cwnd(1.0),
        epoch_start(0),
        last_update_time(rdtsc()) {}
  void init(uint32_t _max_cwnd) { max_cwnd = _max_cwnd; }

  /* https://book.systemsapproach.org/congestion/tcpcc.html */
  inline void on_ack_received(
      uint32_t distance = 1) {  // Distance: number of packets acknowledged
    auto now = rdtsc();
    auto duration = to_usec(now - last_update_time, freq_ghz);

    double time_since_epoch =
        epoch_start + (duration / 1'000'000.0);  // seconds

    if (cwnd < ssthresh) {
      // Slow start phase
      cwnd += distance;  // Increase proportional to packets acknowledged
    } else {
      // Congestion avoidance using CUBIC
      if (epoch_start == 0) {
        epoch_start = time_since_epoch;
        K = cbrt((last_max_cwnd * (1 - beta)) /
                 C);  // Calculate the inflection point K
      }

      double time_diff = time_since_epoch - epoch_start;
      double cubic_target =
          C * pow(time_diff - K, 3) + last_max_cwnd;  // CUBIC formula

      if (cubic_target > cwnd) {
        cwnd += (cubic_target - cwnd);  // Adjust to target
      } else {
        cwnd +=
            (distance * (1.0 / cwnd));  // Scale additive increase by distance
      }
    }
    cwnd = std::min(cwnd, max_cwnd);  // Cap cwnd at max_cwnd

    last_update_time = now;
    VLOG(3) << "ACK received: distance=" << distance << ", cwnd=" << cwnd
            << ", ssthresh=" << ssthresh << std::endl;
  }

  inline void on_packet_loss() {
    last_max_cwnd = cwnd;
    ssthresh = cwnd * beta;  // Apply multiplicative decrease factor
    // Adjust cwnd based on last_max_cwnd and beta, similar to Linux
    // behavior
    cwnd =
        std::max(1.0,
                 last_max_cwnd *
                     (1 - beta));  // Avoid cwnd falling below a minimum of 1.0
    epoch_start = 0;
    VLOG(3) << "Packet loss detected: cwnd=" << cwnd
            << ", ssthresh=" << ssthresh << std::endl;
  }

  inline double get_cwnd() const { return cwnd; }

  double max_cwnd;            // Maximum congestion window size
  double cwnd;                // Congestion window size
  double ssthresh;            // Slow start threshold
  double C;                   // CUBIC scaling constant
  double beta;                // Multiplicative decrease factor
  double last_max_cwnd;       // Last maximum congestion window
  double epoch_start;         // Start time of the current epoch
  uint64_t last_update_time;  // Last update time
  double K;                   // Inflection point in CUBIC formula
};