#include <gtest/gtest.h>
#include <time.h>

#include <algorithm>
#include <vector>

#define private public
#include "timing_wheel.h"

using namespace uccl;

#define test_printf(...)         \
    do {                         \
        printf("[          ] "); \
        printf(__VA_ARGS__);     \
        fflush(stderr);          \
        fflush(stdout);          \
    } while (0)

static constexpr size_t kTestPktSize = 1024;
static constexpr size_t kTestNumPkts = 8000;

using namespace std::placeholders;

class TimingWheelTest : public ::testing::Test {
   public:
    TimingWheelTest() {
        timing_wheel_args_t args;
        args.freq_ghz_ = measure_rdtsc_freq();
        wheel_ = new TimingWheel(args);

        freq_ghz_ = measure_rdtsc_freq();
    }

    ~TimingWheelTest() {}

    TimingWheel *wheel_;
    double freq_ghz_;
};

TEST_F(TimingWheelTest, Basic) {
    // Empty wheel
    wheel_->reap(rdtsc());
    ASSERT_EQ(wheel_->ready_queue_.size(), 0);

    // One entry. Check that it's eventually sent.
    size_t ref_tsc = rdtsc();
    size_t abs_tx_tsc = ref_tsc + wheel_->wslot_width_tsc_;
    wheel_->insert(TimingWheel::get_dummy_ent(), ref_tsc, abs_tx_tsc);

    wheel_->reap(abs_tx_tsc + wheel_->wslot_width_tsc_);
    ASSERT_EQ(wheel_->ready_queue_.size(), 1);
}

// This is not a fixture test because we use a different wheel for each rate
TEST(TimingWheelRateTest, RateTest) {
    const std::vector<double> target_gbps = {1.0, 5.0, 10.0, 20.0, 40.0, 80.0};
    const double freq_ghz = measure_rdtsc_freq();

    for (size_t iters = 0; iters < target_gbps.size(); iters++) {
        const double target_rate = Timely::gbps_to_rate(target_gbps[iters]);
        test_printf("Target rate = %.2f Gbps\n", target_gbps[iters]);
        const double ns_per_pkt = 1000000000 * (kTestPktSize / target_rate);
        const size_t cycles_per_pkt = std::ceil(freq_ghz * ns_per_pkt);

        TscTimer rate_timer;

        // Create the a new wheel so we automatically clean up extra packets
        // from each iteration
        timing_wheel_args_t args;
        args.freq_ghz_ = freq_ghz;

        TimingWheel wheel(args);

        // Update the wheel and start measurement
        wheel.catchup();
        rate_timer.start();

        size_t abs_tx_tsc = rdtsc();  // TX tsc for this session

        // Send one window
        size_t ref_tsc = rdtsc();
        for (size_t i = 0; i < kSessionCredits; i++) {
            abs_tx_tsc = std::max(ref_tsc, abs_tx_tsc + cycles_per_pkt);
            wheel.insert(TimingWheel::get_dummy_ent(), ref_tsc, abs_tx_tsc);
        }

        size_t num_pkts_sent = 0;
        while (num_pkts_sent < kTestNumPkts) {
            size_t cur_tsc = rdtsc();
            wheel.reap(cur_tsc);

            size_t num_ready = wheel.ready_queue_.size();
            assert(num_ready <= kSessionCredits);

            if (num_ready > 0) {
                num_pkts_sent += num_ready;

                for (size_t i = 0; i < num_ready; i++) wheel.ready_queue_.pop();

                // Send more packets
                ref_tsc = rdtsc();
                for (size_t i = 0; i < num_ready; i++) {
                    abs_tx_tsc = std::max(ref_tsc, abs_tx_tsc + cycles_per_pkt);
                    wheel.insert(TimingWheel::get_dummy_ent(), ref_tsc,
                                 abs_tx_tsc);
                }
            }
        }

        rate_timer.stop();
        double seconds = rate_timer.avg_sec(freq_ghz);
        double achieved_rate = num_pkts_sent * kTestPktSize / seconds;
        test_printf("Achieved rate = %.2f Gbps\n",
                    Timely::rate_to_gbps(achieved_rate));
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
