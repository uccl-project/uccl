#include "timely.h"

using namespace uccl;

static constexpr double kLinkBandwidth = 100.0 * 1024 * 1024 * 1024 / 8;

void test(size_t mean_rtt, size_t random_add_rtt) {
    double freq_ghz = measure_rdtsc_freq();
    Timely timely(freq_ghz, kLinkBandwidth);

    std::vector<double> sample_us;
    for (size_t i = 0; i < 2000; i++) {
        // double rtt_sample = 500 + (0.2 * i);
        double rtt_sample =
            mean_rtt + static_cast<size_t>(rand()) % random_add_rtt;
        sample_us.push_back(rtt_sample);
    }

    for (double rtt_us : sample_us) {
        timely.update_rate(rdtsc(), us_to_cycles(rtt_us, freq_ghz));
        nano_sleep(1000, freq_ghz);  // Update every one microsecond
    }

    printf("mean %zu us, random %zu us, tput %.2f Gbps\n", mean_rtt,
           random_add_rtt, timely.get_rate_gbps());
}

int main() {
    size_t random_add_rtt = 5;
    for (size_t iter = 0; iter < 20; iter++) {
        test(Timely::kTLow + iter * 10, random_add_rtt);
    }
}
