import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Data
# ---------------------------
num_sms = [6, 12, 18, 24, 32]

# FP8 Throughput (GB/s)
fp8_dispatch_rdma = np.array([7.00, 5.91, 7.00, 7.06, 5.91])
fp8_dispatch_nvl = np.array([22.70, 19.15, 22.71, 22.88, 19.15])
fp8_combine_rdma = np.array([3.13, 2.03, 2.99, 4.79, 4.51])
fp8_combine_nvl = np.array([10.16, 6.57, 9.68, 15.52, 14.62])

# FP8 Aggregated throughput
fp8_dispatch_total = fp8_dispatch_rdma + fp8_dispatch_nvl
fp8_combine_total = fp8_combine_rdma + fp8_combine_nvl

# BF16 Latency (µs)
bf16_dispatch_lat = np.array([270.32, 320.34, 270.16, 268.21, 320.46])
bf16_combine_lat = np.array([1171.00, 1811.00, 1229.00, 766.59, 814.01])

# ---------------------------
# Plot settings
# ---------------------------
plt.rcParams.update(
    {
        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "lines.linewidth": 3,
        "lines.markersize": 10,
    }
)

# ---------------------------
# Plot FP8 Dispatch Throughput
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(num_sms, fp8_dispatch_total, marker="^", label="FP8 Dispatch (Total)")
plt.plot(
    num_sms, fp8_dispatch_rdma, marker="x", linestyle="--", label="FP8 Dispatch (RDMA)"
)
plt.plot(
    num_sms, fp8_dispatch_nvl, marker="d", linestyle="--", label="FP8 Dispatch (NVLink)"
)
plt.xlabel("Number of SMs")
plt.ylabel("Dispatch Throughput (GB/s)")
plt.title("FP8 Dispatch Throughput vs SMs")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("fp8_dispatch_throughput_vs_sms.png", dpi=300)
plt.show()

# ---------------------------
# Plot FP8 Combine Throughput
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(num_sms, fp8_combine_total, marker="^", label="FP8 Combine (Total)")
plt.plot(
    num_sms, fp8_combine_rdma, marker="x", linestyle="--", label="FP8 Combine (RDMA)"
)
plt.plot(
    num_sms, fp8_combine_nvl, marker="d", linestyle="--", label="FP8 Combine (NVLink)"
)
plt.xlabel("Number of SMs")
plt.ylabel("Combine Throughput (GB/s)")
plt.title("FP8 Combine Throughput vs SMs")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("fp8_combine_throughput_vs_sms.png", dpi=300)
plt.show()

# ---------------------------
# Plot BF16 Dispatch Latency
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(num_sms, bf16_dispatch_lat, marker="o", label="BF16 Dispatch Latency")
plt.xlabel("Number of SMs")
plt.ylabel("Dispatch Latency (µs)")
plt.title("BF16 Dispatch Latency vs SMs")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("bf16_dispatch_latency_vs_sms.png", dpi=300)
plt.show()

# ---------------------------
# Plot BF16 Combine Latency
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(num_sms, bf16_combine_lat, marker="s", label="BF16 Combine Latency")
plt.xlabel("Number of SMs")
plt.ylabel("Combine Latency (µs)")
plt.title("BF16 Combine Latency vs SMs")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("bf16_combine_latency_vs_sms.png", dpi=300)
plt.show()
