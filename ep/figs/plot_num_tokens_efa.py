import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Data
# ---------------------------
num_tokens = [128, 256, 512, 768, 1024]

# FP8 Throughput (GB/s)
fp8_dispatch_rdma = np.array([7.68, 11.07, 16.86, 20.65, 26.19])
fp8_dispatch_nvl = np.array([26.01, 36.79, 56.23, 66.90, 86.66])
fp8_combine_rdma = np.array([4.22, 3.66, 4.35, 5.27, 5.35])
fp8_combine_nvl = np.array([14.30, 12.17, 14.51, 17.07, 17.70])

# FP8 Aggregated throughput
fp8_dispatch_total = fp8_dispatch_rdma + fp8_dispatch_nvl
fp8_combine_total = fp8_combine_rdma + fp8_combine_nvl

# BF16 Latency (µs)
bf16_dispatch_lat = np.array([244.44, 339.77, 446.42, 547.75, 576.97])
bf16_combine_lat = np.array(
    [1992.00, 3355.00, 4164.00, 5478.00, np.nan]
)  # Added NaN to match 5 tokens

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
# plt.plot(num_tokens, fp8_dispatch_total, marker="^", label="FP8 Dispatch (Total)")
plt.plot(
    num_tokens,
    fp8_dispatch_rdma,
    marker="x",
    linestyle="--",
    label="FP8 Dispatch (RDMA)",
)
plt.plot(
    num_tokens,
    fp8_dispatch_nvl,
    marker="d",
    linestyle="--",
    label="FP8 Dispatch (NVLink)",
)
plt.xlabel("Number of Tokens")
plt.ylabel("Dispatch Throughput (GB/s)")
plt.title("FP8 Dispatch Throughput vs Tokens")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("fp8_dispatch_throughput_vs_tokens.png", dpi=300)
plt.show()

# ---------------------------
# Plot FP8 Combine Throughput
# ---------------------------
plt.figure(figsize=(8, 6))
# plt.plot(num_tokens, fp8_combine_total, marker="^", label="FP8 Combine (Total)")
plt.plot(
    num_tokens, fp8_combine_rdma, marker="x", linestyle="--", label="FP8 Combine (RDMA)"
)
plt.plot(
    num_tokens,
    fp8_combine_nvl,
    marker="d",
    linestyle="--",
    label="FP8 Combine (NVLink)",
)
plt.xlabel("Number of Tokens")
plt.ylabel("Combine Throughput (GB/s)")
plt.title("FP8 Combine Throughput vs Tokens")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("fp8_combine_throughput_vs_tokens.png", dpi=300)
plt.show()

# ---------------------------
# Plot BF16 Dispatch Latency
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(num_tokens, bf16_dispatch_lat, marker="o", label="BF16 Dispatch Latency")
plt.xlabel("Number of Tokens")
plt.ylabel("Dispatch Latency (µs)")
plt.title("BF16 Dispatch Latency vs Tokens")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("bf16_dispatch_latency_vs_tokens.png", dpi=300)
plt.show()

# ---------------------------
# Plot BF16 Combine Latency
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(num_tokens, bf16_combine_lat, marker="s", label="BF16 Combine Latency")
plt.xlabel("Number of Tokens")
plt.ylabel("Combine Latency (µs)")
plt.title("BF16 Combine Latency vs Tokens")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("bf16_combine_latency_vs_tokens.png", dpi=300)
plt.show()
