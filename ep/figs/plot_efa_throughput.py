import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Data (Throughput in GB/s)
# ---------------------------
gpus = [16, 24, 32]  # EP=16, 24, 32 corresponds to 2, 3, 4 nodes/GPUs

# Sparse mode (UCCL (Low Latency), Pplx)
uccl_dispatch = [2.788, 1.131, 0.7224]
uccl_combine = [4.249, 1.941, 1.325]
pplx_dispatch = [1.027, 0.809, 0.693]
pplx_combine = [1.840, 1.324, 1.098]

# Normal mode (top = RDMA, bottom = NVLink)
# normal_mode_dispatch_rdma = np.array([13.01, 14.19, 15.67])
# normal_mode_dispatch_nvl = np.array([42.19, 35.18, 33.45])
normal_mode_dispatch_rdma = np.array([8.03, 7.95, 9.49])
normal_mode_dispatch_nvl = np.array([25.09, 18.71, 18.73])

normal_mode_combine_rdma = np.array([4.38, 4.44, 10.34])
normal_mode_combine_nvl = np.array([14.20, 11.01, 22.07])

# Aggregate: RDMA + NVLink throughput
normal_mode_dispatch = np.minimum(normal_mode_dispatch_rdma, normal_mode_dispatch_nvl)
normal_mode_combine = np.minimum(normal_mode_combine_rdma, normal_mode_combine_nvl)

# ---------------------------
# Data (Latency in µs)
# ---------------------------
uccl_dispatch_lat = [2694, 6643, 10399]
uccl_combine_lat = [3421, 7491, 10968]
pplx_dispatch_lat = [7446, 9363, 10924]
pplx_combine_lat = [9841, 11095, 13380]

# Normal mode latencies (averaged across RDMA/NVLink)
# normal_mode_dispatch_lat_rdma = np.array([282.04, 366.80, 419.98])
# normal_mode_combine_lat_rdma = np.array([238.02, 1172.00, 636.69])

normal_mode_dispatch_lat = np.array([282.04, 366.80, 419.98])
normal_mode_combine_lat = np.array([238.02, 1172.00, 636.69])

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
# Plot Dispatch Throughput
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(gpus, uccl_dispatch, marker="o", label="UCCL (Low Latency)")
plt.plot(gpus, pplx_dispatch, marker="s", label="Pplx")
plt.plot(gpus, normal_mode_dispatch, marker="^", label="UCCL (Normal)")
plt.xlabel("Number of GPUs")
plt.ylabel("Dispatch Throughput (GB/s)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("throughput_dispatch_vs_gpus.png", dpi=300)
plt.show()

# ---------------------------
# Plot Combine Throughput
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(gpus, uccl_combine, marker="o", label="UCCL (Low Latency)")
plt.plot(gpus, pplx_combine, marker="s", label="Pplx")
plt.plot(gpus, normal_mode_combine, marker="^", label="UCCL (Normal)")
plt.xlabel("Number of GPUs")
plt.ylabel("Combine Throughput (GB/s)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("throughput_combine_vs_gpus.png", dpi=300)
plt.show()

# ---------------------------
# Plot Dispatch Latency
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(gpus, uccl_dispatch_lat, marker="o", label="UCCL (Low Latency)")
plt.plot(gpus, pplx_dispatch_lat, marker="s", label="Pplx")
plt.plot(gpus, normal_mode_dispatch_lat, marker="^", label="UCCL (Normal)")
plt.xlabel("Number of GPUs")
plt.ylabel("Dispatch Latency (µs)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("latency_dispatch_vs_gpus.png", dpi=300)
plt.show()

# ---------------------------
# Plot Combine Latency
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(gpus, uccl_combine_lat, marker="o", label="UCCL (Low Latency)")
plt.plot(gpus, pplx_combine_lat, marker="s", label="Pplx")
plt.plot(gpus, normal_mode_combine_lat, marker="^", label="UCCL (Normal)")
plt.xlabel("Number of GPUs")
plt.ylabel("Combine Latency (µs)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("latency_combine_vs_gpus.png", dpi=300)
plt.show()
