"""Plot compressed vs non-compressed RDMA WRITE throughput across message sizes.

Bench: benchmark_uccl_readwrite_random.py with --mode write, random bf16 payload.
"""

import matplotlib.pyplot as plt

sizes_mb = [
    4 / 1024,          # 4 KB
    16 / 1024,         # 16 KB
    64 / 1024,         # 64 KB
    256 / 1024,        # 256 KB
    1.0,
    10.0,
    16.0,
    32.0,
    64.0,
    100.0,
]
size_labels = ["4 KB", "16 KB", "64 KB", "256 KB", "1 MB", "10 MB",
               "16 MB", "32 MB", "64 MB", "100 MB"]

compressed = [
    0.51, 1.53, 5.49, 7.28, 34.48, 47.39, 64.15, 58.78, 60.59, 68.88,
]
no_compress = [
    0.42, 1.26, 4.89, 15.09, 32.98, 47.24, 48.14, 48.78, 49.18, 48.69,
]

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(sizes_mb, no_compress, "o-", label="Non-compressed (baseline)",
        color="#1f77b4", linewidth=2, markersize=8)
ax.plot(sizes_mb, compressed, "s-", label="Compressed (kSplitOnly, BF16 random)",
        color="#d62728", linewidth=2, markersize=8)

ax.set_xscale("log")
ax.set_xticks(sizes_mb)
ax.set_xticklabels(size_labels, rotation=30, ha="right")
ax.set_xlabel("Message size", fontsize=11)
ax.set_ylabel("Effective throughput (GB/s)", fontsize=11)
ax.set_title("UCCL P2P RDMA WRITE: compressed vs non-compressed throughput\n"
             "(BF16 uniform random in [-1, 1))",
             fontsize=12)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)
ax.set_ylim(bottom=0, top=80)

plt.tight_layout()
out_path = "/home/uccl/nfs/shuangma/uccl/p2p/benchmarks/compress_vs_nocompress.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"Saved: {out_path}")
