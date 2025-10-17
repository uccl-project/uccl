import re
import matplotlib.pyplot as plt

log_file = "node1_dispatch.log"

# --- Regex patterns ---
tuning_re = re.compile(
    r"\[tuning\s+(dispatch|combine)\].*BW:\s*([\d.]+)\s*GB/s\s*\(RDMA\)",
    re.IGNORECASE)
post_re = re.compile(
    r"\[\d+\.\d+s\s*\|\s*\+([\d.]+)ms\].*\[post_rdma_async_batched\].*cmd\.bytes:\s*(\d+)",
    re.IGNORECASE)

# --- Storage ---
bw_dispatch, bytes_dispatch = [], []
bw_combine, bytes_combine = [], []

# --- State ---
current_type = None
current_bw = None
current_bytes = []

with open(log_file, "r") as f:
    for line in f:
        t_match = tuning_re.search(line)
        p_match = post_re.search(line)

        if t_match:
            # Finalize previous block
            if current_bw is not None and current_bytes:
                avg_bytes_kb = sum(current_bytes) / len(current_bytes) / 1024.0
                if current_type == "dispatch":
                    bw_dispatch.append(current_bw)
                    bytes_dispatch.append(avg_bytes_kb)
                elif current_type == "combine":
                    bw_combine.append(current_bw)
                    bytes_combine.append(avg_bytes_kb)

            # Start new block
            current_type = t_match.group(1).lower()
            current_bw = float(t_match.group(2))
            current_bytes = []

        elif p_match and current_bw is not None:
            bytes_sent = int(p_match.group(2))
            current_bytes.append(bytes_sent)

# Finalize last block
if current_bw is not None and current_bytes:
    avg_bytes_kb = sum(current_bytes) / len(current_bytes) / 1024.0
    if current_type == "dispatch":
        bw_dispatch.append(current_bw)
        bytes_dispatch.append(avg_bytes_kb)
    elif current_type == "combine":
        bw_combine.append(current_bw)
        bytes_combine.append(avg_bytes_kb)

# --- Scatter plot ---
plt.figure(figsize=(8, 5))
plt.scatter(bw_dispatch, bytes_dispatch, s=80, alpha=0.8, edgecolors="black",
            color="royalblue", label="dispatch")
plt.scatter(bw_combine, bytes_combine, s=80, alpha=0.8, edgecolors="black",
            color="red", label="combine")
plt.xlabel("RDMA Bandwidth (GB/s)")
plt.ylabel("Average cmd.bytes (KB)")
plt.title("Average RDMA Command Size vs RDMA Bandwidth")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("debug_msg_size.png", dpi=200)
plt.show()