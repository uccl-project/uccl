import re
import matplotlib.pyplot as plt
import numpy as np

# Path to the perf_comparison file
PERF_FILE = "perf_comparison"

# Patterns to match result lines
UCCL_PATTERN = re.compile(
    r"\[Client\]\s+([\d.]+)\s*([A-Za-z]+) :\s+([\d.]+) Gbps \|\s+([\d.]+) GB/s\s+\| ([\d.]+) s"
)
NIXL_PATTERN = re.compile(
    r"\[client\]\s+([\d.]+)\s*([A-Za-z]+) :\s+([\d.]+) Gbps \|\s+([\d.]+) GB/s \| ([\d.]+) s"
)


# Helper to convert size to bytes
def size_to_bytes(size, unit):
    unit = unit.upper()
    if unit == "B":
        return float(size)
    elif unit == "KB":
        return float(size) * 1024
    elif unit == "MB":
        return float(size) * 1024 * 1024
    elif unit == "GB":
        return float(size) * 1024 * 1024 * 1024
    else:
        raise ValueError(f"Unknown unit: {unit}")


def parse_results(lines, pattern):
    results = []
    for line in lines:
        m = pattern.search(line)
        if m:
            size, unit, gbps, gbs, t = m.groups()
            size_bytes = size_to_bytes(size, unit)
            results.append(
                {
                    "size_bytes": size_bytes,
                    "size_str": f"{size} {unit}",
                    "gbps": float(gbps),
                    "gbs": float(gbs),
                    "time": float(t),
                }
            )
    return results


def extract_blocks(lines):
    blocks = {}
    current = None
    block_lines = []
    for line in lines:
        if line.strip().startswith("UCCL:"):
            if current and block_lines:
                blocks[current] = block_lines
            current = "UCCL"
            block_lines = []
        elif line.strip().startswith("NIXL"):
            if current and block_lines:
                blocks[current] = block_lines
            current = "NIXL"
            block_lines = []
        elif line.strip() == "":
            continue
        else:
            if current:
                block_lines.append(line)
    if current and block_lines:
        blocks[current] = block_lines
    return blocks


def get_all_result_blocks(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    # Find all UCCL/NIXL result blocks (host and GPU)
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "UCCL:":
            block = {"type": "UCCL", "start": i}
            # Find next NIXL or end
            j = i + 1
            while (
                j < len(lines)
                and not lines[j].strip().startswith("NIXL")
                and not lines[j].strip().startswith("UCCL:")
            ):
                j += 1
            block["end"] = j
            blocks.append(block)
            i = j
        elif lines[i].strip().startswith("NIXL"):
            block = {"type": "NIXL", "start": i}
            j = i + 1
            while (
                j < len(lines)
                and not lines[j].strip().startswith("UCCL:")
                and not lines[j].strip().startswith("NIXL")
            ):
                j += 1
            block["end"] = j
            blocks.append(block)
            i = j
        else:
            i += 1
    return blocks, lines


def main():
    blocks, lines = get_all_result_blocks(PERF_FILE)
    # There are two UCCL and two NIXL blocks: first is host, second is GPU
    results = {"host": {}, "gpu": {}}
    block_types = []
    for block in blocks:
        block_types.append(block["type"])
    # Assign blocks to host/gpu
    if block_types.count("UCCL") >= 2 and block_types.count("NIXL") >= 2:
        # First UCCL/NIXL: host, second: gpu
        host_blocks = [b for b in blocks if b["type"] == "UCCL"][:1] + [
            b for b in blocks if b["type"] == "NIXL"
        ][:1]
        gpu_blocks = [b for b in blocks if b["type"] == "UCCL"][1:2] + [
            b for b in blocks if b["type"] == "NIXL"
        ][1:2]
        # Parse host
        results["host"]["UCCL"] = parse_results(
            lines[host_blocks[0]["start"] : host_blocks[0]["end"]], UCCL_PATTERN
        )
        results["host"]["NIXL"] = parse_results(
            lines[host_blocks[1]["start"] : host_blocks[1]["end"]], NIXL_PATTERN
        )
        # Parse gpu
        results["gpu"]["UCCL"] = parse_results(
            lines[gpu_blocks[0]["start"] : gpu_blocks[0]["end"]], UCCL_PATTERN
        )
        results["gpu"]["NIXL"] = parse_results(
            lines[gpu_blocks[1]["start"] : gpu_blocks[1]["end"]], NIXL_PATTERN
        )
    else:
        print("Could not find both host and gpu blocks for UCCL and NIXL.")
        return
    # Plot
    for memtype in ["host", "gpu"]:
        plt.figure(figsize=(8, 6))
        for label, color in [("UCCL", "tab:blue"), ("NIXL", "tab:orange")]:
            data = results[memtype][label]
            sizes = [d["size_bytes"] for d in data]
            gbs = [d["gbs"] for d in data]
            plt.plot(sizes, gbs, marker="o", label=label, color=color)
        plt.xscale("log")
        plt.xlabel("Message Size (bytes)")
        plt.ylabel("Bandwidth (GB/s)")
        plt.title(f"Bandwidth vs Message Size ({memtype.capitalize()} Memory)")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"perf_comparison_{memtype}.png")
        print(f"Saved plot: perf_comparison_{memtype}.png")


if __name__ == "__main__":
    main()
