import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", action="store_true", help="Run latency mode")
parser.add_argument("-b", action="store_true", help="Run burst mode")
args = parser.parse_args()

if args.l:
    cmd = ["./benchmark_fifo", "-l"]
elif args.b:
    cmd = ["./benchmark_fifo", "-b"]
else:
    cmd = ["./benchmark_fifo"]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
