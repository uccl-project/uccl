import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-l", action="store_true", help="Run latency mode")
parser.add_argument("-b", action="store_true", help="Run burst mode")
parser.add_argument("-r", action="store_true", help="Run random mode")
parser.add_argument("-c", action="store_true", help="Run mops-controlled mode")
args = parser.parse_args()

if args.l:
    cmd = ["./benchmark", "-l"]
elif args.b:
    cmd = ["./benchmark", "-b"]
elif args.r:
    cmd = ["./benchmark", "-r"]
elif args.e:
    cmd = ["./benchmark", "-c"]
else:
    cmd = ["./benchmark"]

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

print(f"Running: {local_rank}/{world_size} {cmd}")

if local_rank == 0:
    subprocess.run(cmd, check=True)
else:
    with open("/dev/null", "w") as devnull:
        subprocess.run(cmd, check=True, stdout=devnull, stderr=devnull)
