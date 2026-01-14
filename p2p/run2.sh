# Run all tests on node2

ITERATIONS=50

# SEND/RECV
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=216.128.145.174 benchmarks/benchmark_uccl.py --num-kvblocks=1 --iters=${ITERATIONS}

# SEND/RECV Async
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=216.128.145.174 benchmarks/benchmark_uccl.py --num-kvblocks=1 --iters=${ITERATIONS} --async-api

# SEND/RECV (vector)
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=216.128.145.174 benchmarks/benchmark_uccl.py --num-kvblocks=8 --iters=${ITERATIONS}

# SEND/RECV Async (vector)
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=216.128.145.174 benchmarks/benchmark_uccl.py --num-kvblocks=8 --iters=${ITERATIONS} --async-api

# WRITE
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=216.128.145.174 benchmarks/benchmark_uccl_readwrite.py --mode=write --num-iovs=1 --iters=${ITERATIONS}

# WRITE Async
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=216.128.145.174 benchmarks/benchmark_uccl_readwrite.py --mode=write --num-iovs=1 --async-api --iters=${ITERATIONS}

# WRITE (vector)
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=216.128.145.174 benchmarks/benchmark_uccl_readwrite.py --mode=write --num-iovs=8 --iters=${ITERATIONS}

# WRITE Async (vector)
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=216.128.145.174 benchmarks/benchmark_uccl_readwrite.py --mode=write --num-iovs=8 --async-api --iters=${ITERATIONS}

# READ
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=216.128.145.174 benchmarks/benchmark_uccl_readwrite.py --mode=read --num-iovs=1 --iters=${ITERATIONS}

# READ Async
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=216.128.145.174 benchmarks/benchmark_uccl_readwrite.py --mode=read --num-iovs=1 --async-api --iters=${ITERATIONS}

# READ (vector)
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=216.128.145.174 benchmarks/benchmark_uccl_readwrite.py --mode=read --num-iovs=8 --iters=${ITERATIONS}

# READ Async (vector)
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=216.128.145.174 benchmarks/benchmark_uccl_readwrite.py --mode=read --num-iovs=8 --async-api --iters=${ITERATIONS}