# UCCL GPU-Driven Expert-parallelism Engine

For UCCL's host/CPU-driven P2P engine, see [p2p](../p2p/) folder.

## Install

Installing `ep` as a Python package:
```bash
./build_and_install.sh cuda ep 3.11
```
Alternatively, 
```bash
make -j install
```

## Benchmark
In `bench` folder:

```bash
# On **sender** node (rank 0)
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=<master_ip> --master_port=<master_port> \
  benchmark_remote.py

# On **receiver** node (rank 1)
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=<master_ip> --master_port=<master_port> \
  benchmark_remote.py
```

Running DeepEP-style dispatch / combine tests:
```bash
# On **sender** node (rank 0)
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=10.1.209.224 --master_port=12356 \
  bench/test_internode_simple.py

# On **receiver** node (rank 1)
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=10.1.209.224 --master_port=12356 \
  bench/test_internode_simple.py
```
