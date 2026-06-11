# UCCL-GIN Standalone

This directory is the clean standalone home for UCCL-GIN rail primitives.
It is intentionally outside `ep/` so the original UCCL EP path can keep running
unchanged.

Current scope:

- `put<Rail>`
- `red_add_rel<Rail>`
- `put_tail_add<Rail>`
- `quiet`
- C++/MPI microbench with correctness and payload-before-tail checks
- Minimal Python/C++ `Context` binding for host setup/teardown smoke
- Python pytest wrappers for context smoke and standalone microbench

Not in scope yet:

- DeepEP integration
- `Lsa`/NVLink forwarding

## Build

```bash
cd experimental/uccl_gin
make CUDA_HOME=/usr/local/cuda-13.0 PYTHON=$VIRTUAL_ENV/bin/python SM=90 -j
```

The build compiles copied transport sources from `experimental/uccl_gin/src`.
It must not link `ep/src/*.o`.

## Two-node smoke

Use the OpenMPI launcher that matches the linked MPI library. On AWS DLAMI this
has been `/opt/amazon/openmpi/bin/mpirun`.

```bash
export LD_LIBRARY_PATH=/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-master/lib:/opt/amazon/efa/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export NCCL_NET_PLUGIN=ofi
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export OFI_NCCL_FORCE_NUM_RAILS=2
export NCCL_SOCKET_IFNAME=enp71s0
export LOCAL_WORLD_SIZE=8

/opt/amazon/openmpi/bin/mpirun \
  --host <node0>,<node1> -np 16 -npernode 8 \
  -x LD_LIBRARY_PATH -x NCCL_NET_PLUGIN -x FI_PROVIDER \
  -x FI_EFA_USE_DEVICE_RDMA -x OFI_NCCL_FORCE_NUM_RAILS \
  -x NCCL_SOCKET_IFNAME -x LOCAL_WORLD_SIZE \
  ./build/uccl_gin_microbench --sizes 1024,4096,65536,1048576 --iters 20
```

The microbench reports bandwidth only after correctness passes.

## Python Package Entries

The Python package has two test-facing entries:

- `python -m uccl_gin.context_smoke` creates and destroys a real C++ `Context`
  through the `_uccl_gin` extension.
- `python -m uccl_gin.microbench` drives the standalone binary and checks
  correctness markers.

```bash
export PYTHONPATH=$PWD/experimental/uccl_gin/python
export UCCL_GIN_ROOT=$PWD/experimental/uccl_gin
export UCCL_GIN_MPI_HOSTS=<node0>:8,<node1>:8
export LOCAL_WORLD_SIZE=8

python -m uccl_gin.microbench
```

Context smoke:

```bash
/opt/amazon/openmpi/bin/mpirun \
  --host <node0>:8,<node1>:8 -np 16 -npernode 8 \
  -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_SOCKET_IFNAME -x LOCAL_WORLD_SIZE \
  /usr/bin/python3 -m uccl_gin.context_smoke
```

The test files use the same package APIs:

```bash
UCCL_GIN_RUN_MICROBENCH=1 python experimental/uccl_gin/tests/python/test_microbench.py
UCCL_GIN_RUN_CONTEXT_SMOKE=1 python experimental/uccl_gin/tests/python/test_context.py
```
