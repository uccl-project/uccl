# CCL

This directory contains the current collective communication runtime prototype
for `experimental/ukernel`.

## What Is Here

- `plan.*`: collective algorithm planning IR
- `selector.*`: lowering from collective plan to execution ops
- `executor.*`: runtime scheduler and background progress thread
- `backend/`: device and transport backend implementations
- `test/`: component tests and multiprocess collective tests

The current implementation focuses on:

- `AllReduce` with `Ring`
- `AllToAll` with `Pairwise`

## Build

From this directory:

```bash
cd experimental/ukernel/src/ccl
make clean -f Makefile
make -j$(nproc) all -f Makefile
```

If your environment is different, you can override build variables:

```bash
make -j$(nproc) all -f Makefile \
  CUDA_HOME=/usr/local/cuda \
  CONDA_LIB_HOME=/usr/lib \
  SM=80
```

## Run Tests

Run component tests:

```bash
./test_components
```

Run the full test suite:

```bash
make test -f Makefile
```

This runs:

- `test_components`
- `test_multiprocess_collective` through `torchrun`

## Run Multiprocess Collectives

The helper script is:

- `test/run_ccl_multiprocess_suite.sh`

Run it with:

```bash
bash ./test/run_ccl_multiprocess_suite.sh ./test_multiprocess_collective
```

By default it launches:

- `allreduce`
- `alltoall`
- `4` local ranks with `torchrun`
- `transport=auto`
- `1 MiB` per rank
- `64 KiB` tile size
- `2` flows

## Useful Script Knobs

Edit the variables at the top of
[run_ccl_multiprocess_suite.sh](/Users/jacelau/code/opencode/uccl/experimental/ukernel/src/ccl/test/run_ccl_multiprocess_suite.sh):

- `NPROC_PER_NODE`
- `NNODES`
- `NODE_RANK`
- `MASTER_ADDR`
- `TORCHRUN_MASTER_PORT`
- `EXCHANGER_PORT_BASE`
- `GPU_IDS`
- `TRANSPORT`
- `BYTES_PER_RANK`
  The current multiprocess test harness emulates float tensors, so this value
  should be a multiple of `world_size * sizeof(float)`.
- `TILE_BYTES`
- `NUM_FLOWS`

## Manual Torchrun Example

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --no-python \
  --nproc-per-node 4 \
  --nnodes 1 \
  --node-rank 0 \
  --master-addr 127.0.0.1 \
  --master-port 29500 \
  ./test_multiprocess_collective \
  --collective allreduce \
  --transport auto \
  --bytes-per-rank 1048572 \
  --tile-bytes 65536 \
  --num-flows 2 \
  --exchanger-ip 127.0.0.1 \
  --exchanger-port 29600
```

## Current Runtime Model

- `submit()` enqueues one collective plan
- `Executor` owns a background progress thread
- one queued collective is executed at a time
- ops become ready when all dependency counts reach zero
- backend completions unlock successor ops

## Notes

- This code currently assumes one active collective is progressed at a time by
  the executor thread.
- The local test script is oriented around single-node `torchrun`.
- Real backend compilation depends on CUDA, RDMA, and transport dependencies
  being available on the machine.
