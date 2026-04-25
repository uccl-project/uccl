# CCL

## Build

```bash
cd experimental/ukernel/src/ccl
make clean
make -j$(nproc)
```

Common override:

```bash
make -j$(nproc) CUDA_HOME=/usr/local/cuda CONDA_LIB_HOME=/usr/lib SM=80
```

## Test

Run unit tests:

```bash
make test-unit
```

Run integration tests:

```bash
make test-integration
```

Run everything:

```bash
make test
```

## Manual Commands

Device backend integration:

```bash
./test_device_backend
```

Transport backend integration:

```bash
bash ./test/integration/run_transport_backend_suite.sh ./test_transport_backend
```

Multiprocess collective integration:

```bash
bash ./test/integration/run_ccl_multiprocess_suite.sh ./test_multiprocess_collective
```

Manual `torchrun` example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --no-python \
  --nproc-per-node 3 \
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

## Notes

- `test-unit` covers planner, lowering, simulator, and executor behavior.
- `test-integration` covers device backend, transport backend, and multiprocess collectives.
- Multiprocess tests require `torchrun` and enough local GPUs.
