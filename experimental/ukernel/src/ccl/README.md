# CCL

`ccl` is the collective layer on top of ukernel transport + device backends.

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

```bash
make test-unit
make test-integration
make test
```

## Integration Binaries

- `test_device_backend`: validates device backend execution path
- `test_transport_backend`: validates transport backend wiring
- `test_multiprocess_collective`: end-to-end multiprocess collectives

## Manual Multiprocess Example

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

- Multiprocess tests require `torchrun` and enough visible local GPUs.
- `--transport uccl` selects the RDMA adapter path via compatibility naming.
