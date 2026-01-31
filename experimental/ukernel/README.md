# UKernel
Ultra Unified and Fine-grained Kernel

```
sudo apt-get update
sudo apt-get install -y libelf-dev
```

## transport/runtime develpment:
on AMD
```
cd experimental/ukernel
make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
```

on Nvidia
```
cd experimental/ukernel
make clean -f Makefile && make -j$(nproc) -f Makefile
```
> The CUDA build container uses glog 0.5 (libglog.so.0), but many host systems use glog 0.6 (libglog.so.1), causing runtime linking errors.

## test transport communicator
```
CUDA_VISIBLE_DEVICES=5 ./test_main --role=server
CUDA_VISIBLE_DEVICES=5 ./test_main --role=client

# notifier version transport
CUDA_VISIBLE_DEVICES=5 ./test_main --role=server-notifier
CUDA_VISIBLE_DEVICES=5 ./test_main --role=client-notifier
```


## compute develpment
on AMD
```
cd experimental/ukernel/src/compute
make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
```

on Nvidia
```
cd experimental/ukernel/src/compute
make clean -f Makefile && make -j$(nproc) -f Makefile
```

## test compute
```
CUDA_VISIBLE_DEVICES=5 ./test_persistent

# bench
CUDA_VISIBLE_DEVICES=5 ./benchmarks/bench_fifo
CUDA_VISIBLE_DEVICES=5 ./benchmarks/bench_full_fifo
CUDA_VISIBLE_DEVICES=5 ./benchmarks/bench_sm_fifo 83
```