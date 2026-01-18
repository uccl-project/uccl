# UKernel
Ultra Unified and Fine-grained Kernel

```
sudo apt-get update
sudo apt-get install -y libelf-dev
```

## develpment:
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

## test communicator
```
CUDA_VISIBLE_DEVICES=5 ./test_main --role=server
CUDA_VISIBLE_DEVICES=5 ./test_main --role=client
```


## device develpment
on AMD
```
cd experimental/ukernel/src/device
make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
```

on Nvidia
```
cd experimental/ukernel/src/device
make clean -f Makefile && make -j$(nproc) -f Makefile
```

## test device
```
CUDA_VISIBLE_DEVICES=5 ./test_persistent

# bench
CUDA_VISIBLE_DEVICES=5 ./bench_fifo
CUDA_VISIBLE_DEVICES=5 ./bench_full_fifo
```