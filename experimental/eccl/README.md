# ECCL

```
sudo apt-get update
sudo apt-get install -y libelf-dev
```

## develpment node:

on AMD
```
bash build_and_install.sh rocm6 eccl
```

on Nvidia
```
cd experimental/eccl
make clean -f Makefile && make -j$(nproc) -f Makefile
```
> The CUDA build container uses glog 0.5 (libglog.so.0), but many host systems use glog 0.6 (libglog.so.1), causing runtime linking errors.


## device
on AMD
```
cd experimental/eccl/src/device
make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
```

on Nvidia
```
cd experimental/eccl/src/device
make clean -f Makefile && make -j$(nproc) -f Makefile

CUDA_VISIBLE_DEVICES=5 ./test_persistent
```


## test
```
# test communicator
CUDA_VISIBLE_DEVICES=5 ./test_main --role=server
CUDA_VISIBLE_DEVICES=5 ./test_main --role=client

# test others
CUDA_VISIBLE_DEVICES=5 ./test_main --role=server
```