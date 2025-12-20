# ECCL

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