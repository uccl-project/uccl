on AMD
```
cd experimental/eccl/benchmarks
make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
```

on Nvidia
```
cd experimental/eccl/benchmarks
make clean -f Makefile && make -j$(nproc) -f Makefile
```