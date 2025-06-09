# Minimal P2P Prototype

This prototype implements a GPU-direct P2P communication test using CPU Proxy.

## Build
```bash
make            
```

## Run (two nodes)
```bash
./benchmark_remote 0 192.168.0.100 # sender
./benchmark_remote 1 192.168.0.58 # receiver
```
