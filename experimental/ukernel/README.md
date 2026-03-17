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

## transport development
```
cd experimental/ukernel/src/transport
make clean -f Makefile && make -j$(nproc) -f Makefile

# or from the ukernel root
cd experimental/ukernel
make transport_test
make transport_suite
```

## test transport
```
cd experimental/ukernel/src/transport

./test_transport_main core
./test_transport_main communicator-local

# One-shot suite for same-host IPC transport coverage
CUDA_VISIBLE_DEVICES=5 ./test/run_transport_suite.sh ./test_transport_main

# Manual communicator scenarios
CUDA_VISIBLE_DEVICES=5 ./test_transport_main communicator --role=server --case=basic --exchanger-port 16979
CUDA_VISIBLE_DEVICES=5 ./test_transport_main communicator --role=client --case=basic --exchanger-ip 127.0.0.1 --exchanger-port 16979

CUDA_VISIBLE_DEVICES=5 ./test_transport_main communicator --role=server --case=batch --exchanger-port 16980
CUDA_VISIBLE_DEVICES=5 ./test_transport_main communicator --role=client --case=batch --exchanger-ip 127.0.0.1 --exchanger-port 16980

CUDA_VISIBLE_DEVICES=5 ./test_transport_main communicator --role=server --case=poll-release --exchanger-port 16981
CUDA_VISIBLE_DEVICES=5 ./test_transport_main communicator --role=client --case=poll-release --exchanger-ip 127.0.0.1 --exchanger-port 16981

CUDA_VISIBLE_DEVICES=5 ./test_transport_main communicator --role=server --case=notifier --exchanger-port 16982
CUDA_VISIBLE_DEVICES=5 ./test_transport_main communicator --role=client --case=notifier --exchanger-ip 127.0.0.1 --exchanger-port 16982

./test_transport_main oob-socket
./test_transport_main oob-socket-meta --world-size 4
./test_transport_main oob-uds

# when built with USE_REDIS_OOB=1
./test_transport_main oob-redis
./test_transport_main oob-redis-meta --world-size 4

./test_transport_main utils-host-id

# Optional: force same-node processes onto the UCCL path for transport testing
TRANSPORT_RUN_UCCL=1 CUDA_VISIBLE_DEVICES=5 ./test/run_transport_suite.sh ./test_transport_main

CUDA_VISIBLE_DEVICES=5 python py/test_p2p.py
```

`UHM_HOST_ID_OVERRIDE` can be used to give a process a synthetic host identity during transport testing. The optional same-node UCCL suite uses it automatically.


## device develpment
on Nvidia
```
cd experimental/ukernel/src/device
make clean -f Makefile && make -j$(nproc) -f Makefile

# or from the ukernel root
cd experimental/ukernel
make device_test
make device_bench
```

## test device
```
CUDA_VISIBLE_DEVICES=5 ./test_device_runtime

# bench
CUDA_VISIBLE_DEVICES=5 ./benchmarks/bench_device_fifo
CUDA_VISIBLE_DEVICES=5 ./benchmarks/bench_device_full_fifo
CUDA_VISIBLE_DEVICES=5 ./benchmarks/bench_device_sm_fifo 83
```

## ccl development
```
cd experimental/ukernel/src/ccl
make clean -f Makefile && make -j$(nproc) -f Makefile

# or from the ukernel root
cd experimental/ukernel
make ccl_test
```

## test ccl backends
```
cd experimental/ukernel/src/ccl
CUDA_VISIBLE_DEVICES=5 ./test_ccl_persistent_backend
CUDA_VISIBLE_DEVICES=5 ./test_ccl_copy_engine_backend
```

## test ccl planner and executor
```
cd experimental/ukernel/src/ccl
./test_ccl_main ccl-plan
./test_ccl_main ccl-exec

# rdma allgather smoke test
./test_ccl_main ccl-rdma-ag --role=server
UHM_EXCHANGER_SERVER_IP=127.0.0.1 ./test_ccl_main ccl-rdma-ag --role=client
```

## Transport Benchmark

Build the transport benchmark:
```bash
cd experimental/ukernel
make bench
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `UHM_EXCHANGER_SERVER_IP` | Exchanger server IP | `0.0.0.0` |
| `UHM_EXCHANGER_SERVER_PORT` | Exchanger server port | 6979 |

### Usage

The benchmark requires two processes (sender and receiver) running simultaneously. Lower rank acts as sender, higher rank acts as receiver.

The transport runtime now has two runtime paths:
- cross-host traffic uses UCCL
- same-host cross-rank traffic uses IPC automatically

**Using the default transport runtime:**
```bash
# Terminal 1 (Receiver - 先启动)
./bench_transport --rank 1 --peer-rank 0 --gpu-id 0 --msg-size 1048576 --iterations 1000 --ip 192.168.2.243 --port 6979
# Terminal 2 (Sender - 后启动)
./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 1000 --ip 192.168.2.243 --port 6979
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--rank` | Current process rank (0 or 1) | Required |
| `--peer-rank` | Peer process rank | Required |
| `--gpu-id` | GPU device ID | 0 |
| `--msg-size` | Message size in bytes | 1024 |
| `--iterations` | Number of test iterations | 1000 |
| `--warmup` | Number of warmup iterations | 100 |
| `--ip` | Local IP address | 127.0.0.1 |
| `--port` | Listen port | 6979 |

### Output Metrics

The benchmark reports three test results:

1. **Latency Test (Ping-Pong)**
   - Measures round-trip latency
   - Reports: min, p50, p90, p99, max (microseconds)

2. **Throughput Test (One-way)**
   - Measures unidirectional send bandwidth
   - Reports: Total data (GB), Time (sec), Throughput (GB/s, Gbps), Messages/sec

3. **Bidirectional Throughput Test**
   - Measures simultaneous send/receive bandwidth
   - Reports: Total data (GB), Time (sec), Throughput (GB/s, Gbps)

### Examples

**Test with 1MB messages:**
```bash
# Terminal 1
./bench_transport --rank 0 --peer-rank 1 --msg-size 1048576

# Terminal 2
./bench_transport --rank 1 --peer-rank 0 --msg-size 1048576
```
