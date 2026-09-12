# RDMA congestion-control regression tests

The CC send path must reserve each WR's byte credits before posting, return
those exact credits on completion, and drive CQ progress while waiting for the
window. RDMA WRITE send completions do not supply a valid byte length. CC WRs
are posted immediately and signaled individually so deferred batches cannot
change their encoded IDs or delay their RTT start until a later flush.

## Credit accounting (no GPU required)

From `p2p/`:

```sh
c++ -std=c++17 -O2 -pthread -I. tests/test_cc_send_tracker.cc -o /tmp/test_cc_send_tracker
/tmp/test_cc_send_tracker
```

This checks unequal and out-of-order completions, duplicate/unknown IDs,
rejected-post rollback, ID wrap, and concurrent posting/completion.
`make test_cc_send_tracker` also builds the test in a configured build environment.

## Two-node CUDA RDMA regression

Build the P2P extension on both nodes. Configure each node's local GPU and RDMA
interface as usual (`CUDA_VISIBLE_DEVICES`, `UCCL_SOCKET_IFNAME`,
`UCCL_P2P_RDMA_DEV`, and `UCCL_P2P_RDMA_GID_INDEX`). Ensure `libcudart.so` is on
`LD_LIBRARY_PATH`, or set `UCCL_TEST_CUDART` to its full path. PyTorch is not
required. The test forces the IB transport, disables IPC and compression, and
uses the selected GPU index within `CUDA_VISIBLE_DEVICES`.

Start the acceptor, then the initiator (substitute the acceptor's address):

```sh
python tests/test_rdma_cc.py --role acceptor --module ./p2p.abi3.so --cc swift
python tests/test_rdma_cc.py --role initiator --peer <acceptor-address> --module ./p2p.abi3.so --cc swift
```

Use identical workload options on both nodes. The default is three WRITE and
three READ operations, each containing 64 iovs of 128 KiB (8 MiB total).
The original CC integration can stall on the first WRITE; `--cc none` provides
a control. Also run `--cc timely` to exercise the shared accounting path.

Recommended additional shapes:

| Options | Coverage |
| --- | --- |
| `--sizes 4096,8192,16384 --iov-repeats 8` | Unequal small WRs below the deferred-batch threshold |
| `--sizes 8388610 --iov-repeats 2` | Internal chunk splitting, unequal final chunk, and window waits |
| `--sizes 67108864 --iov-repeats 64 --iterations 1` | 4 GiB transfer with internal chunking |

Every destination byte is checked against a nonzero pattern that changes with
iov and iteration. A control handshake prevents buffer reuse before peer
verification. `--mode write` or `--mode read` isolates an operation. The default
`--timeout 60` bounds the entire local subprocess, including native calls and
teardown; exit status 124 indicates a timeout. Increase it for large workloads
or slow environments. `--port` changes the control rendezvous port.

These are correctness/progress tests, not throughput or congestion benchmarks.
They do not validate compressed transfers or other GPU/provider backends.
