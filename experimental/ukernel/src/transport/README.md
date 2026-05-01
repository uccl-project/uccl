# Transport

## Build

```bash
cd experimental/ukernel/src/transport
make clean
make -j$(nproc)
```

## Test

Run unit tests:

```bash
make test-unit
```

### OOB Testing

Build the unit test binary:

```bash
cd experimental/ukernel/src/transport
make -j$(nproc) test-unit
```

Run OOB serializer round-trip tests (`serialize_object` / `deserialize_object`):

```bash
./test_transport_unit oob-serde
```

Run `ShmExchanger` tests (`put/get/wait`, relay-state helpers):

```bash
./test_transport_unit oob-shm
```

Run `SocketExchanger` and `HierarchicalExchanger` tests:

```bash
./test_transport_unit oob-socket
./test_transport_unit oob-socket-meta --world-size 4
```

Recommended OOB-only check on a server:

```bash
./test_transport_unit oob-serde && \
./test_transport_unit oob-shm && \
./test_transport_unit oob-socket && \
./test_transport_unit oob-socket-meta --world-size 4
```

If you want only socket-layer behavior in manual debugging, use different
`UHM_OOB_NAMESPACE` values between different node groups so same-host shared
memory does not mask cross-node relay behavior.

Optional manual SHM OOB case:

```bash
./test_transport_unit oob-shm
```

Run integration tests:

```bash
make test-integration
```

This only builds `test_transport_integration` and runs the local single-process smoke case.

Run everything:

```bash
make test
```

## Manual Cases

Local integration smoke:

```bash
./test_transport_integration communicator-local
CUDA_VISIBLE_DEVICES=5 ./test_transport_integration communicator --role=server --case=ipc-buffer-meta --transport ipc --exchanger-port 16980
CUDA_VISIBLE_DEVICES=6 ./test_transport_integration communicator --role=client --case=ipc-buffer-meta --transport ipc --exchanger-ip 127.0.0.1 --exchanger-port 16980
```

Manual server/client cases:

```bash
CUDA_VISIBLE_DEVICES=5 ./test_transport_integration communicator --role=server --case=exchange --exchanger-port 16979
CUDA_VISIBLE_DEVICES=6 ./test_transport_integration communicator --role=client --case=exchange --exchanger-ip 127.0.0.1 --exchanger-port 16979
```

`exchange` is the normal data-path case: connect, accept, send/recv, and verify
payload.

Available communicator cases:

```bash
exchange
ipc-buffer-meta
```

Common manual runs:

```bash
CUDA_VISIBLE_DEVICES=5 ./test_transport_integration communicator --role=server --case=ipc-buffer-meta --transport ipc --exchanger-port 16982
CUDA_VISIBLE_DEVICES=6 ./test_transport_integration communicator --role=client --case=ipc-buffer-meta --transport ipc --exchanger-ip 127.0.0.1 --exchanger-port 16982
```

`ipc-buffer-meta` is a metadata-only IPC case. It does not establish a transport
send/recv path; it only checks `notify_ipc_buffer -> wait_ipc_buffer ->
resolve_ipc_buffer_pointer`.

Optional full multi-process suite:

```bash
make suite
TRANSPORT_RUN_UCCL=1 make suite
```

## Notes

- `test-unit` covers memory registry, socket OOB, peer session, host bounce pool, and TCP adapter.
- OOB unit coverage includes `ShmExchanger`, `SocketExchanger`, and `HierarchicalExchanger`.
- `test-integration` is a lightweight smoke target.
- Multi-process communicator checks are manual by default.
- Use `test_transport_integration communicator --role=server|client --case=exchange ...` for normal two-process transport bring-up.
- `oob-shm` is kept as a manual diagnostic case and is not part of the default unit suite.
- For `PreferredTransport::Uccl`, both peers must be RDMA-capable.
