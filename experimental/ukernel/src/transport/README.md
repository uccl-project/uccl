# Transport

`transport` provides peer path establishment and async data movement APIs for ukernel.

## Transport Modes

- `auto`: same-host -> `ipc`, cross-host RDMA-capable -> `rdma`, otherwise `tcp`
- `ipc`: same-host shared-memory + optional relay path
- `rdma`: lightweight RDMA adapter path
- `tcp`: socket transport fallback path

## Build

```bash
cd experimental/ukernel/src/transport
make clean
make -j$(nproc)
```

## Test

```bash
make test-unit
make test-integration
make test
```

`test-integration` runs the local single-process smoke by default.

## Manual Checks

Local smoke:

```bash
./test_transport_integration communicator-local
```

Two-process exchange:

```bash
# server
./test_transport_integration communicator --role=server --case=exchange --exchanger-port 16979

# client
./test_transport_integration communicator --role=client --case=exchange --exchanger-ip 127.0.0.1 --exchanger-port 16979
```

IPC metadata only:

```bash
# server
./test_transport_integration communicator --role=server --case=ipc-buffer-meta --transport ipc --exchanger-port 16980

# client
./test_transport_integration communicator --role=client --case=ipc-buffer-meta --transport ipc --exchanger-ip 127.0.0.1 --exchanger-port 16980
```

## Suite

```bash
make suite
TRANSPORT_RUN_RDMA=1 make suite
```

Backward-compatible env alias:

- `TRANSPORT_RUN_UCCL=1` is accepted and mapped to `TRANSPORT_RUN_RDMA=1`.

## Notes

- For `PreferredTransport::Rdma` and `PreferredTransport::Uccl`, both peers must be RDMA-capable.
- `uccl` remains available as a legacy adapter selection.
- `oob-shm` remains a manual diagnostic case, not default unit coverage.
