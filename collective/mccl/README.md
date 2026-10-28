# MCCL collective plugin

The MUSA collective path is an external `mcclNet_v11` plugin for the frozen
MCCL `release_musa_5.3.0` ABI. It is separate from the existing NCCL/RCCL
plugins and does not implement the optional `mcclCollNet_v11` interface.

Build and install on a host with the MUSA SDK and RDMA headers:

```bash
cd collective/rdma
make -f Makefile.musa
make -f Makefile.musa install
```

The official MCCL net loader selects the installed library with
`MCCL_NET_PLUGIN=uccl`. The correctness runner also sets `MCCL_NET` to the
plugin name and requires the test output to confirm `UCCL-MCCL-RDMA`:

```bash
python3 collective/mccl/run_mccl_correctness.py \
  --tests-home "$MCCL_TESTS_HOME" \
  --plugin "$(python3 -c 'import uccl; print(uccl.mccl_plugin_path())')"
```

The default matrix runs `all_reduce_perf`, `all_gather_perf`,
`reduce_scatter_perf`, and `alltoall_perf` for two ranks, repeated iterations,
and a bounded message range. A missing binary, timeout, non-zero exit, or
missing plugin confirmation is a failed result. The runner does not claim
MUSA correctness when the MCCL test tree or hardware is unavailable.
