# CCL logical op model

How collectives are expressed as a DAG of behavior-level ops, and how
execution details stay out of that layer. This is the target model for
the algorithm / lowering / executor split; the current implementation
still uses `(ExecOpKind + Cmd flags)` for the fused forms (see the
mapping section).

## Layering principle

- **Logical layer** (what algorithm code writes): behavior only. An op
  states what happens — place data, reduce, notify, wait — and in what
  order, with data/peer/tag as parameters.
- **Execution layer** (executor/backends at enqueue time): decides
  *how* — transport channel (IPC / RDMA / proxy), device vs host task,
  signal delivery (shm ring / RDMA write-with-imm / device flag), wait
  mechanism (tag map / imm value / flag poll). None of these appear in a
  logical op's name or definition.

Example: `putsignal` is the same logical op whether the execution layer
implements it over an IPC shm ring (same host), an RDMA write-with-imm
(cross node), or a device flag (hardware without a GPU-mapped ring).
`reduceput` is the same logical op whether executed as one device
reduce+copy kernel or as a device reduce followed by a host-posted
proxy put.

## One-sided data flow

Peer data flow is one-sided. `put` places data directly into the
destination buffer — a local buffer role (staging, local AG copy) or a
peer's buffer — and the receiving side has **no `recv` op**: for a peer
put, the receiver's DAG simply has a `wait` for the sender's data-ready
notification before the op that consumes the buffer. A two-sided
`send`/`recv` vocabulary was considered and rejected:

- a `recv` is executionally identical to a `wait` (both are host waits
  on the same notification channels — tag map / imm / flag); the only
  difference would be the *meaning* of the tag, not the op;
- it implies an active receive that does not exist — the data is placed
  by the sender's one-sided put.

`put` is the one-sided placement verb and covers both local and peer
movement — the destination (local buffer role or peer) is a parameter,
like bytes and offsets. `copy` is not a separate op: local and peer
placement are the same asynchronous behavior modulo the destination.
The transport mechanism underneath (device copy, RDMA write, IPC write,
proxy post) is an execution detail.

## Naming convention

Action-sequence concatenation. `put`, `reduce`, `signal` are the verbs
that compose; a fused op is the sequence of actions it performs, in
order. `wait` (notification) is standalone. No execution word (IPC /
RDMA / proxy / imm / flag / device) appears in a logical op name.

## Op catalog

| op | behavior (行文) | parameters | execution layer may implement via |
|---|---|---|---|
| `put` | place data into a destination — local buffer role (staging, local AG copy) or peer buffer — asynchronously, one-sided | dst (local role or peer), src role, offsets, bytes | device copy / IPC / RDMA / proxy put |
| `reduce` | local reduction | redop, dtype, src/dst | device reduce kernel |
| `signal` | notify a peer (data-ready / copies-done handshake) | dst_peer, tag | IPC ring / RDMA signal QP |
| `wait` | wait for a peer notification (data-ready or handshake) | src_peer, tag | tag map / imm value / flag poll |
| `putsignal` | place data to a peer, then notify once it lands | + signal tag | IPC shm ring / RDMA write-with-imm / device flag |
| `reduceput` | reduce, then place the result (peer or local dst) | + dst | device reduce+copy kernel / reduce + proxy put |
| `reduceputsignal` | reduce, place to a peer, notify on landing | + signal tag | fused reduce+copy+flag task / reduce+copy + host signal |

The receive side of any fused form is expressed by prefixing `wait`:
`wait` → `reduce`, `wait` → `put`, `wait` → `reduceput`, etc. The fused
forms are peer-oriented (a signal needs a receiver); a local put never
fuses with `reduce` as a logical op — `reduce` writes its result to its
destination, and any further local movement is a separate `put` node
(merging them in a device kernel is an execution-layer optimization, not
a distinct behavior). Group counts (one `signal` per G tiles, `wait`
counting G arrivals) and chunk counts are parameters of the op, not
separate ops.

Note on legacy `*Copy*` names: the current implementation's
`kCmdFlagReduceCopy` / `kCmdFlagCopySignal` both copy to a *peer*
buffer, so under this model they are `reduceput` / `putsignal`, not
`copyreduce*`.

## Synchronization convention (wait semantics)

`wait` is behavior ("wait for notification T from peer P"); the
*mechanism* is an execution-layer decision. This project's contract is
**CPU orchestration**:

- The GPU only computes; all collective-internal waiting is done on the
  host — host polls, host signals, host-enqueued tasks. A `wait`
  completes on the host, and only then is the dependent reduce/put
  enqueued to the device worker. The GPU never spins polling for data
  arrival.
- Non-data-wait exceptions (not `wait` semantics):
  - D2H ring producer backpressure in the proxy path — the device
    kernel spins until the host consumes the notify slot (flow control,
    not a data wait).
  - User-stream output dependency — `cudaStreamWaitValue` on the
    completion flag (stream-level sync, not SM polling).

## Implementation status

The plan-level DAG (`TiledOp`) and the backend command (`Cmd`) both carry
`LogicalOpKind` directly; the fused forms are first-class kinds and the
old kind+flag combinations are gone. The remaining `Cmd` flags are
execution-side orthogonals chosen at enqueue time:

- `kCmdFlagImmWait` — a `wait` matches RDMA write-with-imm values;
- `kCmdFlagRdmaFusedProxy` — a `reduce` notifies the host proxy via the
  D2H ring (its linked `putsignal` is posted by the proxy);
- `kCmdFlagCopySignal` — a device `put` writes the peer completion flag.

Lowering picks the fusion granularity per hop and emits the logical
kinds; the executor fills channels/encodings (IPC vs RDMA imm vs flag)
at enqueue time. No per-op fusion metadata (`put_to_sig` etc.) remains.

## Decisions recorded

- Data flow is one-sided; there is no `recv` op (and no `send`/`recv`
  symmetry). The receiving side is `wait` + the consuming op.
- `copy` is not a separate op: `put` covers local and peer placement,
  the destination (local role or peer) being a parameter.
- `reduceputsignal` is kept as the full fused pipeline op; the execution
  layer may decompose it (e.g. `reduceput` + `signal`) when that is
  cheaper on a given path.
- `wait`'s mechanism (tag map / imm / flag) is an execution decision,
  recorded here as CPU orchestration.

## Execution machinery per op (执行机)

A logical op is a DAG node; the execution layer decomposes it into one or
more backend operations (device task, transport put/signal, host wait).
The decomposition is chosen at **lowering time** from topology and
hardware — it is not part of the op. Crucially, lowering picks the fusion
granularity so that every emitted DAG node maps to **exactly one backend
operation**: a hop that can run as a single device reduce+copy is one
`reduceput` node, while a cross-node hop that must go through the proxy
is emitted as `reduce` + `putsignal` (the reduce notifies the host, the
proxy later posts the putsignal). No node aggregates multiple backend
ops, so completion stays per-node.

| logical op | execution decomposition | backend(s) | completion observed via |
|---|---|---|---|
| `put` (local) | device copy task (or host copy) | DeviceBackend (`CollCopy`) | device drain |
| `put` (peer, IPC) | shm/D2D put into peer buffer | TransportBackend → Communicator IPC | tpt drain (send done) |
| `put` (peer, RDMA) | one-sided RDMA write | TransportBackend → RdmaAdapter | tpt drain |
| `put` (peer, proxy) | host proxy posts RDMA put after device notify | RdmaFusedProxy → TransportBackend | tpt drain |
| `reduce` | device reduce kernel | DeviceBackend (`CollReduce`) | device drain |
| `signal` | write tag to peer shm ring / RDMA signal QP send | SignalBackend → Communicator | signal drain |
| `wait` | host-side match: 64-bit tag map / imm value / flag poll | SignalBackend → Communicator | notification arrival |
| `putsignal` | put + deliver tag on landing: IPC shm ring (64-bit) / RDMA write-with-imm (epoch-encoded) / device flag | TransportBackend put-signal or DeviceBackend (`CollPut`) | tpt / device drain; receiver's `wait` matches the tag |
| `reduceput` (direct) | one device kernel: reduce + copy to peer buffer | DeviceBackend (`CollReduce` + reduce-copy) | device drain |
| `reduce` + `putsignal` (proxy path) | device reduce + D2H ring notify (the `reduce` node), then host proxy posts the RDMA putsignal | DeviceBackend + RdmaFusedProxy + TransportBackend | device drain for `reduce`; tpt drain for `putsignal` |
| `reduceputsignal` | one device task: reduce + copy to peer + write completion flag on landing | DeviceBackend (`CollReduce` + reduce-copy + flag) | device drain |

Notes:

- The proxy's D2H ring notify is the hand-off between the `reduce` and
  `putsignal` nodes, not a third backend op.
- If a hop's signal cannot be fused into the same backend op, lowering
  emits the finer nodes (`reduceput` + `signal`) instead of
  `reduceputsignal`.
- `wait` never spins the GPU: the host completes the notification and
  only then enqueues the dependent op (CPU orchestration, see above).
- The current implementation's proxy `synthetic Put` + locally-completing
  `Signal` nodes map to `putsignal` under this model: the notification is
  part of `putsignal`, and the sender-side local completion bookkeeping
  moves into it.
