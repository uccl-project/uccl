# Official-main GIN native validation

Source tested: `5129f467141b137a86fbb6c12acfef870a05da92`, based on official
`uccl-project/uccl:main` at `540cc775ba8ab231122f9a98a6d832699f239f70`.

Actual hardware: **2×RTX PRO 6000 Blackwell Server Edition**, 96 GB per GPU,
SM120, CUDA 13.0.88, driver 580.95.05. This is not an RTX 5090 result.
Both native SM120 and SM90 builds returned 0. Every capability/queue child
returned 0; all command fields, uniqueness, FIFO reuse and scalar QUIET counts
were checked. The 100%-utilization/zero-memory idle telemetry reported by the
host before the campaign is retained as a measurement limitation.

Each row is three complete rounds, 100,000 commands/round, 1,024 GPU producers,
FIFO capacity 4,096. These measurements include the host correctness oracle.

| Physical GPU | Other GPU running fixture | Median ms/round | Commands/s | Commands checked | Result |
|---|---|---:|---:|---:|---|
| 0 | No | 7.940 | 12,594,458 | 300,000 | PASS |
| 1 | No | 13.292 | 7,523,322 | 300,000 | PASS |
| 0 | Yes | 8.459 | 11,821,728 | 300,000 | PASS |
| 1 | Yes | 11.928 | 8,383,635 | 300,000 | PASS |

Official main has no standalone GIN API or corresponding fixture, so this
table reports measured throughput without claiming a speedup over a missing
baseline. Cooperative drain speed comparisons belong to the combined fix.

The consumer supplies test completions. No RDMA endpoint, distributed receiver,
NCCL communicator, model forward or HTTP concurrency is exercised. Thor/RTX
5080 model E2E remains unqualified until execution on those actual devices.

Original lock inodes were reacquired after every child and controller exited;
all logs and command receipts were collected. Changes after the tested commit
in this branch are documentation, clang-format 14 formatting, and restoring
the shared GPU runtime header exactly to official main. The local device/FIFO
operations exercised here do not use its DMA-BUF function-pointer typedef.
