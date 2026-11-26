# Launch geometry and parameter-storage follow-ups

The complete goal remains active: zero regression versus original UCCL, not
native NCCL. These are isolated fixed-build diagnostics on two four-L4 hosts,
FP32, all 21 powers of two from 1 KiB to 1 GiB and both layouts, three sweeps,
20 warmups and 30 timed iterations. Timing is maximum-rank host enqueue through
completion batch means; each row reports its three-run median. No best-per-size
curve is constructed. Original-reference harness differences remain explicit.

At exactly 256-KiB AR, 1024 threads measure 85.09/89.18 us (out/in), versus
74.55 us UCCL reference, and are rejected. 256 threads measure 77.49/71.25 us:
in-place improves, out-of-place regresses. Both pass the existing 144-call mixed
ABI with per-rank alignment/layout differences and two lifetimes. A separate
384-thread screen is pending; no mixed-build curve is treated as admission.

The 128-KiB source-packet experiment for medium persistent AG passes 60
component calls/rank and the 144-call mixed ABI. Its full curves are retained
under ar512_persistent/uccl_ag_packet128/. It worsens the medium path and is not
retained. The one-packet cooperative path and large-message packet size are
unchanged in this screen.

A static SASS/resource inspection of the current persistent kernel found 141
local-memory load/store instructions and a 160-byte stack frame. The read-only
parameter experiment uses CUDA grid-constant kernel parameters and const-reference
helpers. The resulting build reports zero stack frame and 56 registers versus
80; the cooperative kernel also reports zero stack. This is compilation evidence,
not performance evidence. Correctness and uninstrumented full sweeps are required.
CUDA semantics are documented in the [CUDA programming guide](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-c-programming-guide/index.html#grid-constant).


The 384-thread AR screen is now complete. The 144-call mixed ABI and all three
full AR sweeps pass. At 256 KiB, out/in are 69.14/65.93 us versus 74.55 us
reference. All 42 AR diagnostic medians in that fixed build are <= reference;
this is not a combined full-matrix or matched admission.

The grid-constant build removes all 141 static LDL/STL instructions in the
persistent kernel (zero stack; 56 registers). Its 60-call component, 144-call
mixed ABI and three full AG sweeps pass, but timing does not improve: 16-MiB
out/in 725.23/721.62 us and 1-GiB 38.215/38.210 ms. Static code/resource reduction
alone is not evidence of a performance win. This experiment is not retained.

Eight consumers only at 16–32 MiB, retaining the existing large-message grid,
also pass 60 component and 144 ABI calls/rank. Three complete AG sweeps give
16-MiB 721.57/712.57 us and 32-MiB 1374.93/1361.58 us. The 32-MiB regression
rejects this policy. The existing 16-consumer policy remains the control.


Parallel publication of the three independent cooperative reader ACKs retains
both grid barriers and every reader/retirement condition. It passes 60 component
and 144 mixed ABI calls/rank, but does not improve the full AG curve enough:
8-MiB out/in are 357.49/356.08 us versus 355.37 us reference. It is not retained.

The 384-thread AR binary is being completed into a single-library AG/AR/RS
matrix. Earlier isolated results must not be combined into an acceptance curve.
All previously rejected variants and raw measurements remain preserved.
