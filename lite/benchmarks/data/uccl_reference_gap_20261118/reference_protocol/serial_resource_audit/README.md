# Serial-kernel resource audit (read-only)

The current fixed output-overlap/ordered candidate has a generic stagedCollective
kernel with runtime AG/AR/RS selection. CUDA resource metadata reports 224 bytes
of stack for its nonpacket kernel and 416 bytes for the packet kernel. The frozen
README RS binary's specialized hierarchicalReduceScatterSerial kernel reports
zero stack. Registers are respectively 40, 48, and 43; this is not register-spill
proof. Nonpacket disassembly contains local-memory loads/stores. No profiler or
GPU work was run for this audit, and the active paired test was not modified.

This supports testing compile-time operation specialization of the same existing
collective bodies, which may let the compiler remove unrelated branches and local
state. It is a hypothesis, not a measured improvement. Keep original synchronization,
packets, data path, input/output alignment support and bank lifetimes. Test this
only after the current paired run and queued original-candidate phase profiles.

The initial RS paired round separately shows a 512-KiB serial-path gap near
15–19 percent while the adjacent 1-MiB path uses mediumScatterStreamed. Extending
the latter path to exactly 512 KiB with its existing two-bank protocol is another
isolated candidate, not an accepted route change. Do not mix both hypotheses in
one experiment or discard the current complete baseline/negative measurements.
