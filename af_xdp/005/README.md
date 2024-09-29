# 005

According to this paper: https://hal.science/hal-04458274v1/document, the lowest latency for AF_XDP is obtained when:

1. Busy polling is disabled
2. Polling is disabled (calling poll from the program to do work, instead of driving via interrupts) -- not sure if this applies to recieving packets only, or both send and receive?
3. Need wakeup is disabled (manually waking up the driver to do work to send packets)

I've followed these recommendations from the first versions of the program, but now we're not actually trying to optimize away nanoseconds, we're trying to get close to line rate.

It's possible that some of these settings are reducing throughput?

In this version, I force to zero copy mode on the AF_XDP, to make absolutely sure we're getting zero copy mode, and enable the need wakeup for sends. Theoretically, this should help.

I also tried decreasing the size of the frame to fit the packets we're sending tightly, increasing the size of the UMEM where sent packets are written, and increase the batch size, so we're sending more packets per-iteration.

Results:

```
sent delta 7506235
sent delta 7533866
sent delta 7535626
sent delta 7525465
sent delta 7550305
sent delta 7530540
```

No improvement!
