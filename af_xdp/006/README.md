# 006

In this version, I notice that on the sender there's a lot of CPU spent in *ksoftirqd* for 8 threads.

```
top - 12:34:12 up 3 days, 23:59,  4 users,  load average: 36.26, 19.75, 12.08
Tasks: 947 total,   9 running, 938 sleeping,   0 stopped,   0 zombie
%Cpu(s):  0.0 us,  0.3 sy, 80.0 ni,  0.0 id,  0.0 wa,  0.0 hi, 19.7 si,  0.0 st
MiB Mem :  64256.0 total,  57675.0 free,   4526.3 used,   2054.7 buff/cache
MiB Swap:  20479.5 total,  20479.5 free,      0.0 used.  56313.7 avail Mem

    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
  33225 root      26   6  817224 545536   4352 S  2569   0.8  52:35.60 client
     43 root      20   0       0      0      0 R  79.4   0.0 494:00.20 ksoftirqd/4
     61 root      20   0       0      0      0 R  79.4   0.0 496:43.26 ksoftirqd/7
     15 root      20   0       0      0      0 R  79.1   0.0 496:35.97 ksoftirqd/0
     25 root      20   0       0      0      0 R  79.1   0.0   5207:16 ksoftirqd/1
     31 root      20   0       0      0      0 R  79.1   0.0 539:14.71 ksoftirqd/2
     37 root      20   0       0      0      0 R  79.1   0.0 496:35.48 ksoftirqd/3
     49 root      20   0       0      0      0 R  79.1   0.0 493:45.00 ksoftirqd/5
     55 root      20   0       0      0      0 R  79.1   0.0 496:53.91 ksoftirqd/6
  33259 glenn     26   6   23880   4864   3328 R   1.0   0.0   0:00.83 top
```

I've read that by using polling you can avoid the soft irqs and have the driver do work in the userspace program.

I'm not sure if this applies to sending packets or just receive packets, but let's try it...

I also tried increasing the size of the tx queue in the ring buffer, but that didn't work.

Results:

```
sent delta 7534649
sent delta 7526574
sent delta 7562380
sent delta 7522961
sent delta 7536313
sent delta 7532544
sent delta 7532371
sent delta 7530179
sent delta 7527757
sent delta 7530456
sent delta 7541331
sent delta 7525405
sent delta 7534757
sent delta 7535870
sent delta 7526928
```

No improvement.
