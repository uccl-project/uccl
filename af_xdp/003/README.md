# 003

The receive side is the bottleneck. Because the packets all come from one IP address, they all get the same hash of source and dest IP address, and are sent to the same receive queue.

To ensure that received packets are evenly distributed across RSS queues, in this version I _lie_ about the source IP address when I construct the raw packet to send over the AF_XDP socket.

Instead of sending the real client source address in the IPv4 header, I set it to 192.168.*.* where the low 16 bits are a counter that I increment with each packet sent, per-send thread.

This way received packets get load balanced across all RSS queues on the 10G NIC (hash of source and dest IP address), and it's much cheaper than buying hundreds of linux machines for testing :)

At 2 send threads, we now see a slightly improved result:

```
received delta 6592579
received delta 6596238
received delta 6602746
received delta 6605822
received delta 6600152
received delta 6601785
received delta 6600495
received delta 6599516
```

Increasing to 32 send threads, the results are now slightly improved, but are starting to hit some sort of limit:

```
sent delta 7536128
sent delta 7526829
sent delta 7533959
sent delta 7528073
sent delta 7533217
sent delta 7528522
sent delta 7532370
```
