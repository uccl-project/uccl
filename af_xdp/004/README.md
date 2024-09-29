# 004

I'm using an Intel 10G NIC for these tests.

How many 100 byte packets can we _theoretically_ send over a 10gbit NIC per-second?

This is called "line rate", and it's covered really nicely in this article:

https://www.fmad.io/blog/what-is-10g-line-rate

It feels like at 100 byte packets, we might be getting close to the line rate. Let's calculate it from the article.

100 byte UDP payload = 20 byte IPv4 header + 8 byte UDP header + 100 bytes = 124 bytes ethernet payload on the wire:

```
Preamble                          blue      8 bytes       8 B
Payload                           green   124 bytes     132 B
Frame Check Sequence              yellow    4 bytes     136 B
Epilogue                          purple    1 bytes     137 B
Inter Frame Gap                   red      11 bytes     148 B
```

So our 100 byte UDP packet expands to 148 bytes on the wire.

```
10.00e9 bits / (8 bits * 148 bytes) = ~8,445,945
```

So the theoretical maximum number of 100 byte packets we should be able to send over 10G is 8.4M packets per-second.

We're close (~7.5M pps) but we're not quite hitting line rate, even though all cores on the client are sending packets. 

What's up?
