# 008

According to https://www.fmad.io/blog/what-is-10g-line-rate I should be able to get ~14.8M 64 byte packets per-second on 10G ethernet.

To get a 64 byte packet at the ethernet layer we need to consider packet overhead:

1. 20 bytes for IPv4 header
2. 8 bytes for UDP header
3. 4 bytes for the ethernet frame check sequence at the end of the packet.

That's a total of 32 bytes of overhead per-packet.

Since we want 64 bytes on the wire per-packet, we need a UDP payload of _32 bytes_.

Results:

```
received delta 10933782
received delta 10957707
received delta 10927147
received delta 10953418
received delta 10955817
received delta 10956795
received delta 10952854
received delta 10928762
```

With a smaller packet size I can send more packets per-second, but I'm not exactly sure why it's not hitting ~14.8M packets per-second.

What do I have to do to hit line rate on a 10G NIC with AF_XDP?

Please email glenn@mas-bandwidth.com if you know!

UPDATE:

After much noodling, the best I can do is:

```
sent delta 1275622
sent delta 1275366
sent delta 1275639
sent delta 1275697
sent delta 1275890
sent delta 1275365
sent delta 1275917
sent delta 1275354
sent delta 1275698
sent delta 1275627
sent delta 1275675
sent delta 1275621
sent delta 1275680
sent delta 1275617
```

With a whole bunch of shit, disabling hyperthreading and trying to pin the IRQ processing to cores -- the basic theory being that my machine has NUMA nodes, and I really want the processing for sends to happen on the same CPU and NUMA node that queued them up in the UMEM.

```console
# disable hyperthreading
echo off | sudo tee /sys/devices/system/cpu/smt/control

# stop and disable the irqbalance service
sudo systemctl stop irqbalance.service
sudo systemctl disable irqbalance.service

# disable numa balancing
echo kernel.numa_balancing=0 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# limit nic to 4 queues
sudo ethtool -L enp8s0f0 combined 4

# pin irqs to cores (in theory...)
echo 1 | sudo tee /proc/irq/135/smp_affinity
echo 2 | sudo tee /proc/irq/136/smp_affinity
echo 4 | sudo tee /proc/irq/137/smp_affinity
echo 8 | sudo tee /proc/irq/138/smp_affinity
echo 10 | sudo tee /proc/irq/139/smp_affinity
```

It's helping a bit _maybe_ but I don't think it's completely working, but I'm also quite a bit over my head here... HELP! :)
