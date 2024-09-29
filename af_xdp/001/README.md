# 001

Let's send millions of packets per-second.

Using only *one core* on a 5 year old bare metal linux box, I'm able to send ~6.1 million 100 byte UDP packets per-second with AF_XDP.

To build, first make sure you have Linux setup according to the instructions here: https://mas-bandwidth.com/xdp-for-game-programmers/

Next, edit the source to specify your network interface, and your own ethernet addresses and IP addresses for your client and server (use ifconfig to see them).

In client.c:

```c
const char * INTERFACE_NAME = "enp8s0f0";

const uint8_t CLIENT_ETHERNET_ADDRESS[] = { 0xa0, 0x36, 0x9f, 0x68, 0xeb, 0x98 };

const uint8_t SERVER_ETHERNET_ADDRESS[] = { 0xa0, 0x36, 0x9f, 0x1e, 0x1a, 0xec };

const uint32_t CLIENT_IPV4_ADDRESS = 0xc0a8b779; // 192.168.183.121

const uint32_t SERVER_IPV4_ADDRESS = 0xc0a8b77c; // 192.168.183.124
```

In server.c:

```c
const char * INTERFACE_NAME = "enp8s0f0";
```

Run the server:

`make && sudo ./server`

Run the client:

`make && sudo ./client`

You should see this on the server:

```
received delta 6119827
received delta 6117111
received delta 6065739
received delta 6010190
received delta 6103727
received delta 6115668
received delta 6109344
received delta 6113992
received delta 6098613
received delta 6126000
received delta 6127398
received delta 6112895
received delta 6115302
received delta 6119923
received delta 6115348
received delta 6119834
```

To reproduce this result, you'll need a 10G NIC and a 10G router between two bare metal linux machines running linux kernel 6.5+.

I'm using the Intel x540 T2 10G NIC ($160 USD each):

https://www.newegg.com/intel-x540t2/p/N82E16833106083

With a Netgear 10G router ($450 USD):

https://www.newegg.com/netgear-xs508m-100nas-7-x-10-gig-multi-gig-copper-ports-1-x-10g-1g-sfp-and-copper/p/N82E16833122954

You could also try this on Google Cloud, but I doubt you'll get the same performance as bare metal.

ps. If you do try this on Google Cloud, make sure you use send packets between VMs using _internal addresses_ (!!!)
