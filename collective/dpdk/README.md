# UCCL DPDK Transport Layer

A high-performance network transport layer built on DPDK (Data Plane Development Kit) for unified collective communication.

## TODO

- [ ] Improve `PacketBuf` allocation strategy to reduce allocation overhead and improve memory locality.
- [ ] Run and document performance tests on CloudLab xl170 nodes.
- [ ] Run and document performance tests on CloudLab d6515 nodes.

## Overview

UCCL (Unified Collective Communication Library) DPDK Transport is a user-space network transport layer designed for high-throughput, low-latency communication in distributed systems. It leverages DPDK for kernel-bypass networking and provides efficient data plane operations for collective communication patterns.

## Prerequisites

- **DPDK**: version 24.11.3
- **Compiler**: g++-13 or later with C++17 support
- **Libraries**:
  - libdpdk
  - libglog
  - libgflags
- **Linux Kernel**: 5.x or later
- **Hugepages**: configured for DPDK
- **NIC**: two network interfaces are used:
  - One for normal communication, assumed IPs: **11.11.11.39 / 11.11.11.40**
  - One for DPDK, assumed IPs: **172.168.0.1 / 172.168.0.2**, with MAC addresses **6c:92:bf:f3:2e:1a** and **6c:92:bf:f3:83:1e** respectively


## How to Run

###  Install dependencies (Ubuntu/Debian)

```shell
$ su root

# apt update
# apt install -y pkg-config build-essential python3-pip libnuma-dev python3-pyelftools libgoogle-glog-dev libibverbs-dev
# pip3 install meson ninja --break-system-packages

# wget https://fast.dpdk.org/rel/dpdk-24.11.3.tar.xz
# tar -xvf dpdk-24.11.3.tar.xz
# cd dpdk-stable-24.11.3
# meson setup build
# cd build && ninja
# meson install
# ldconfig

# echo 1024 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
# echo 1024 > /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages
```

### Setup DPDK NIC

Intel NIC: Replace `ens44f3` with your NIC

```shell
$ sudo ip addr flush dev ens44f3
$ sudo modprobe vfio-pci
$ sudo dpdk-devbind.py --bind=vfio-pci 0000:b3:00.3
$ dpdk-devbind.py -s
```

### Build & Run

Server：

```shell
$ make && && scp transport_test 11.11.11.40:~/
$ sudo ./transport_test --logtostderr=1 --localip=172.168.0.1 --localmac=6c:92:bf:f3:2e:1a --test=mq
```

Client:

```shell
$ sudo ./transport_test --logtostderr=1 --client --localip=172.168.0.2 --localmac=6c:92:bf:f3:83:1e --serverip=11.11.11.39 --clientip=11.11.11.40 --test=mq
```

### Stop

```shell
$ Ctrl+Z
$ ps aux | grep transport_test | grep -v grep | awk '{print $2}' | xargs sudo kill -9
```

## Results

You will see something like this.

### Server

```shell
$ sudo ./transport_test --logtostderr=1 --localip=172.168.0.1 --localmac=6c:92:bf:f3:2e:1a --test=mq
I1115 16:41:52.916944 3987317 dpdk.h:25] Initializing DPDK with args
EAL: Detected CPU lcores: 64
EAL: Detected NUMA nodes: 2
EAL: Detected shared linkage of DPDK
EAL: Multi-process socket /var/run/dpdk/rte/mp_socket
EAL: Selected IOVA mode 'VA'
EAL: VFIO support initialized
EAL: Using IOMMU type 1 (Type 1)
I1115 16:41:54.179028 3987317 dpdk.h:36] DPDK runs in VA mode.
I1115 16:41:54.179069 3987317 dpdk.h:42] DPDK initialized successfully
I1115 16:41:54.179078 3987317 transport_test.cc:66] Getting port ID for device 6c:92:bf:f3:2e:1a
I1115 16:41:54.179101 3987317 dpdk.h:67] Checking 1 ports
I1115 16:41:54.179117 3987317 dpdk.h:72] Checking port 0 with MAC 6c:92:bf:f3:2e:1a
I1115 16:41:54.179328 3987317 transport.cc:1529] Creating DPDKFactory
I1115 16:41:54.179343 3987317 pmd_port.cc:31] [PMDPORT] [port_id: 0, driver: net_e1000_igb, RXQ: 8, TXQ: 8, l2addr: 6c:92:bf:f3:2e:1a]
I1115 16:41:54.179359 3987317 pmd_port.cc:111] Rings nr: 1
I1115 16:41:54.179447 3987317 pmd_port.cc:179] RSS indirection table (size 128):
0:      0       0       0       0       0       0       0       0
8:      0       0       0       0       0       0       0       0
16:     0       0       0       0       0       0       0       0
24:     0       0       0       0       0       0       0       0
32:     0       0       0       0       0       0       0       0
40:     0       0       0       0       0       0       0       0
48:     0       0       0       0       0       0       0       0
56:     0       0       0       0       0       0       0       0
64:     0       0       0       0       0       0       0       0
72:     0       0       0       0       0       0       0       0
80:     0       0       0       0       0       0       0       0
88:     0       0       0       0       0       0       0       0
96:     0       0       0       0       0       0       0       0
104:    0       0       0       0       0       0       0       0
112:    0       0       0       0       0       0       0       0
120:    0       0       0       0       0       0       0       0
I1115 16:41:54.179688 3987317 pmd_port.cc:218] Initializing TX ring: 0
I1115 16:41:54.179699 3987317 packet_pool.cc:50] [ALLOC] [type:mempool, name:mbufpool1, nmbufs:1023, mbuf_size:1646]
I1115 16:41:54.180552 3987317 pmd_port.cc:230] Initializing RX ring: 0
I1115 16:41:54.180588 3987317 packet_pool.cc:50] [ALLOC] [type:mempool, name:mbufpool2, nmbufs:1023, mbuf_size:1646]
I1115 16:41:54.181296 3987317 pmd_port.cc:239] Promiscuous mode: 0
I1115 16:41:54.274348 3987317 pmd_port.cc:259] Waiting for link to get up...
I1115 16:41:59.276666 3987317 pmd_port.cc:274] [PMDPORT: 0] Link is UP 1000 (AutoNeg) Full Duplex
I1115 16:41:59.276713 3987317 transport.cc:1536] Creating Channels
I1115 16:41:59.276765 3987317 transport.cc:1543] Creating Engines
I1115 16:41:59.278606 3990074 transport.cc:1556] [Engine] thread 0 running on CPU 16
I1115 16:41:59.282670 3990075 transport.cc:1581] [Engine] deser thread 0 running on CPU 17
I1115 16:41:59.362879 3987317 transport.cc:1613] [Endpoint] server ready, listening on port 30000
I1115 16:42:00.181847 3987939 util_dpdk.h:90] rx: 26 rx_dropped: 0 rx_nombuf: 0
I1115 16:42:00.181915 3987939 transport.cc:1953] 
        [Uccl Engine] 
                        [DPDK] [TX] 0 [RX] 0 [POOL] (1023, 0)
I1115 16:42:02.182142 3987939 util_dpdk.h:90] rx: 35 rx_dropped: 0 rx_nombuf: 0
I1115 16:42:02.182204 3987939 transport.cc:1953] 
        [Uccl Engine] 
                        [DPDK] [TX] 0 [RX] 0 [POOL] (1023, 0)
I1115 16:42:04.182394 3987939 util_dpdk.h:90] rx: 46 rx_dropped: 0 rx_nombuf: 0
I1115 16:42:04.182447 3987939 transport.cc:1953] 
        [Uccl Engine] 
                        [DPDK] [TX] 0 [RX] 0 [POOL] (1023, 0)
I1115 16:42:06.182686 3987939 util_dpdk.h:90] rx: 59 rx_dropped: 0 rx_nombuf: 0
I1115 16:42:06.182760 3987939 transport.cc:1953] 
        [Uccl Engine] 
                        [DPDK] [TX] 0 [RX] 0 [POOL] (1023, 0)
I1115 16:42:08.182965 3987939 util_dpdk.h:90] rx: 68 rx_dropped: 0 rx_nombuf: 0
I1115 16:42:08.183032 3987939 transport.cc:1953] 
        [Uccl Engine] 
                        [DPDK] [TX] 0 [RX] 0 [POOL] (1023, 0)
I1115 16:42:09.307632 3987317 transport.cc:1738] [Endpoint] accept from 11.11.11.40:35764
I1115 16:42:09.307762 3987317 transport.cc:1763] [Endpoint] accept: propose FlowID: 0x42a1da7b83406f65
I1115 16:42:09.307977 3987317 transport.cc:1792] [Endpoint] remote IP: 172.168.0.2
I1115 16:42:09.478638 3990074 transport.cc:1403] [Engine] handle_install_flow_on_engine 0
I1115 16:42:09.517925 3990074 transport.cc:1420] [Engine] start RSS probing
I1115 16:42:09.518924 3990074 transport.cc:1487] [Engine] handle_install_flow_on_engine dst_ports size: 64
I1115 16:42:09.518944 3990074 transport.cc:1496] [Engine] install FlowID 0x42a1da7b83406f65: 172.168.0.1(0) <-> 172.168.0.2(0)
I1115 16:42:09.518988 3990074 transport.cc:369] [Flow] RSS probing rsp packet received, ignoring...
I1115 16:42:09.519093 3987317 transport_test.cc:473] Received 0 messages, rtt 7 us
I1115 16:42:10.184691 3987939 util_dpdk.h:90] rx: 74234 rx_dropped: 45663 rx_nombuf: 33
I1115 16:42:10.184710 3987939 transport.cc:1953] 
        [Uccl Engine] 
                Engine 0 Flow 0x42a1da7b83406f65: 172.168.0.1 (0) <-> 172.168.0.2 (0)
                        [CC] pcb:         snd_nxt: 0, snd_una: 0, rcv_nxt: 54965, fast_rexmits: 0, fast_recovers: 0, rto_rexmits: 0
                             cubic_pp[0]: cwnd: 1.00, effective_cwnd: 1, ssthresh: 64.00, last_max_cwnd: 1.00
                             timely:      prev_rtt: 2.00 us, avg_rtt_diff: 0.00 us, rate: 1.00 Gbps
                        [TX] pending msgbufs unsent: 0
                        [RX] ready msgs unconsumed: 0
                        [DPDK] [TX] 56645 [RX] 55527 [POOL] (511, 512)
I1115 16:42:12.184898 3987939 util_dpdk.h:90] rx: 239560 rx_dropped: 45663 rx_nombuf: 33
```

### Client

```shell
$ sudo ./transport_test --logtostderr=1 --client --localip=172.168.0.2 --localmac=6c:92:bf:f3:83:1e --serverip=11.11.11.39 --clientip=11.11.11.40 --test=mq
I1115 16:42:02.881856 215466 dpdk.h:25] Initializing DPDK with args
EAL: Detected CPU lcores: 64
EAL: Detected NUMA nodes: 2
EAL: Detected shared linkage of DPDK
EAL: Multi-process socket /var/run/dpdk/rte/mp_socket
EAL: Selected IOVA mode 'VA'
EAL: VFIO support initialized
EAL: Using IOMMU type 1 (Type 1)
I1115 16:42:04.161296 215466 dpdk.h:36] DPDK runs in VA mode.
I1115 16:42:04.161337 215466 dpdk.h:42] DPDK initialized successfully
I1115 16:42:04.161347 215466 transport_test.cc:66] Getting port ID for device 6c:92:bf:f3:83:1e
I1115 16:42:04.161370 215466 dpdk.h:67] Checking 1 ports
I1115 16:42:04.161398 215466 dpdk.h:72] Checking port 0 with MAC 6c:92:bf:f3:83:1e
I1115 16:42:04.161631 215466 transport.cc:1529] Creating DPDKFactory
I1115 16:42:04.161648 215466 pmd_port.cc:31] [PMDPORT] [port_id: 0, driver: net_e1000_igb, RXQ: 8, TXQ: 8, l2addr: 6c:92:bf:f3:83:1e]
I1115 16:42:04.161665 215466 pmd_port.cc:111] Rings nr: 1
I1115 16:42:04.161757 215466 pmd_port.cc:179] RSS indirection table (size 128):
0:      0       0       0       0       0       0       0       0
8:      0       0       0       0       0       0       0       0
16:     0       0       0       0       0       0       0       0
24:     0       0       0       0       0       0       0       0
32:     0       0       0       0       0       0       0       0
40:     0       0       0       0       0       0       0       0
48:     0       0       0       0       0       0       0       0
56:     0       0       0       0       0       0       0       0
64:     0       0       0       0       0       0       0       0
72:     0       0       0       0       0       0       0       0
80:     0       0       0       0       0       0       0       0
88:     0       0       0       0       0       0       0       0
96:     0       0       0       0       0       0       0       0
104:    0       0       0       0       0       0       0       0
112:    0       0       0       0       0       0       0       0
120:    0       0       0       0       0       0       0       0
I1115 16:42:04.161971 215466 pmd_port.cc:218] Initializing TX ring: 0
I1115 16:42:04.161981 215466 packet_pool.cc:50] [ALLOC] [type:mempool, name:mbufpool1, nmbufs:1023, mbuf_size:1646]
I1115 16:42:04.162890 215466 pmd_port.cc:230] Initializing RX ring: 0
I1115 16:42:04.162914 215466 packet_pool.cc:50] [ALLOC] [type:mempool, name:mbufpool2, nmbufs:1023, mbuf_size:1646]
I1115 16:42:04.163662 215466 pmd_port.cc:239] Promiscuous mode: 0
I1115 16:42:04.256861 215466 pmd_port.cc:259] Waiting for link to get up...
I1115 16:42:09.259245 215466 pmd_port.cc:274] [PMDPORT: 0] Link is UP 1000 (AutoNeg) Full Duplex
I1115 16:42:09.259298 215466 transport.cc:1536] Creating Channels
I1115 16:42:09.259361 215466 transport.cc:1543] Creating Engines
I1115 16:42:09.259714 215644 transport.cc:1556] [Engine] thread 0 running on CPU 16
I1115 16:42:09.263172 215645 transport.cc:1581] [Engine] deser thread 0 running on CPU 17
I1115 16:42:09.341822 215466 transport.cc:1613] [Endpoint] server ready, listening on port 30000
I1115 16:42:09.341941 215466 transport.cc:1671] [Endpoint] connecting to 11.11.11.39:30000
I1115 16:42:09.342502 215466 transport.cc:1691] [Endpoint] connect: receive proposed FlowID: 0x42a1da7b83406f65
I1115 16:42:09.342655 215466 transport.cc:1717] [Endpoint] remote IP: 172.168.0.1
I1115 16:42:09.459901 215644 transport.cc:1403] [Engine] handle_install_flow_on_engine 0
I1115 16:42:09.498687 215644 transport.cc:1420] [Engine] start RSS probing
I1115 16:42:09.552707 215644 transport.cc:1487] [Engine] handle_install_flow_on_engine dst_ports size: 64
I1115 16:42:09.552726 215644 transport.cc:1496] [Engine] install FlowID 0x42a1da7b83406f65: 172.168.0.2(0) <-> 172.168.0.1(0)
I1115 16:42:09.552764 215644 transport.cc:369] [Flow] RSS probing rsp packet received, ignoring...
I1115 16:42:10.165158 215593 util_dpdk.h:90] rx: 52180 rx_dropped: 0 rx_nombuf: 0
I1115 16:42:10.165184 215593 transport.cc:1953] 
        [Uccl Engine] 
                Engine 0 Flow 0x42a1da7b83406f65: 172.168.0.2 (0) <-> 172.168.0.1 (0)
                        [CC] pcb:         snd_nxt: 50996, snd_una: 50484, rcv_nxt: 0, fast_rexmits: 0, fast_recovers: 0, rto_rexmits: 0
                             cubic_pp[0]: cwnd: 8.00, effective_cwnd: 1, ssthresh: 64.00, last_max_cwnd: 1.00
                             timely:      prev_rtt: 2.00 us, avg_rtt_diff: 0.00 us, rate: 1.00 Gbps
                        [TX] pending msgbufs unsent: 511
                        [RX] ready msgs unconsumed: 0
                        [DPDK] [TX] 115851 [RX] 52101 [POOL] (511, 512)
I1115 16:42:12.165382 215593 util_dpdk.h:90] rx: 217453 rx_dropped: 0 rx_nombuf: 0
```

