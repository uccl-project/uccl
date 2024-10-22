# Unified CCL

### Building the system

```
sudo apt install clang llvm libelf-dev libpcap-dev build-essential libc6-dev-i386 linux-tools-$(uname -r) libgoogle-glog-dev -y
make
```

### Run TCP testing

```
cd afxdp; make -j

# On both server and client
./setup.sh ens6 4 9001 tcp

# On server
./sync.sh 172.31.25.5
./server_tcp_main

# On client
./client_tcp_main -a 172.31.18.199
```

### Run AFXDP testing

```
cd afxdp; make -j

# On both server and client
./setup.sh ens6 1 3498 afxdp

# On server
./sync.sh 172.31.25.5
sudo ./server_main

# On client
sudo ./client_main
```

### Debugging the transport stack

Note that any program that leverages util_afxdp no long needs root to use AFXDP sockets.

```
sudo ./afxdp_daemon_main --logtostderr=1
./transport_test --logtostderr=1 --vmodule=transport=1,util_afxdp=1

sudo ./afxdp_daemon_main --logtostderr=1
./transport_test --client --logtostderr=1 --vmodule=transport=1,util_afxdp=1
```
