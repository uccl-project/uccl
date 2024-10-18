# Unified CCL

### Building the system

```
make
```

### Run TCP testing

```
cd afxdp; make -j

# On both server and client
./setup.sh ens6 4 9001 tcp

# On server
./sync.sh 172.31.30.246
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
./sync.sh 172.31.30.246
sudo ./server_main

# On client
sudo ./client_main
```

### Debugging the transport stack

```
sudo ./transport_test --logtostderr=1 --vmodule=transport=1,util_afxdp=1
sudo ./transport_test --client --logtostderr=1 --vmodule=transport=1,util_afxdp=1
```
