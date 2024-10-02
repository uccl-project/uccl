# Unified CCL

### Building the system

```
make
```

### Run TCP testing

```
cd afxdp

# On both server and client
./run.sh ens6 1
sudo ifconfig ens6 mtu 9000

# On server
./server_tcp_ep

# On client
./client_tcp_ep -a 172.31.22.249
```

### Run AFXDP testing

```
cd afxdp

# On both server and client
./run.sh ens6 1

# On server
sudo ./server

# On client
sudo ./client
```
