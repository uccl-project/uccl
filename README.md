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
./sync.sh 172.31.19.147
./server_tcp_main

# On client
./client_tcp_main -a 172.31.22.249
```

### Run AFXDP testing

```
cd afxdp; make -j

# On both server and client
./setup.sh ens6 1 3498 afxdp

# On server
./sync.sh 172.31.19.147
sudo ./server_main

# On client
sudo ./client_main
```

To output VLOG(3) for debugging, `sudo GLOG_v=3 GLOG_logtostderr=1 ./server_main`