# Unified CCL

### Building the system

```
make
```

### Run TCP testing

```
cd afxdp; make -j

# On both server and client
../setup.sh ens6 1 9000

# On server
../sync.sh 172.31.19.147
./server_tcp

# On client
./client_tcp -a 172.31.22.249
```

### Run AFXDP testing

```
cd afxdp; make -j

# On both server and client
../setup.sh ens6 1 3498

# On server
../sync.sh 172.31.19.147
sudo ./server

# On client
sudo ./client
```
