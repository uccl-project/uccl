# Unified CCL

### Building the system

```
make
```

### Run TCP testing

```
cd afxdp

# On both server and client
./run.sh ens6 1 9000

# On server
./server_tcp

# On client
./client_tcp -a 172.31.22.249
```

### Run AFXDP testing

```
cd afxdp

# On both server and client
./run.sh ens6 1 3498

# On server
sudo ./server

# On client
sudo ./client
```
